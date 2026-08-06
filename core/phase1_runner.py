"""
Phase 1: Heatmap Generation Runner.

Generates attribution heatmaps for all model-method-image combinations.
This phase creates sorted pixel indices files that are used in Phase 2 for occlusion evaluation.
"""

import numpy as np
import torch
import logging
import cv2
import gc
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Tuple
from matplotlib import pyplot as plt
from core.gpu_manager import GPUManager
from core.file_manager import FileManager
from core.gpu_utils import prepare_batch_tensor, clear_cache_if_needed
from attribution.registry import get_attribution_method
from attribution.base import ModelIndependentMethod

_COLORMAP_CV2 = {
    "hot": cv2.COLORMAP_HOT,
    "jet": cv2.COLORMAP_JET,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "rainbow": cv2.COLORMAP_RAINBOW,
    "turbo": cv2.COLORMAP_TURBO,
}


class Phase1Runner:
    """Handles Phase 1: Heatmap generation for all model-method-image combinations."""

    def __init__(
        self,
        config,
        gpu_manager: GPUManager,
        file_manager: FileManager,
        dataset_handler,
        model_provider
    ):
        self.config = config
        self.gpu_manager = gpu_manager
        self.file_manager = file_manager
        self.dataset_handler = dataset_handler
        self.model_provider = model_provider

    def run(self):
        """Generate heatmaps for all model-method-image combinations."""
        dataset_name = self.config.DATASET_NAME

        logging.info(f"Starting Phase 1 - Dataset: {dataset_name}")
        logging.info(
            f"Models: {len(self.config.GENERATING_MODELS)} | "
            f"Methods: {len(self.config.ATTRIBUTION_METHODS)}"
        )

        try:
            heatmap_dir = self.file_manager.get_heatmap_dir(dataset_name)
            self.file_manager.ensure_dir_exists(heatmap_dir)

            # Load all images once into RAM (avoids re-reading disk per method/model)
            image_label_map = self._load_image_label_map()
            logging.info(f"Loaded {len(image_label_map)} images into memory")

            independent_methods = []
            dependent_methods = []
            for method_name in self.config.ATTRIBUTION_METHODS:
                method_instance = get_attribution_method(method_name)
                if isinstance(method_instance, ModelIndependentMethod):
                    independent_methods.append(method_name)
                else:
                    dependent_methods.append(method_name)

            for method_name in independent_methods:
                self.gpu_manager.check_and_throttle()
                try:
                    self._process_method_batch(
                        model=None,
                        model_name=None,
                        method_name=method_name,
                        image_label_map=image_label_map,
                        dataset_name=dataset_name,
                    )
                except Exception as e:
                    logging.error(f"Error executing Independent Method {method_name}: {e}", exc_info=True)

            for model_idx, model_name in enumerate(self.config.GENERATING_MODELS, 1):

                for method_idx, method_name in enumerate(dependent_methods, 1):
                    self.gpu_manager.check_and_throttle()
                    logging.info(
                        f"[{model_idx}/{len(self.config.GENERATING_MODELS)}] {model_name[:12]} | "
                        f"[{method_idx}/{len(dependent_methods)}] {method_name[:15]}"
                    )

                    # ADD model instantiation HERE (Fresh model per method)
                    model = self.model_provider.get_model(model_name)
                    model = model.to(self.config.DEVICE)

                    try:
                        self._process_method_batch(
                            model=model,
                            model_name=model_name,
                            method_name=method_name,
                            image_label_map=image_label_map,
                            dataset_name=dataset_name,
                        )
                    except Exception as e:
                        logging.error(f"Error: {model_name}-{method_name}: {e}", exc_info=True)
                    finally:
                        # Clean up the model immediately after the method finishes
                        del model
                        gc.collect()
                        if self.config.DEVICE == "cuda":
                            torch.cuda.empty_cache()


            logging.info(f"Heatmaps successfully saved to: {heatmap_dir}")

        except Exception as e:
            logging.error(f"Phase 1 execution failed: {e}")
            raise

    def _load_image_label_map(self) -> Dict[str, Tuple[Tuple[torch.Tensor, str], int]]:
        """Read the dataset once; reuse tensors/paths for every model/method."""
        loader_batch = getattr(self.config, "HEATMAP_BATCH_SIZE", 12)
        dataloader = self.dataset_handler.get_dual_dataloader(batch_size=loader_batch, shuffle=False)

        image_label_map = {}
        global_idx = 0
        for batch_images, batch_labels in dataloader:
            _, attr_batch, path_batch = batch_images
            for attr, path, lbl in zip(attr_batch, path_batch, batch_labels):
                image_label_map[f"image_{global_idx:05d}"] = ((attr, path), lbl.item())
                global_idx += 1
        return image_label_map

    def _process_method_batch(
        self,
        model: Optional[Any],
        model_name: Optional[str],
        method_name: str,
        image_label_map: Dict[str, Tuple[Tuple[torch.Tensor, str], int]],
        dataset_name: str,
    ):
        """Process images for one model-method pair, batching from the in-memory cache."""
        method = get_attribution_method(method_name)
        batch_size = self.gpu_manager.get_batch_size(method_name)

        images_to_process: List[Tuple[torch.Tensor, str]] = []
        labels: List[int] = []
        sorted_paths: List[Path] = []
        regular_paths: List[Path] = []

        for img_id, (data_tuple, label) in image_label_map.items():
            attr, path = data_tuple
            category_name = self.dataset_handler.get_category_name(label)
            s_path = self.file_manager.get_sorted_heatmap_path(
                dataset_name, model_name, method_name, img_id, category_name
            )
            r_path = self.file_manager.get_regular_heatmap_path(
                dataset_name, model_name, method_name, img_id, category_name
            )

            if not s_path.exists() or not r_path.exists():
                images_to_process.append(data_tuple)
                labels.append(label)
                sorted_paths.append(s_path)
                regular_paths.append(r_path)

        seen_dirs = set()
        for p in sorted_paths + regular_paths:
            if p.parent not in seen_dirs:
                self.file_manager.ensure_dir_exists(p.parent)
                seen_dirs.add(p.parent)

        if not images_to_process:
            return

        for i in tqdm(
            range(0, len(images_to_process), batch_size),
            desc=f"  → Processing {len(images_to_process)} images",
            dynamic_ncols=True,
        ):
            if i > 0 and i % (batch_size * 5) == 0:
                self.gpu_manager.check_and_throttle()

            end_idx = min(i + batch_size, len(images_to_process))
            batch_data = images_to_process[i:end_idx]

            attr_list = [t[0] for t in batch_data]
            paths = [t[1] for t in batch_data]

            batch_attr = prepare_batch_tensor(
                attr_list,
                device=self.config.DEVICE,
                memory_format=torch.channels_last,
            )
            batch_labels = torch.tensor(labels[i:end_idx]).to(
                self.config.DEVICE, non_blocking=True
            )

            # Predict target classes using classification cropped images
            if model is not None:
                # Retrieve dynamic transforms assigned to the model
                transform = getattr(model, "transforms", None)
                if transform is None:
                    from data.loader import get_clf_transform
                    transform = get_clf_transform(self.config.DATASET_NAME)

                from PIL import Image
                clf_list = []
                for p in paths:
                    img_pil = Image.open(p).convert("RGB")
                    clf_list.append(transform(img_pil))

                batch_clf = prepare_batch_tensor(
                    clf_list,
                    device=self.config.DEVICE,
                    memory_format=torch.channels_last,
                )

                model.eval()
                with torch.no_grad():
                    logits = model(batch_clf)
                    batch_targets = torch.argmax(logits, dim=1)
            else:
                batch_targets = batch_labels

            # Pass paths to attribution method
            method.current_paths = paths

            if self.config.DEVICE == "cuda":
                with torch.amp.autocast(self.config.DEVICE, enabled=False):
                    heatmaps = method.compute(model, batch_attr, batch_targets)
            else:
                heatmaps = method.compute(model, batch_attr, batch_targets)

            if heatmaps is not None:
                for j, heatmap in enumerate(heatmaps):
                    heatmap_np = heatmap.detach().cpu().numpy()
                    if heatmap_np.ndim == 3:
                        heatmap_np = np.mean(heatmap_np, axis=0)

                    ranking = heatmap_np.flatten() + np.random.uniform(0, 1e-9, size=heatmap_np.size)
                    sorted_indices = np.argsort(ranking).astype(np.uint32)
                    
                    pred_class_id = int(batch_targets[j].item())
                    pred_class_name = self.dataset_handler.get_category_name(pred_class_id)
                    gt_class_id = int(batch_labels[j].item())
                    gt_class_name = self.dataset_handler.get_category_name(gt_class_id)
                    
                    np.save(
                        sorted_paths[i + j],
                        {
                            "sorted_idx": sorted_indices,
                            "shape": heatmap_np.shape,
                            "predicted_class": pred_class_id,
                            "predicted_class_name": pred_class_name,
                            "ground_truth_class": gt_class_id,
                            "ground_truth_class_name": gt_class_name
                        }
                    )
                    self._save_heatmap_png(heatmap_np, regular_paths[i + j])

        self.gpu_manager.check_and_throttle()
        clear_cache_if_needed()

    def _save_heatmap_png(self, heatmap: np.ndarray, path: Path):
        """Save heatmap as PNG image with colormap matching the stage1 reference code."""
        path.parent.mkdir(parents=True, exist_ok=True)
        colormap = getattr(self.config, "HEATMAP_COLORMAP", "jet")
        plt.imsave(str(path), heatmap, cmap=colormap.lower(), vmin=0.0, vmax=1.0)


if __name__ == "__main__":
    from core._bootstrap import bootstrap_phase1
    runner = bootstrap_phase1()
    runner.run()
