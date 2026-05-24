"""
Phase 1: Heatmap Generation Runner (Optimized for GPU & Streaming I/O).

Generates attribution heatmaps for all model-method-image combinations.
This phase creates sorted pixel indices files that are used in Phase 2 for occlusion evaluation.
"""

import numpy as np
import torch
import logging
import cv2
import gc
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from core.gpu_manager import GPUManager
from core.file_manager import FileManager
from core.gpu_utils import prepare_batch_tensor
from attribution.registry import get_attribution_method
from attribution.base import ModelIndependentMethod
from evaluation.occlusion import sort_pixels


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

        # Asynchronous IO pool to prevent disk writes from blocking the GPU
        self.io_pool = ThreadPoolExecutor(max_workers=4)

    def run(self):
        """Generate heatmaps for all model-method-image combinations."""
        dataset_name = self.config.DATASET_NAME

        logging.info(f"Starting Phase 1 - Dataset: {dataset_name}")
        logging.info(
            f"Models: {len(self.config.GENERATING_MODELS)} | "
            f"Methods: {len(self.config.ATTRIBUTION_METHODS)}"
        )

        try:
            # Ensure dataset heatmap directory exists
            heatmap_dir = self.file_manager.get_heatmap_dir(dataset_name)
            self.file_manager.ensure_dir_exists(heatmap_dir)

            # Separate independent and dependent methods dynamically
            independent_methods = []
            dependent_methods = []
            for method_name in self.config.ATTRIBUTION_METHODS:
                method_instance = get_attribution_method(method_name)
                if isinstance(method_instance, ModelIndependentMethod):
                    independent_methods.append(method_name)
                else:
                    dependent_methods.append(method_name)

            # =====================================================================
            # 1. PROCESS MODEL-INDEPENDENT METHODS
            # =====================================================================
            for method_name in independent_methods:
                self.gpu_manager.check_and_throttle()
                try:
                    self._process_streaming(
                        model=None,
                        model_name=None,
                        method_name=method_name,
                        dataset_name=dataset_name
                    )
                except Exception as e:
                    logging.error(f"Error executing Independent Method {method_name}: {e}", exc_info=True)

            # =====================================================================
            # 2. PROCESS MODEL-DEPENDENT METHODS
            # =====================================================================
            if dependent_methods:
                for model_idx, model_name in enumerate(self.config.GENERATING_MODELS, 1):
                    # Load model
                    model = self.model_provider.get_model(model_name)
                    model = model.to(self.config.DEVICE)

                    for method_idx, method_name in enumerate(dependent_methods, 1):
                        self.gpu_manager.check_and_throttle()
                        logging.info(
                            f"[{model_idx}/{len(self.config.GENERATING_MODELS)}] {model_name[:12]} | "
                            f"[{method_idx}/{len(dependent_methods)}] {method_name[:15]}"
                        )

                        try:
                            self._process_streaming(
                                model=model,
                                model_name=model_name,
                                method_name=method_name,
                                dataset_name=dataset_name
                            )
                        except Exception as e:
                            logging.error(f"Error: {model_name}-{method_name}: {e}", exc_info=True)

                    # CRITICAL: Free GPU memory before loading the next model
                    del model
                    gc.collect()
                    if self.config.DEVICE == "cuda":
                        torch.cuda.empty_cache()

            # Wait for all background saving threads to finish before exiting
            self.io_pool.shutdown(wait=True)
            logging.info(f"Heatmaps successfully saved to: {heatmap_dir}")

        except Exception as e:
            logging.error(f"Phase 1 execution failed: {e}")
            self.io_pool.shutdown(wait=False)
            raise

    def _process_streaming(
        self,
        model: Optional[Any],
        model_name: Optional[str],
        method_name: str,
        dataset_name: str
    ):
        """
        Streams batches from the DataLoader directly into the GPU, skipping
        images that have already been processed to allow resume-capability.
        """
        method = get_attribution_method(method_name)

        # Request a fresh dataloader configured to the optimal batch size for this method
        batch_size = self.gpu_manager.get_batch_size(method_name)
        dataloader = self.dataset_handler.get_dataloader(batch_size=batch_size, shuffle=False)

        global_idx = 0
        seen_dirs = set()

        for batch_images, batch_labels in tqdm(dataloader, desc=f"  → Processing {method_name}", dynamic_ncols=True):
            current_batch_size = len(batch_images)

            valid_indices = []
            batch_s_paths = []
            batch_r_paths = []

            # 1. Pre-computation Filter: Check which images actually need processing
            for i in range(current_batch_size):
                img_id = f"image_{global_idx + i:05d}"
                label = batch_labels[i].item()
                category_name = self.dataset_handler.get_category_name(label)

                s_path = self.file_manager.get_sorted_heatmap_path(dataset_name, model_name, method_name, img_id, category_name)
                r_path = self.file_manager.get_regular_heatmap_path(dataset_name, model_name, method_name, img_id, category_name)

                if not s_path.exists() or not r_path.exists():
                    valid_indices.append(i)
                    batch_s_paths.append(s_path)
                    batch_r_paths.append(r_path)

                    # Ensure directory exists once
                    if s_path.parent not in seen_dirs:
                        self.file_manager.ensure_dir_exists(s_path.parent)
                        seen_dirs.add(s_path.parent)

            global_idx += current_batch_size

            # Skip GPU computation if all images in this batch already exist on disk
            if not valid_indices:
                continue

            # 2. Filter tensors to only the ones we need to compute
            sub_batch_images = batch_images[valid_indices]
            sub_batch_labels = batch_labels[valid_indices]

            # 3. GPU Processing
            sub_batch_images = prepare_batch_tensor(sub_batch_images, device=self.config.DEVICE, memory_format=torch.channels_last)
            sub_batch_labels = sub_batch_labels.to(self.config.DEVICE, non_blocking=True)

            if self.config.DEVICE == "cuda":
                with torch.amp.autocast(self.config.DEVICE):
                    heatmaps = method.compute(model, sub_batch_images, sub_batch_labels)
            else:
                heatmaps = method.compute(model, sub_batch_images, sub_batch_labels)

            # 4. Dispatch CPU-bound saving tasks to background threads
            if heatmaps is not None:
                heatmaps_np = heatmaps.cpu().numpy()
                for j, heatmap_np in enumerate(heatmaps_np):

                    if heatmap_np.ndim == 3:
                        heatmap_np = np.mean(heatmap_np, axis=0)

                    # Submit to background thread so GPU can move to next batch instantly
                    self.io_pool.submit(
                        self._process_and_save_outputs,
                        heatmap_np,
                        batch_s_paths[j],
                        batch_r_paths[j]
                    )

            # Thermal check after every processed batch
            self.gpu_manager.check_and_throttle()

    def _process_and_save_outputs(self, heatmap: np.ndarray, sorted_path: Path, regular_path: Path):
        """
        Background worker task: Handles sorting pixels, generating images, and saving to disk.
        """
        try:
            # 1. Sort and save indices
            sorted_indices = sort_pixels(heatmap)
            np.save(sorted_path, sorted_indices)

            # 2. Generate and save colored PNG
            hmap = heatmap.copy()
            hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
            hmap_uint8 = (hmap * 255).astype(np.uint8)

            colormap = getattr(self.config, 'HEATMAP_COLORMAP', 'hot')
            cv_colormap = _COLORMAP_CV2.get(colormap.lower(), cv2.COLORMAP_HOT)

            heatmap_colored = cv2.applyColorMap(hmap_uint8, cv_colormap)
            heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            Image.fromarray(heatmap_rgb).save(regular_path, 'PNG')
        except Exception as e:
            logging.error(f"Background thread failed to save {regular_path.name}: {e}")


def main():
    """Simple main function to run Phase 1."""
    from core._bootstrap import bootstrap_phase1
    runner = bootstrap_phase1()
    runner.run()


if __name__ == "__main__":
    main()