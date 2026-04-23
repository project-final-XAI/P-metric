"""
Phase 2: Pre-Generate All Occluded Images.

Loads sorted heatmaps from Phase 1 and generates ALL occluded images
for all combinations (model × method × strategy × level × image).
This pre-generation makes Phase 3 super-fast (only loading and testing).
"""

import os
import numpy as np
import torch
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Tuple, List, Optional, Set
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.gpu_manager import GPUManager
from core.file_manager import FileManager
from core.phase1_runner import Phase1Runner
from data.loader import get_dataloader
from evaluation.occlusion import apply_occlusion_batch


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class Phase2Runner:
    """Handles Phase 2: Pre-generation of all occluded images."""

    def __init__(
        self,
        config,
        gpu_manager: GPUManager,
        file_manager: FileManager,
        model_cache: Dict[str, Any]
    ):
        self.config = config
        self.gpu_manager = gpu_manager
        self.file_manager = file_manager
        self.model_cache = model_cache

        # Caches built once at run() time
        self._sorted_path_cache: Dict[str, Dict[str, Path]] = {}  # (model, method) -> {img_id -> path}
        self._existing_occluded_cache: Dict[str, Set[str]] = {}   # dir_key -> set of filenames

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, get_cached_model_func):
        dataset_name = self.config.DATASET_NAME

        if dataset_name not in self.config.DATASET_CONFIG:
            raise ValueError(
                f"Dataset '{dataset_name}' not found in DATASET_CONFIG. "
                f"Available datasets: {list(self.config.DATASET_CONFIG.keys())}"
            )

        logging.info(f"Starting Phase 2 - Dataset: {dataset_name}")

        # Load dataset once — reused by every combination
        image_label_map = self._load_dataset_images(dataset_name)
        total_images = len(image_label_map)

        # Check / run Phase 1 for missing heatmaps (uses already-known total)
        self._ensure_phase1_complete(dataset_name, total_images, get_cached_model_func)

        # Pre-build sorted-heatmap path lookup for every (model, method) pair
        logging.info("Building sorted heatmap path cache...")
        for model_name in self.config.GENERATING_MODELS:
            for method_name in self.config.ATTRIBUTION_METHODS:
                self._build_sorted_path_cache(dataset_name, model_name, method_name, image_label_map)

        # Generate all occluded images
        total_combinations = (
            len(self.config.GENERATING_MODELS) *
            len(self.config.ATTRIBUTION_METHODS) *
            len(self.config.FILL_STRATEGIES) *
            len(self.config.OCCLUSION_LEVELS)
        )

        with tqdm(total=total_combinations, desc="Phase 2 Progress") as pbar:
            for model_name in self.config.GENERATING_MODELS:
                for method_name in self.config.ATTRIBUTION_METHODS:
                    for strategy in self.config.FILL_STRATEGIES:
                        for level in self.config.OCCLUSION_LEVELS:
                            pbar.set_description(
                                f"{model_name[:12]}/{method_name[:12]}/{strategy}/{level}%"
                            )
                            try:
                                self._generate_occluded_images(
                                    dataset_name, model_name, method_name,
                                    strategy, level, image_label_map
                                )
                            except Exception as e:
                                logging.error(
                                    f"Error: {model_name}-{method_name}-{strategy}-{level}%: {e}"
                                )
                            finally:
                                pbar.update(1)

        logging.info(
            f"Phase 2 complete! Occluded images saved to: "
            f"{self.file_manager.get_occluded_dir(dataset_name)}"
        )

    # ------------------------------------------------------------------
    # Phase 1 completeness check
    # ------------------------------------------------------------------

    def _ensure_phase1_complete(self, dataset_name: str, total_images: int, get_cached_model_func):
        """Check if Phase 1 is complete; run it for missing combinations if needed."""
        missing_items = [
            (model_name, method_name)
            for model_name in self.config.GENERATING_MODELS
            for method_name in self.config.ATTRIBUTION_METHODS
            if len(self.file_manager.scan_sorted_heatmaps(dataset_name, model_name, method_name)) < total_images
        ]

        if missing_items:
            logging.info(f"Running Phase 1 for {len(missing_items)} missing combinations...")
            phase1 = Phase1Runner(self.config, self.gpu_manager, self.file_manager, self.model_cache)
            phase1.run(get_cached_model_func)

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_dataset_images(self, dataset_name: str) -> Dict[str, Tuple[torch.Tensor, int]]:
        """Load all dataset images into memory once."""
        batch_size = getattr(self.config, 'PHASE2_BATCH_SIZE', 256)
        dataloader = get_dataloader(dataset_name, batch_size=batch_size, shuffle=False)
        image_label_map: Dict[str, Tuple[torch.Tensor, int]] = {}
        global_idx = 0

        for batch_images, batch_labels in dataloader:
            for img, lbl in zip(batch_images, batch_labels):
                image_label_map[f"image_{global_idx:05d}"] = (img, lbl.item())
                global_idx += 1

        return image_label_map

    # ------------------------------------------------------------------
    # Sorted-heatmap path cache
    # ------------------------------------------------------------------

    def _build_sorted_path_cache(
        self,
        dataset_name: str,
        model_name: str,
        method_name: str,
        image_label_map: Dict[str, Tuple[torch.Tensor, int]]
    ):
        """
        Pre-compute the sorted-heatmap path for every image under one
        (model, method) pair and store it in self._sorted_path_cache.

        For ImageNet the synset list is read from disk exactly once.
        """
        cache_key = (model_name, method_name)
        if cache_key in self._sorted_path_cache:
            return  # already built

        # --- ImageNet: resolve synset IDs once ---
        synset_ids: Optional[List[str]] = None
        imagenet_mapping = None
        format_fn = None

        if dataset_name == "imagenet":
            try:
                from data.imagenet_class_mapping import get_cached_mapping, format_class_for_llm
                from config import DATASET_CONFIG
                mapping = get_cached_mapping()
                dataset_path = DATASET_CONFIG.get("imagenet", {}).get("path")
                if dataset_path and os.path.exists(dataset_path):
                    synset_ids = sorted(
                        d for d in os.listdir(dataset_path)
                        if os.path.isdir(os.path.join(dataset_path, d))
                    )
                    imagenet_mapping = mapping
                    format_fn = format_class_for_llm
            except Exception as e:
                logging.debug(f"Could not load ImageNet mapping: {e}")

        # --- Build path for every image ---
        path_map: Dict[str, Path] = {}

        for img_id, (_, label) in image_label_map.items():
            resolved: Optional[Path] = None

            # Try ImageNet category-name path first
            if synset_ids is not None and label < len(synset_ids):
                synset_id = synset_ids[label]
                category_name_full = imagenet_mapping.get(synset_id, "")
                if category_name_full:
                    category_name = format_fn(category_name_full)
                    candidate = self.file_manager.get_sorted_heatmap_path(
                        dataset_name, model_name, method_name, img_id, category_name
                    )
                    if candidate.exists():
                        resolved = candidate

            # Fallback: old format without category name
            if resolved is None:
                candidate = self.file_manager.get_sorted_heatmap_path(
                    dataset_name, model_name, method_name, img_id
                )
                if candidate.exists():
                    resolved = candidate

            if resolved is not None:
                path_map[img_id] = resolved
            else:
                logging.warning(
                    f"Missing sorted heatmap for {model_name}/{method_name}/{img_id}"
                )

        self._sorted_path_cache[cache_key] = path_map
        logging.info(
            f"  Cached {len(path_map)}/{len(image_label_map)} heatmap paths "
            f"for {model_name}/{method_name}"
        )

    # ------------------------------------------------------------------
    # Existing-file cache (per output directory)
    # ------------------------------------------------------------------

    def _get_existing_occluded(self, directory: Path) -> Set[str]:
        """
        Return the set of filenames already present in *directory*.
        Result is cached; call _invalidate_occluded_cache() after saves
        if the same directory will be queried again in the same run
        (not required here because we only ever check before writing).
        """
        key = str(directory)
        if key not in self._existing_occluded_cache:
            if directory.exists():
                self._existing_occluded_cache[key] = {p.name for p in directory.iterdir()}
            else:
                self._existing_occluded_cache[key] = set()
        return self._existing_occluded_cache[key]

    def _mark_occluded_saved(self, directory: Path, filename: str):
        """Keep the in-memory cache consistent after a successful save."""
        key = str(directory)
        if key in self._existing_occluded_cache:
            self._existing_occluded_cache[key].add(filename)

    # ------------------------------------------------------------------
    # Core generation logic
    # ------------------------------------------------------------------

    def _generate_occluded_images(
        self,
        dataset_name: str,
        model_name: str,
        method_name: str,
        strategy: str,
        level: int,
        image_label_map: Dict[str, Tuple[torch.Tensor, int]]
    ):
        cache_key = (model_name, method_name)
        sorted_path_map = self._sorted_path_cache.get(cache_key, {})

        images_to_process: List[torch.Tensor] = []
        sorted_indices_list: List[np.ndarray] = []
        occluded_paths: List[Path] = []

        for img_id, (img, _) in image_label_map.items():
            if img_id not in sorted_path_map:
                continue

            occ_path = self.file_manager.get_occluded_image_path(
                dataset_name, model_name, strategy, method_name, level, img_id
            )

            existing = self._get_existing_occluded(occ_path.parent)
            if occ_path.name in existing:
                continue

            sorted_indices = np.load(sorted_path_map[img_id])
            images_to_process.append(img)
            sorted_indices_list.append(sorted_indices)
            occluded_paths.append(occ_path)

        if not images_to_process:
            return

        batch_size = getattr(self.config, 'PHASE2_BATCH_SIZE', 256)
        save_workers = getattr(self.config, 'PHASE2_SAVE_WORKERS', 8)

        for i in range(0, len(images_to_process), batch_size):
            batch_images = images_to_process[i:i + batch_size]
            batch_indices = sorted_indices_list[i:i + batch_size]
            batch_occ_paths = occluded_paths[i:i + batch_size]

            occluded_images = apply_occlusion_batch(
                batch_images,
                batch_indices,
                level,
                strategy,
                image_shape=getattr(self.config, 'OCCLUSION_IMAGE_SHAPE', (224, 224))
            )

            seen_dirs: Set[Path] = set()
            save_tasks: List[Tuple[torch.Tensor, Path]] = []

            for j, occluded_img in enumerate(occluded_images):
                occ_path = batch_occ_paths[j]
                if occ_path.parent not in seen_dirs:
                    self.file_manager.ensure_dir_exists(occ_path.parent)
                    seen_dirs.add(occ_path.parent)
                save_tasks.append((occluded_img, occ_path))

            # Parallel I/O saves
            if save_workers > 1 and len(save_tasks) > 1:
                with ThreadPoolExecutor(max_workers=save_workers) as executor:
                    futures = {
                        executor.submit(self._save_occluded_image, img, path): (img, path)
                        for img, path in save_tasks
                    }
                    for future in as_completed(futures):
                        _, path = futures[future]
                        try:
                            future.result()
                            self._mark_occluded_saved(path.parent, path.name)
                        except Exception as e:
                            logging.error(f"Error saving {path}: {e}")
            else:
                for img, path in save_tasks:
                    self._save_occluded_image(img, path)
                    self._mark_occluded_saved(path.parent, path.name)

    # ------------------------------------------------------------------
    # Image serialisation
    # ------------------------------------------------------------------

    def _save_occluded_image(self, image_tensor: torch.Tensor, path: Path):
        img_tensor = image_tensor.detach().cpu().clone()

        if img_tensor.ndim == 4:
            img_tensor = img_tensor[0]
        elif img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)

        img_tensor = torch.clamp(img_tensor * _IMAGENET_STD + _IMAGENET_MEAN, 0, 1)

        img_array = (img_tensor.numpy() * 255).astype(np.uint8)

        if img_array.ndim == 3:
            if img_array.shape[0] == 3:          # (C, H, W) -> (H, W, C)
                img_array = np.transpose(img_array, (1, 2, 0))
            # else already (H, W, C)
        elif img_array.ndim == 2:                 # grayscale -> RGB
            img_array = np.stack([img_array] * 3, axis=-1)
        else:
            logging.warning(f"Unexpected image shape {img_array.shape}, skipping {path}")
            return

        Image.fromarray(img_array).save(path, 'PNG')


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------

def main():
    from core._bootstrap import bootstrap_runner
    config, gpu_manager, file_manager, model_cache, get_cached_model = bootstrap_runner()

    runner = Phase2Runner(config, gpu_manager, file_manager, model_cache)
    runner.run(get_cached_model)


if __name__ == "__main__":
    main()