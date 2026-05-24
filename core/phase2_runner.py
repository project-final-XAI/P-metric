"""
Phase 2: Pre-Generate All Occluded Images (Fixed for Model-Independent Methods)

Enforces architectural parity with Phase 1 by splitting processing paths into:
1. Model-Independent Methods (Runs once globally, model_name=None)
2. Model-Dependent Methods (Iterates per model, fresh instantiation lifecycle)
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from core.file_manager import FileManager
from core.gpu_manager import GPUManager
from core.gpu_utils import clear_cache_if_needed
from data.loader import get_dataset_handler
from models.loader import get_model_provider
from attribution.registry import get_attribution_method
from attribution.base import ModelIndependentMethod
from evaluation.occlusion import apply_fill_to_batch, build_occlusion_mask_batch

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _resolve_io_workers(config) -> int:
    cfg = getattr(config, "PHASE2_SAVE_WORKERS", None)
    return int(cfg) if cfg and cfg > 0 else min(8, os.cpu_count() or 4)


# ---------------------------------------------------------------------------
# High-Efficiency Processing Helpers
# ---------------------------------------------------------------------------

def _load_npy(path: Path) -> np.ndarray:
    return np.load(path)


def _save_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path, "PNG")


def _batch_to_uint8(batch: torch.Tensor) -> np.ndarray:
    """(B,C,H,W) ImageNet-normalised float -> (B,H,W,3) uint8, one vectorised op."""
    device = batch.device
    mean = _IMAGENET_MEAN.to(device)
    std = _IMAGENET_STD.to(device)

    b = (batch.float() * std + mean).clamp_(0, 1)
    return (b.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Runner Class
# ---------------------------------------------------------------------------

class Phase2Runner:
    """Handles Phase 2: Pre-generation of all occluded images with model-independent support."""

    def __init__(
        self,
        config,
        gpu_manager: GPUManager,
        file_manager: FileManager,
        dataset_handler: Any = None,
        model_provider: Any = None
    ) -> None:
        self.config          = config
        self.gpu_manager     = gpu_manager
        self.file_manager    = file_manager

        # Gracefully handle older callers that passed model_cache as the 4th positional argument
        if isinstance(dataset_handler, dict) or dataset_handler is None:
            self.dataset_handler = get_dataset_handler(config.DATASET_NAME)
            self.model_provider  = get_model_provider(config.DATASET_NAME)
        else:
            self.dataset_handler = dataset_handler
            self.model_provider  = model_provider

        # Keyed on Tuple[Optional[str], str] -> maps (model_name, method_name) securely
        self._sorted_path_cache: Dict[Tuple[Optional[str], str], Dict[str, Path]] = {}
        self._existing_occluded: Dict[str, Set[str]]                              = {}
        self._created_dirs:      Set[Path]                                        = set()

        self._io_workers = _resolve_io_workers(config)
        self._io_pool    = ThreadPoolExecutor(max_workers=self._io_workers)
        self._pending_saves: List[Future]                                 = []

    def __del__(self) -> None:
        if hasattr(self, "_io_pool"):
            self._io_pool.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Main Architectural Entry Point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Executes Phase 2 pre-generation loop utilizing structural method parity."""
        dataset_name = self.config.DATASET_NAME

        if dataset_name not in self.config.DATASET_CONFIG:
            raise ValueError(
                f"Dataset '{dataset_name}' not found in DATASET_CONFIG. "
                f"Available: {list(self.config.DATASET_CONFIG.keys())}"
            )

        logging.info(f"Starting Phase 2 – Dataset: {dataset_name}")

        # Get total images from dataset handler
        dataloader = self.dataset_handler.get_dataloader(batch_size=1, shuffle=False)
        total_images = len(dataloader.dataset)
        if hasattr(self.config, "MAX_IMAGES") and self.config.MAX_IMAGES is not None:
            total_images = min(total_images, self.config.MAX_IMAGES)

        self._ensure_phase1_complete(dataset_name, total_images)

        # ── 1. Split Methods via the Framework Registry ──
        independent_methods: List[str] = []
        dependent_methods: List[str] = []
        for method_name in self.config.ATTRIBUTION_METHODS:
            method_instance = get_attribution_method(method_name)
            if isinstance(method_instance, ModelIndependentMethod):
                independent_methods.append(method_name)
            else:
                dependent_methods.append(method_name)

        logging.info(
            f"Wired Methods -> Independent: {len(independent_methods)} | "
            f"Dependent: {len(dependent_methods)}"
        )

        # ── 2. Pre-cache Heatmap Disk Paths ──
        logging.info("Building sorted heatmap path cache…")
        for method_name in independent_methods:
            paths = self.file_manager.scan_sorted_heatmaps(dataset_name, None, method_name)
            path_map = {}
            for p in paths:
                parts = p.stem.split('_')
                if len(parts) >= 2:
                    img_id = f"{parts[0]}_{parts[1]}"
                    path_map[img_id] = p
            self._sorted_path_cache[(None, method_name)] = path_map

        for model_name in self.config.GENERATING_MODELS:
            for method_name in dependent_methods:
                paths = self.file_manager.scan_sorted_heatmaps(dataset_name, model_name, method_name)
                path_map = {}
                for p in paths:
                    parts = p.stem.split('_')
                    if len(parts) >= 2:
                        img_id = f"{parts[0]}_{parts[1]}"
                        path_map[img_id] = p
                self._sorted_path_cache[(model_name, method_name)] = path_map

        # Calculate exact total step boundaries for progress bars
        total_combos = (
            (len(independent_methods) * len(self.config.OCCLUSION_LEVELS) * len(self.config.FILL_STRATEGIES)) +
            (len(self.config.GENERATING_MODELS) * len(dependent_methods) * len(self.config.OCCLUSION_LEVELS) * len(self.config.FILL_STRATEGIES))
        )

        with tqdm(total=total_combos, desc="Phase 2 Progress") as pbar:
            # ── PATH A: Process Model-Independent Methods (model_name = None) ──
            for method_name in independent_methods:
                sorted_indices_map = self._load_sorted_indices(None, method_name)
                if not sorted_indices_map:
                    pbar.update(len(self.config.OCCLUSION_LEVELS) * len(self.config.FILL_STRATEGIES))
                    continue

                image_cache = self._stream_images(sorted_indices_map)

                for level in self.config.OCCLUSION_LEVELS:
                    try:
                        self._process_level(
                            dataset_name, None, method_name,
                            level, sorted_indices_map, image_cache, pbar,
                        )
                    except Exception as e:
                        logging.error(f"Error Independent-{method_name}-{level}%: {e}", exc_info=True)
                        pbar.update(len(self.config.FILL_STRATEGIES))

                del image_cache
                clear_cache_if_needed(threshold_percent=40.0)

            # ── PATH B: Process Model-Dependent Methods ──
            for model_name in self.config.GENERATING_MODELS:
                for method_name in dependent_methods:
                    sorted_indices_map = self._load_sorted_indices(model_name, method_name)
                    if not sorted_indices_map:
                        pbar.update(len(self.config.OCCLUSION_LEVELS) * len(self.config.FILL_STRATEGIES))
                        continue

                    image_cache = self._stream_images(sorted_indices_map)

                    for level in self.config.OCCLUSION_LEVELS:
                        try:
                            self._process_level(
                                dataset_name, model_name, method_name,
                                level, sorted_indices_map, image_cache, pbar,
                            )
                        except Exception as e:
                            logging.error(f"Error {model_name}-{method_name}-{level}%: {e}", exc_info=True)
                            pbar.update(len(self.config.FILL_STRATEGIES))

                    del image_cache
                    clear_cache_if_needed(threshold_percent=40.0)

        self._drain_saves(block=True)
        self._io_pool.shutdown(wait=True)
        self._io_pool = ThreadPoolExecutor(max_workers=self._io_workers)

        logging.info(f"Phase 2 complete! Pre-generated files saved to: {self.file_manager.get_occluded_dir(dataset_name)}")

    # ------------------------------------------------------------------
    # Component Helpers & Lifecycle Guards
    # ------------------------------------------------------------------

    def _ensure_phase1_complete(self, dataset_name: str, total_images: int) -> None:
        missing = []
        for a in self.config.ATTRIBUTION_METHODS:
            method_instance = get_attribution_method(a)
            if isinstance(method_instance, ModelIndependentMethod):
                if len(self.file_manager.scan_sorted_heatmaps(dataset_name, None, a)) < total_images:
                    missing.append((None, a))
            else:
                for m in self.config.GENERATING_MODELS:
                    if len(self.file_manager.scan_sorted_heatmaps(dataset_name, m, a)) < total_images:
                        missing.append((m, a))

        if missing:
            logging.info(f"Phase 1 components missing for {len(missing)} metrics. Triggering dynamic fallback...")
            exit(1)

    def _load_sorted_indices(self, model_name: Optional[str], method_name: str) -> Dict[str, np.ndarray]:
        path_map = self._sorted_path_cache.get((model_name, method_name), {})
        if not path_map:
            return {}

        futures = {self._io_pool.submit(_load_npy, p): img_id for img_id, p in path_map.items()}
        loaded: Dict[str, np.ndarray] = {}
        for fut in as_completed(futures):
            img_id = futures[fut]
            try:
                loaded[img_id] = fut.result()
            except Exception as e:
                logging.error(f"Failed to load sorted indices for {img_id}: {e}")
        return loaded

    def _stream_images(self, sorted_indices_map: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        needed: Set[str] = set(sorted_indices_map.keys())
        bs     = getattr(self.config, "PHASE2_BATCH_SIZE", 256)
        dl     = self.dataset_handler.get_dataloader(batch_size=bs, shuffle=False)

        cache: Dict[str, torch.Tensor] = {}
        idx = 0
        for dl_imgs, _ in dl:
            for img in dl_imgs:
                img_id = f"image_{idx:05d}"
                if img_id in needed:
                    cache[img_id] = img
                idx += 1
            if len(cache) == len(needed):
                break
        return cache

    # ------------------------------------------------------------------
    # Asynchronous Disk Output Trackers
    # ------------------------------------------------------------------

    def _is_occluded_done(self, path: Path) -> bool:
        key = str(path.parent)
        if key not in self._existing_occluded:
            self._existing_occluded[key] = {p.name for p in path.parent.iterdir()} if path.parent.exists() else set()
        return path.name in self._existing_occluded[key]

    def _mark_saved(self, path: Path) -> None:
        key = str(path.parent)
        if key in self._existing_occluded:
            self._existing_occluded[key].add(path.name)

    def _ensure_dir(self, path: Path) -> None:
        if path not in self._created_dirs:
            self.file_manager.ensure_dir_exists(path)
            self._created_dirs.add(path)

    def _drain_saves(self, block: bool = False) -> None:
        if block:
            pending = self._pending_saves
        else:
            pending = [f for f in self._pending_saves if f.done()]

        remaining = []
        for fut in self._pending_saves:
            if fut.done() or block:
                try:
                    fut.result()
                except Exception as e:
                    logging.error(f"Background Save error: {e}")
            else:
                remaining.append(fut)
        self._pending_saves = remaining

    # ------------------------------------------------------------------
    # Vectorized Batch Execution
    # ------------------------------------------------------------------

    def _process_level(
        self,
        dataset_name:       str,
        model_name:         Optional[str],
        method_name:        str,
        level:              int,
        sorted_indices_map: Dict[str, np.ndarray],
        image_cache:        Dict[str, torch.Tensor],
        pbar,
    ) -> None:
        batch_size = getattr(self.config, "PHASE2_BATCH_SIZE", 256)
        img_shape  = getattr(self.config, "OCCLUSION_IMAGE_SHAPE", (224, 224))
        device     = self.config.DEVICE

        per_strategy_ids:        Dict[str, Set[str]]        = {}
        per_strategy_path_lookup: Dict[str, Dict[str, Path]] = {}
        all_ids_needed: Set[str] = set()

        for strategy in self.config.FILL_STRATEGIES:
            id_set:    Set[str]        = set()
            path_dict: Dict[str, Path] = {}
            for img_id in sorted_indices_map.keys():
                occ_path = self.file_manager.get_occluded_image_path(
                    dataset_name, model_name, strategy, method_name, level, img_id
                )
                if not self._is_occluded_done(occ_path):
                    id_set.add(img_id)
                    path_dict[img_id] = occ_path
                    all_ids_needed.add(img_id)
            per_strategy_ids[strategy]         = id_set
            per_strategy_path_lookup[strategy] = path_dict

        if not all_ids_needed:
            pbar.update(len(self.config.FILL_STRATEGIES))
            return

        ordered_ids = [iid for iid in sorted_indices_map.keys() if iid in all_ids_needed]

        for batch_start in range(0, len(ordered_ids), batch_size):
            batch_ids = ordered_ids[batch_start: batch_start + batch_size]

            tensors      = [image_cache[iid] for iid in batch_ids]
            indices_list = [sorted_indices_map[iid] for iid in batch_ids]
            batch_tensor = torch.stack(tensors, dim=0).to(device, non_blocking=True)

            masks = build_occlusion_mask_batch(indices_list, level, img_shape, device=device)

            for strategy in self.config.FILL_STRATEGIES:
                ids_for_strategy = per_strategy_ids[strategy]
                if not ids_for_strategy:
                    continue

                occluded    = apply_fill_to_batch(batch_tensor, masks, strategy)
                batch_uint8 = _batch_to_uint8(occluded)

                path_lookup = per_strategy_path_lookup[strategy]

                for j, iid in enumerate(batch_ids):
                    if iid not in ids_for_strategy:
                        continue
                    occ_path = path_lookup[iid]
                    self._ensure_dir(occ_path.parent)

                    self._pending_saves.append(
                        self._io_pool.submit(_save_png, batch_uint8[j].copy(), occ_path)
                    )
                    self._mark_saved(occ_path)

            if len(self._pending_saves) > 500:
                self._drain_saves(block=False)

        pbar.update(len(self.config.FILL_STRATEGIES))


if __name__ == "__main__":
    from core._bootstrap import bootstrap_phase2
    runner = bootstrap_phase2()
    runner.run()