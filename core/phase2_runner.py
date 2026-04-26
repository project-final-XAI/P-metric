"""
Phase 2: Pre-Generate All Occluded Images  (fixed + optimised)

Root causes fixed vs the "super slow" version
----------------------------------------------
1. No repeated GPU transfers      – images transferred to GPU once per (model,method,level)
                                    batch, not re-stacked from RAM every inner loop.
2. Sorted indices not all in RAM  – loaded per-(model,method) on demand, released after
                                    each (model,method) pair is fully processed.
3. Mask built per-image, not per-batch-of-images inside the batch loop.
4. per_strategy path/id lookups   – built ONCE before the batch loop, not inside it.
5. Save futures fire-and-forget   – errors harvested lazily; main thread never blocks
                                    on disk between batches.
6. Loop order fixed               – (model, method, level, strategy) so indices are
                                    loaded once and reused across all levels/strategies.
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
from data.loader import get_dataloader
from evaluation.occlusion import apply_fill_to_batch, build_occlusion_mask_batch

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _resolve_io_workers(config) -> int:
    cfg = getattr(config, "PHASE2_SAVE_WORKERS", None)
    return int(cfg) if cfg and cfg > 0 else min(8, os.cpu_count() or 4)


# ---------------------------------------------------------------------------
# Free helpers
# ---------------------------------------------------------------------------

def _load_npy(path: Path) -> np.ndarray:
    return np.load(path)


def _save_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path, "PNG")


def _save_npy(arr: np.ndarray, path: Path) -> None:
    np.save(path, arr, allow_pickle=False)


def _batch_to_uint8(batch: torch.Tensor) -> np.ndarray:
    """(B,C,H,W) ImageNet-normalised float  ->  (B,H,W,3) uint8, one vectorised op."""
    b = (batch.cpu().float() * _IMAGENET_STD + _IMAGENET_MEAN).clamp_(0, 1)
    return (b.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class Phase2Runner:
    """Handles Phase 2: Pre-generation of all occluded images."""

    def __init__(
        self,
        config,
        gpu_manager: GPUManager,
        file_manager: FileManager,
        model_cache: Dict[str, Any],
    ) -> None:
        self.config       = config
        self.gpu_manager  = gpu_manager
        self.file_manager = file_manager
        self.model_cache  = model_cache

        self._sorted_path_cache:  Dict[Tuple[str, str], Dict[str, Path]] = {}
        self._existing_occluded:  Dict[str, Set[str]]                    = {}
        self._created_dirs:       Set[Path]                               = set()

        self._io_workers = _resolve_io_workers(config)
        self._io_pool    = ThreadPoolExecutor(max_workers=self._io_workers)

        # Lightweight metadata only — no pixel tensors
        self._metadata: Dict[str, int] = {}

        # Pending save futures — harvested lazily so disk never blocks GPU
        self._pending_saves: List[Future] = []

    def __del__(self) -> None:
        self._io_pool.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, get_cached_model_func) -> None:
        dataset_name = self.config.DATASET_NAME

        if dataset_name not in self.config.DATASET_CONFIG:
            raise ValueError(
                f"Dataset '{dataset_name}' not found in DATASET_CONFIG. "
                f"Available: {list(self.config.DATASET_CONFIG.keys())}"
            )

        logging.info(f"Starting Phase 2 – Dataset: {dataset_name}")

        self._metadata  = self._load_metadata(dataset_name)
        total_images    = len(self._metadata)

        self._ensure_phase1_complete(dataset_name, total_images, get_cached_model_func)

        logging.info("Building sorted heatmap path cache…")
        for model_name in self.config.GENERATING_MODELS:
            for method_name in self.config.ATTRIBUTION_METHODS:
                self._build_sorted_path_cache(dataset_name, model_name, method_name)

        # ── Main loop: model → method → (load indices once) → level → strategy ──
        total_combos = (
            len(self.config.GENERATING_MODELS)
            * len(self.config.ATTRIBUTION_METHODS)
            * len(self.config.OCCLUSION_LEVELS)
            * len(self.config.FILL_STRATEGIES)
        )

        with tqdm(total=total_combos, desc="Phase 2 Progress") as pbar:
            for model_name in self.config.GENERATING_MODELS:
                for method_name in self.config.ATTRIBUTION_METHODS:

                    # Load ALL sorted indices for this (model, method) pair once.
                    sorted_indices_map = self._load_sorted_indices(
                        model_name, method_name
                    )
                    if not sorted_indices_map:
                        pbar.update(
                            len(self.config.OCCLUSION_LEVELS)
                            * len(self.config.FILL_STRATEGIES)
                        )
                        continue

                    # Stream images once from disk; build a small RAM cache just
                    # for this (model, method) — freed when we move on.
                    image_cache = self._stream_images(dataset_name, sorted_indices_map)

                    for level in self.config.OCCLUSION_LEVELS:
                        try:
                            self._process_level(
                                dataset_name, model_name, method_name,
                                level, sorted_indices_map, image_cache, pbar,
                            )
                        except Exception as e:
                            logging.error(
                                f"Error {model_name}-{method_name}-{level}%: {e}",
                                exc_info=True,
                            )
                            pbar.update(len(self.config.FILL_STRATEGIES))

                    # Release image tensors — they are not needed for the next
                    # (model, method) pair (each pair has its own index set).
                    del image_cache

        # Drain all background saves before returning
        self._drain_saves(block=True)
        self._io_pool.shutdown(wait=True)
        self._io_pool = ThreadPoolExecutor(max_workers=self._io_workers)

        logging.info(
            f"Phase 2 complete! Occluded images saved to: "
            f"{self.file_manager.get_occluded_dir(dataset_name)}"
        )

    # ------------------------------------------------------------------
    # Metadata pass (labels only)
    # ------------------------------------------------------------------

    def _load_metadata(self, dataset_name: str) -> Dict[str, int]:
        bs = getattr(self.config, "PHASE2_BATCH_SIZE", 256)
        dl = get_dataloader(dataset_name, batch_size=bs, shuffle=False)
        meta: Dict[str, int] = {}
        idx = 0
        for _, lbls in dl:
            for lbl in lbls:
                meta[f"image_{idx:05d}"] = lbl.item()
                idx += 1
        logging.info(f"Loaded metadata for {len(meta)} images.")
        return meta

    # ------------------------------------------------------------------
    # Phase 1 guard
    # ------------------------------------------------------------------

    def _ensure_phase1_complete(
        self, dataset_name: str, total_images: int, get_cached_model_func
    ) -> None:
        missing = [
            (m, a)
            for m in self.config.GENERATING_MODELS
            for a in self.config.ATTRIBUTION_METHODS
            if len(self.file_manager.scan_sorted_heatmaps(dataset_name, m, a)) < total_images
        ]
        if missing:
            logging.info(f"Running Phase 1 for {len(missing)} missing combinations…")
            from core.phase1_runner import Phase1Runner
            Phase1Runner(
                self.config, self.gpu_manager, self.file_manager, self.model_cache
            ).run(get_cached_model_func)

    # ------------------------------------------------------------------
    # Sorted-heatmap path cache
    # ------------------------------------------------------------------

    def _build_sorted_path_cache(
        self, dataset_name: str, model_name: str, method_name: str
    ) -> None:
        key = (model_name, method_name)
        if key in self._sorted_path_cache:
            return

        synset_ids: Optional[List[str]] = None
        imagenet_mapping: Optional[dict] = None
        format_fn = None
        if dataset_name == "imagenet":
            try:
                from config import DATASET_CONFIG
                from data.imagenet_class_mapping import format_class_for_llm, get_cached_mapping
                mapping  = get_cached_mapping()
                ds_path  = DATASET_CONFIG.get("imagenet", {}).get("path")
                if ds_path and os.path.exists(ds_path):
                    synset_ids       = sorted(
                        d for d in os.listdir(ds_path)
                        if os.path.isdir(os.path.join(ds_path, d))
                    )
                    imagenet_mapping = mapping
                    format_fn        = format_class_for_llm
            except Exception as e:
                logging.debug(f"ImageNet mapping unavailable: {e}")

        path_map: Dict[str, Path] = {}
        for img_id, label in self._metadata.items():
            resolved: Optional[Path] = None
            if synset_ids is not None and label < len(synset_ids):
                full = imagenet_mapping.get(synset_ids[label], "")
                if full:
                    c = self.file_manager.get_sorted_heatmap_path(
                        dataset_name, model_name, method_name, img_id, format_fn(full)
                    )
                    if c.exists():
                        resolved = c
            if resolved is None:
                c = self.file_manager.get_sorted_heatmap_path(
                    dataset_name, model_name, method_name, img_id
                )
                if c.exists():
                    resolved = c
            if resolved is not None:
                path_map[img_id] = resolved
            else:
                logging.warning(f"Missing heatmap: {model_name}/{method_name}/{img_id}")

        self._sorted_path_cache[key] = path_map
        logging.info(
            f"  Cached {len(path_map)}/{len(self._metadata)} paths "
            f"for {model_name}/{method_name}"
        )

    # ------------------------------------------------------------------
    # Sorted indices: loaded once per (model, method), not kept long-term
    # ------------------------------------------------------------------

    def _load_sorted_indices(
        self, model_name: str, method_name: str
    ) -> Dict[str, np.ndarray]:
        """
        Load all sorted-index .npy files for (model, method) in parallel.
        Returns {img_id -> np.ndarray}.  Caller is responsible for del-ing
        the result when done to free RAM.
        """
        path_map = self._sorted_path_cache.get((model_name, method_name), {})
        if not path_map:
            return {}

        futures = {
            self._io_pool.submit(_load_npy, p): img_id
            for img_id, p in path_map.items()
        }
        loaded: Dict[str, np.ndarray] = {}
        for fut in as_completed(futures):
            img_id = futures[fut]
            try:
                loaded[img_id] = fut.result()
            except Exception as e:
                logging.error(f"Failed to load sorted indices for {img_id}: {e}")

        logging.info(
            f"  Loaded {len(loaded)} sorted-index arrays for {model_name}/{method_name}"
        )
        return loaded

    # ------------------------------------------------------------------
    # Stream images from disk for the IDs that are actually needed
    # ------------------------------------------------------------------

    def _stream_images(
        self,
        dataset_name: str,
        sorted_indices_map: Dict[str, np.ndarray],
    ) -> Dict[str, torch.Tensor]:
        """
        Stream the dataloader once and keep only tensors whose img_id appears
        in sorted_indices_map.  This avoids holding every dataset image in RAM
        when only a subset have valid heatmaps.
        """
        needed: Set[str] = set(sorted_indices_map.keys())
        bs     = getattr(self.config, "PHASE2_BATCH_SIZE", 256)
        dl     = get_dataloader(dataset_name, batch_size=bs, shuffle=False)

        cache: Dict[str, torch.Tensor] = {}
        idx = 0
        for dl_imgs, _ in dl:
            for img in dl_imgs:
                img_id = f"image_{idx:05d}"
                if img_id in needed:
                    cache[img_id] = img  # keep on CPU; .to(device) happens per-batch
                idx += 1
            if len(cache) == len(needed):
                break  # early exit once all needed images collected

        logging.info(f"  Streamed {len(cache)} images into Phase 2 image cache.")
        return cache

    # ------------------------------------------------------------------
    # Existence tracking
    # ------------------------------------------------------------------

    def _is_occluded_done(self, path: Path) -> bool:
        key = str(path.parent)
        if key not in self._existing_occluded:
            self._existing_occluded[key] = (
                {p.name for p in path.parent.iterdir()}
                if path.parent.exists()
                else set()
            )
        return path.name in self._existing_occluded[key]

    def _mark_saved(self, path: Path) -> None:
        key = str(path.parent)
        if key in self._existing_occluded:
            self._existing_occluded[key].add(path.name)

    def _ensure_dir(self, path: Path) -> None:
        if path not in self._created_dirs:
            self.file_manager.ensure_dir_exists(path)
            self._created_dirs.add(path)

    # ------------------------------------------------------------------
    # Lazy save-error harvesting
    # ------------------------------------------------------------------

    def _drain_saves(self, block: bool = False) -> None:
        """
        Collect completed futures and log errors.
        When block=False only harvests already-done futures (non-blocking).
        When block=True waits for all.
        """
        if block:
            pending = self._pending_saves
        else:
            # Only drain the ones that have already finished
            pending  = [f for f in self._pending_saves if f.done()]
        remaining = []
        for fut in self._pending_saves:
            if fut.done() or block:
                try:
                    fut.result()
                except Exception as e:
                    logging.error(f"Save error: {e}")
            else:
                remaining.append(fut)
        self._pending_saves = remaining

    # ------------------------------------------------------------------
    # Core: one level, all strategies
    # ------------------------------------------------------------------

    def _process_level(
        self,
        dataset_name:       str,
        model_name:         str,
        method_name:        str,
        level:              int,
        sorted_indices_map: Dict[str, np.ndarray],
        image_cache:        Dict[str, torch.Tensor],
        pbar,
    ) -> None:
        batch_size        = getattr(self.config, "PHASE2_BATCH_SIZE", 256)
        img_shape         = getattr(self.config, "OCCLUSION_IMAGE_SHAPE", (224, 224))
        device            = self.config.DEVICE

        # ── Pre-compute per-strategy work lists and lookup dicts ONCE ─────
        # Do NOT rebuild these inside the batch loop.
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

        # Ordered list for stable batching
        ordered_ids = [
            iid for iid in sorted_indices_map.keys()
            if iid in all_ids_needed
        ]

        for batch_start in range(0, len(ordered_ids), batch_size):
            batch_ids = ordered_ids[batch_start: batch_start + batch_size]

            # ── One GPU transfer per batch ─────────────────────────────────
            tensors      = [image_cache[iid] for iid in batch_ids]
            indices_list = [sorted_indices_map[iid] for iid in batch_ids]
            batch_tensor = torch.stack(tensors, dim=0).to(device, non_blocking=True)

            # ── Masks: one per IMAGE in the batch (not per strategy) ───────
            # build_occlusion_mask_batch takes the per-image index arrays and
            # returns a (B, H, W) bool tensor on `device`.
            masks = build_occlusion_mask_batch(
                indices_list, level, img_shape, device=device
            )

            # ── Apply each fill strategy — reuses batch_tensor + masks ─────
            for strategy in self.config.FILL_STRATEGIES:
                ids_for_strategy = per_strategy_ids[strategy]
                if not ids_for_strategy:
                    continue

                # Vectorised fill over the whole batch
                occluded    = apply_fill_to_batch(batch_tensor, masks, strategy)
                batch_uint8 = _batch_to_uint8(occluded)   # (B, H, W, 3) uint8 on CPU

                path_lookup = per_strategy_path_lookup[strategy]   # pre-built, O(1)

                for j, iid in enumerate(batch_ids):
                    if iid not in ids_for_strategy:
                        continue
                    occ_path = path_lookup[iid]
                    self._ensure_dir(occ_path.parent)

                    # Fire-and-forget saves — do NOT block here
                    self._pending_saves.append(
                        self._io_pool.submit(_save_png, batch_uint8[j].copy(), occ_path)
                    )
                    self._mark_saved(occ_path)

            # Opportunistically drain already-finished futures so the list
            # doesn't grow unboundedly across a long run.
            if len(self._pending_saves) > 500:
                self._drain_saves(block=False)

        pbar.update(len(self.config.FILL_STRATEGIES))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    from core._bootstrap import bootstrap_runner
    config, gpu_manager, file_manager, model_cache, get_cached_model = bootstrap_runner()
    Phase2Runner(config, gpu_manager, file_manager, model_cache).run(get_cached_model)


if __name__ == "__main__":
    main()