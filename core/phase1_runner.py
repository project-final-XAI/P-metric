"""
Phase 1: Heatmap Generation Runner  (optimized)

Key changes vs original
------------------------
1. Single dataset decode      – tensors + labels + orig sizes cached once, reused across all (model, method) combos.
2. PIL size reads off-thread  – ThreadPoolExecutor reads image dims concurrently.
3. Method instantiated once   – hoisted out of _process_method_batch.
4. autocast gated by type     – skipped for ModelIndependentMethod subclasses.
5. Batch D→H transfer         – single .cpu() call per batch, not per image.
6. Async save pipeline        – np.save + PNG write submitted to a thread pool.
7. Dir creation deduplicated  – global seen-dirs set shared across all calls.
8. GPU sort_pixels            – torch.argsort on flattened heatmaps (single H→D transfer).
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from attribution.base import ModelIndependentMethod
from attribution.registry import get_attribution_method
from core.file_manager import FileManager
from core.gpu_manager import GPUManager
from core.gpu_utils import prepare_batch_tensor
from data.imagenet_class_mapping import format_class_for_llm, get_cached_mapping
from data.loader import get_dataloader

_COLORMAP_CV2 = {
    "hot": cv2.COLORMAP_HOT,
    "jet": cv2.COLORMAP_JET,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "rainbow": cv2.COLORMAP_RAINBOW,
    "turbo": cv2.COLORMAP_TURBO,
}

# Number of threads for async I/O (size reads + saves)
_IO_WORKERS = min(8, (os.cpu_count() or 4))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_image_size(path: str) -> Tuple[int, int]:
    """Return (W, H) without decoding pixel data."""
    try:
        with Image.open(path) as im:
            return im.size          # (W, H)
    except Exception:
        return (0, 0)               # caller falls back to tensor size


def _save_sorted(path: Path, arr: np.ndarray) -> None:
    np.save(path, arr)


def _save_png(heatmap: np.ndarray, path: Path, colormap_key: str) -> None:
    hmap = heatmap.copy().astype(np.float32)
    span = hmap.max() - hmap.min()
    hmap = (hmap - hmap.min()) / (span + 1e-8)
    hmap_u8 = (hmap * 255).astype(np.uint8)
    cv_cmap = _COLORMAP_CV2.get(colormap_key, cv2.COLORMAP_HOT)
    colored = cv2.applyColorMap(hmap_u8, cv_cmap)
    Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)).save(path, "PNG")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class Phase1Runner:
    """Handles Phase 1: Heatmap generation for all model-method-image combinations."""

    def __init__(
        self,
        config,
        gpu_manager: GPUManager,
        file_manager: FileManager,
        model_cache: Dict[str, Any],
    ) -> None:
        self.config = config
        self.gpu_manager = gpu_manager
        self.file_manager = file_manager
        self.model_cache = model_cache

        # Shared set so we never mkdir the same path twice across all batch calls
        self._created_dirs: set[Path] = set()

        # Thread pool reused for the entire Phase-1 lifetime
        self._io_pool = ThreadPoolExecutor(max_workers=_IO_WORKERS)

        # ImageNet helpers
        self.imagenet_mapping: Optional[dict] = None
        self.synset_ids: List[str] = []
        if config.DATASET_NAME == "imagenet":
            try:
                self.imagenet_mapping = get_cached_mapping()
                from config import DATASET_CONFIG
                ds_path = DATASET_CONFIG.get("imagenet", {}).get("path")
                if ds_path and os.path.exists(ds_path):
                    self.synset_ids = sorted(
                        d for d in os.listdir(ds_path)
                        if os.path.isdir(os.path.join(ds_path, d))
                    )
            except Exception as e:
                logging.warning(f"Could not load ImageNet mapping: {e}")

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

        logging.info(f"Starting Phase 1 - Dataset: {dataset_name}")
        logging.info(
            f"Models: {len(self.config.GENERATING_MODELS)} | "
            f"Methods: {len(self.config.ATTRIBUTION_METHODS)}"
        )

        try:
            heatmap_dir = self.file_manager.get_heatmap_dir(dataset_name)
            self._ensure_dir(heatmap_dir)

            # ── Single decode pass: cache (tensor, label, orig_size) ───────
            image_records = self._build_image_records(dataset_name)

            # ── Inference: iterate the cached tensors per (model, method) ──
            total = len(self.config.GENERATING_MODELS) * len(self.config.ATTRIBUTION_METHODS)
            with tqdm(total=total, desc="Phase 1 Progress") as pbar:
                for m_idx, model_name in enumerate(self.config.GENERATING_MODELS, 1):
                    model = get_cached_model_func(model_name)

                    for a_idx, method_name in enumerate(self.config.ATTRIBUTION_METHODS, 1):
                        self.gpu_manager.check_and_throttle()
                        pbar.set_description(
                            f"[{m_idx}/{len(self.config.GENERATING_MODELS)}] {model_name[:12]} | "
                            f"[{a_idx}/{len(self.config.ATTRIBUTION_METHODS)}] {method_name[:15]}"
                        )
                        try:
                            method = get_attribution_method(method_name)
                            self._process_method_cached(
                                model, model_name, method, method_name,
                                image_records, dataset_name,
                            )
                        except Exception as e:
                            logging.error(
                                f"Error: {model_name}-{method_name}: {e}", exc_info=True
                            )
                        finally:
                            pbar.update(1)

            # Wait for all background saves to finish before returning
            self._io_pool.shutdown(wait=True)
            self._io_pool = ThreadPoolExecutor(max_workers=_IO_WORKERS)   # re-open for safety

            logging.info(f"Heatmaps saved to: {heatmap_dir}")

        except Exception as e:
            logging.error(f"Phase 1 failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Single-pass dataset cache
    # ------------------------------------------------------------------

    def _build_image_records(
        self, dataset_name: str
    ) -> List[Tuple[str, int, Tuple[int, int], torch.Tensor]]:
        """Decode the dataset once and cache (img_id, label, orig_size, tensor).

        Returns a list ordered by global index so that downstream code can
        rely on stable ``image_{idx:05d}`` ids.
        """
        loader_batch = getattr(self.config, "HEATMAP_BATCH_SIZE", 12)
        dataloader = get_dataloader(dataset_name, batch_size=loader_batch, shuffle=False)
        dataset_samples = getattr(dataloader.dataset, "samples", [])

        # Read original image sizes concurrently (no pixel decode).
        size_futures = {
            self._io_pool.submit(_read_image_size, dataset_samples[i][0]): i
            for i in range(len(dataset_samples))
        }
        orig_sizes: Dict[int, Tuple[int, int]] = {}
        for fut in as_completed(size_futures):
            idx = size_futures[fut]
            orig_sizes[idx] = fut.result()

        records: List[Tuple[str, int, Tuple[int, int], torch.Tensor]] = []
        global_idx = 0
        for batch_images, batch_labels in dataloader:
            # ``batch_images`` is a (B, C, H, W) tensor; clone rows so we can
            # release the underlying batch storage at the end of each iter.
            for img, lbl in zip(batch_images, batch_labels):
                raw_size = orig_sizes.get(global_idx, (0, 0))
                if raw_size == (0, 0):
                    raw_size = (img.shape[-1], img.shape[-2])   # (W, H)
                records.append(
                    (
                        f"image_{global_idx:05d}",
                        lbl.item(),
                        raw_size,
                        img.detach().clone(),
                    )
                )
                global_idx += 1

        logging.info(f"Cached {len(records)} dataset images for Phase 1.")
        return records

    # ------------------------------------------------------------------
    # Core processor: iterates the cached image records
    # ------------------------------------------------------------------

    def _process_method_cached(
        self,
        model: Any,
        model_name: str,
        method: Any,
        method_name: str,
        image_records: List[Tuple[str, int, Tuple[int, int], torch.Tensor]],
        dataset_name: str,
    ) -> None:
        """Run a single attribution method against every cached image."""

        batch_size = self.gpu_manager.get_batch_size(method_name)
        colormap_key = getattr(self.config, "HEATMAP_COLORMAP", "hot").lower()

        # Determine whether AMP autocast applies for this method type
        use_autocast = (
            self.config.DEVICE == "cuda"
            and not isinstance(method, ModelIndependentMethod)
        )
        amp_ctx = (
            torch.amp.autocast(self.config.DEVICE) if use_autocast else nullcontext()
        )

        # Rebuild per-image path info (needed to decide what to skip)
        path_meta: List[Tuple[Path, Path]] = []
        for img_id, label, _, _ in image_records:
            cat = self._category_name_for_label(dataset_name, label)
            s = self.file_manager.get_sorted_heatmap_path(
                dataset_name, model_name, method_name, img_id, cat
            )
            r = self.file_manager.get_regular_heatmap_path(
                dataset_name, model_name, method_name, img_id, cat
            )
            path_meta.append((s, r))

        # Collect outstanding work (skip already-saved combinations)
        pending_meta_idx: List[int] = [
            i for i, (s, r) in enumerate(path_meta) if not s.exists() or not r.exists()
        ]
        if not pending_meta_idx:
            return

        # Iterate the in-memory cache in batches sized for the GPU.
        for batch_start in tqdm(
            range(0, len(pending_meta_idx), batch_size),
            desc=f"  → {method_name[:20]}",
            dynamic_ncols=True,
            leave=False,
        ):
            batch_idxs = pending_meta_idx[batch_start: batch_start + batch_size]
            if batch_start > 0 and (batch_start % (batch_size * 5)) == 0:
                self.gpu_manager.check_and_throttle()

            self._flush_batch(
                batch_idxs,
                image_records,
                path_meta,
                model,
                method,
                amp_ctx,
                colormap_key,
            )

        self.gpu_manager.check_and_throttle()

    # ------------------------------------------------------------------
    # Single-batch flush (inference + GPU sort + async save)
    # ------------------------------------------------------------------

    def _flush_batch(
        self,
        idxs: List[int],
        image_records: List[Tuple[str, int, Tuple[int, int], torch.Tensor]],
        path_meta: List[Tuple[Path, Path]],
        model: Any,
        method: Any,
        amp_ctx,
        colormap_key: str,
    ) -> None:
        if not idxs:
            return

        # Pre-create dirs (deduplicated globally)
        for i in idxs:
            for p in path_meta[i]:
                self._ensure_dir(p.parent)

        imgs = [image_records[i][3] for i in idxs]
        batch_tensor = prepare_batch_tensor(
            imgs,
            device=self.config.DEVICE,
            memory_format=torch.channels_last,
        )
        batch_labels = torch.as_tensor(
            [image_records[i][1] for i in idxs],
            dtype=torch.long,
        ).to(self.config.DEVICE, non_blocking=True)

        with amp_ctx:
            heatmaps = method.compute(model, batch_tensor, batch_labels)
        if heatmaps is None:
            return

        # Reduce channel dim if present so heatmaps are (B, H, W).
        if heatmaps.ndim == 4:
            heatmaps = heatmaps.mean(dim=1)
        elif heatmaps.ndim != 3:
            raise ValueError(f"Unexpected heatmap shape {tuple(heatmaps.shape)}")

        B, H, W = heatmaps.shape

        # GPU sort: argsort flattened heatmaps in one shot, then transfer once.
        # ``np.argsort`` orders ascending, which matches the legacy behaviour;
        # we do the same with ``torch.argsort`` (ascending by default).
        sorted_idx_gpu = torch.argsort(heatmaps.reshape(B, -1), dim=1)
        sorted_idx_np: np.ndarray = sorted_idx_gpu.to(dtype=torch.int64).cpu().numpy()

        # Single D→H transfer for the heatmap values themselves.
        heatmaps_np: np.ndarray = heatmaps.detach().cpu().numpy()  # (B, H, W)

        for j, h_np in enumerate(heatmaps_np):
            meta_i = idxs[j]
            orig_w, orig_h = image_records[meta_i][2]
            sorted_path, regular_path = path_meta[meta_i]

            sorted_indices = sorted_idx_np[j]

            # Resize for the visualisation only; the .npy stays at model res.
            if orig_w > 0 and orig_h > 0 and (h_np.shape[1], h_np.shape[0]) != (orig_w, orig_h):
                h_png = cv2.resize(
                    h_np.astype(np.float32),
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                h_png = h_np

            self._io_pool.submit(_save_sorted, sorted_path, sorted_indices)
            self._io_pool.submit(_save_png, h_png.copy(), regular_path, colormap_key)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _ensure_dir(self, path: Path) -> None:
        if path not in self._created_dirs:
            self.file_manager.ensure_dir_exists(path)
            self._created_dirs.add(path)

    def _category_name_for_label(self, dataset_name: str, label: int) -> Optional[str]:
        if dataset_name != "imagenet" or not self.imagenet_mapping or not self.synset_ids:
            return None
        try:
            if label < len(self.synset_ids):
                full = self.imagenet_mapping.get(self.synset_ids[label], "")
                if full:
                    return format_class_for_llm(full)
        except Exception as e:
            logging.debug(f"Could not get category name for label {label}: {e}")
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    from core._bootstrap import bootstrap_runner
    config, gpu_manager, file_manager, model_cache, get_cached_model = bootstrap_runner()
    runner = Phase1Runner(config, gpu_manager, file_manager, model_cache)
    runner.run(get_cached_model)


if __name__ == "__main__":
    main()