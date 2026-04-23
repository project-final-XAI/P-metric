"""
Phase 3: Super-Fast Evaluation Runner.

Loads pre-generated occluded images from Phase 2 and tests them with judging models.
NO image generation - only loading and testing for maximum efficiency.
"""

import numpy as np
import torch
import logging
import time
import queue
import threading
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Set, Optional
from collections import defaultdict
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from threading import Lock

from core.gpu_manager import GPUManager
from core.file_manager import FileManager
from core.gpu_utils import prepare_batch_tensor
from core.phase2_runner import Phase2Runner
from data.loader import get_dataloader, get_default_transforms
from evaluation.judging.base import JudgingModel


class Phase3Runner:
    """Handles Phase 3: Super-fast evaluation of pre-generated occluded images."""

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
        self.transform = get_default_transforms()
        self.load_workers = getattr(config, 'PHASE3_LOAD_WORKERS', 8)
        self.csv_locks: Dict[str, Lock] = {}

        # Persistent thread pool for image loading — created once, reused across all batches
        self._load_executor = ThreadPoolExecutor(max_workers=self.load_workers)

        # Occluded image scan cache: (dataset, gen_model, strategy, method, level) -> List[Path]
        self._scan_cache: Dict[Tuple, List[Path]] = {}

        # Completed-items in-memory cache — avoids re-reading CSV on every level
        self._completed_cache: Dict[str, Set[Tuple[str, float]]] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, get_cached_model_func):
        """Evaluate pre-generated occluded images with judging models."""
        dataset_name = self.config.DATASET_NAME

        if dataset_name not in self.config.DATASET_CONFIG:
            raise ValueError(
                f"Dataset '{dataset_name}' not found in DATASET_CONFIG. "
                f"Available datasets: {list(self.config.DATASET_CONFIG.keys())}"
            )

        logging.info(f"Starting Phase 3 - Dataset: {dataset_name}")

        try:
            self._ensure_phase2_complete(dataset_name, get_cached_model_func)
            image_label_map = self._load_dataset_labels(dataset_name)
            judging_models = self._load_judging_models(get_cached_model_func)

            total_combinations = (
                len(self.config.GENERATING_MODELS) *
                len(self.config.ATTRIBUTION_METHODS) *
                len(self.config.FILL_STRATEGIES) *
                len(self.config.JUDGING_MODELS)
            )

            with tqdm(total=total_combinations, desc="Phase 3 Progress") as pbar:
                for gen_model in self.config.GENERATING_MODELS:
                    for judge_name in self.config.JUDGING_MODELS:
                        for strategy in self.config.FILL_STRATEGIES:
                            for method in self.config.ATTRIBUTION_METHODS:
                                if judge_name == gen_model:
                                    pbar.update(1)
                                    continue

                                pbar.set_description(
                                    f"{gen_model[:12]}/{method[:12]}/{strategy}/{judge_name[:12]}"
                                )

                                try:
                                    self._evaluate_combination(
                                        dataset_name, gen_model, method, strategy,
                                        judge_name, judging_models[judge_name],
                                        image_label_map
                                    )
                                except Exception as e:
                                    logging.error(
                                        f"Error: {gen_model}-{method}-{strategy}-{judge_name}: {e}"
                                    )
                                finally:
                                    pbar.update(1)
        finally:
            self._load_executor.shutdown(wait=False)

        logging.info(
            f"Phase 3 complete! Results saved to: {self.file_manager.get_result_dir(dataset_name)}"
        )

    # ------------------------------------------------------------------
    # Phase 2 completeness check
    # ------------------------------------------------------------------

    def _ensure_phase2_complete(self, dataset_name: str, get_cached_model_func):
        batch_size = getattr(self.config, 'PHASE2_BATCH_SIZE', 256)
        dataloader = get_dataloader(dataset_name, batch_size=batch_size, shuffle=False)
        total_images = len(dataloader.dataset)

        missing_items = [
            (model_name, method_name, strategy, level)
            for model_name in self.config.GENERATING_MODELS
            for method_name in self.config.ATTRIBUTION_METHODS
            for strategy in self.config.FILL_STRATEGIES
            for level in self.config.OCCLUSION_LEVELS
            if len(self._scan_occluded(dataset_name, model_name, strategy, method_name, level)) < total_images
        ]

        if missing_items:
            logging.info(f"Running Phase 2 for {len(missing_items)} missing combinations...")
            phase2 = Phase2Runner(self.config, self.gpu_manager, self.file_manager, self.model_cache)
            phase2.run(get_cached_model_func)

    # ------------------------------------------------------------------
    # Cached filesystem scan
    # ------------------------------------------------------------------

    def _scan_occluded(
        self, dataset_name: str, gen_model: str, strategy: str, method: str, level: int
    ) -> List[Path]:
        """Scan occluded images for a combination, caching the result."""
        key = (dataset_name, gen_model, strategy, method, level)
        if key not in self._scan_cache:
            self._scan_cache[key] = self.file_manager.scan_occluded_images(
                dataset_name, gen_model, strategy, method, level
            )
        return self._scan_cache[key]

    # ------------------------------------------------------------------
    # Dataset / model loading
    # ------------------------------------------------------------------

    def _load_dataset_labels(self, dataset_name: str) -> Dict[str, int]:
        """Load dataset labels only (no images needed in Phase 3)."""
        batch_size = getattr(self.config, 'PHASE3_BATCH_SIZE_PYTORCH', 256)
        dataloader = get_dataloader(dataset_name, batch_size=batch_size, shuffle=False)
        image_label_map: Dict[str, int] = {}
        global_idx = 0

        for _, batch_labels in dataloader:
            for lbl in batch_labels:
                image_label_map[f"image_{global_idx:05d}"] = lbl.item()
                global_idx += 1

        return image_label_map

    def _load_judging_models(self, get_cached_model_func) -> Dict[str, Any]:
        judging_models = {
            name: get_cached_model_func(name) for name in self.config.JUDGING_MODELS
        }

        if (
            self.config.USE_FP16_INFERENCE
            and self.config.DEVICE == "cuda"
            and self.gpu_manager.supports_fp16()
        ):
            for name, model in judging_models.items():
                if isinstance(model, JudgingModel):
                    continue
                try:
                    judging_models[name] = model.half()
                    logging.debug(f"Converted {name} to FP16")
                except Exception as e:
                    logging.warning(f"Failed to convert {name} to FP16: {e}")

        return judging_models

    # ------------------------------------------------------------------
    # Completed-items cache (avoids repeated CSV reads)
    # ------------------------------------------------------------------

    def _get_completed_items(self, result_file: Path) -> Set[Tuple[str, float]]:
        """Return completed items, reading CSV at most once per result file."""
        key = str(result_file)
        if key not in self._completed_cache:
            self._completed_cache[key] = self._read_completed_items(result_file)
        return self._completed_cache[key]

    def _mark_completed(self, result_file: Path, item: Tuple[str, float]):
        """Keep the in-memory completed cache consistent after a new result is recorded."""
        key = str(result_file)
        if key in self._completed_cache:
            self._completed_cache[key].add(item)

    def _read_completed_items(self, result_file: Path) -> Set[Tuple[str, float]]:
        if not result_file.exists():
            return set()

        lock = self._get_csv_lock(result_file)
        with lock:
            rows = self.file_manager.load_csv(result_file, skip_header=True)

        completed = set()
        for row in rows:
            if len(row) >= 2:
                try:
                    completed.add((row[0], float(row[1])))
                except (ValueError, IndexError):
                    continue
        return completed

    # ------------------------------------------------------------------
    # Combination evaluation
    # ------------------------------------------------------------------

    def _evaluate_combination(
        self,
        dataset_name: str,
        gen_model: str,
        method: str,
        strategy: str,
        judge_name: str,
        judge_model: Any,
        image_label_map: Dict[str, int]
    ):
        result_file = self.file_manager.get_result_file_path(
            dataset_name, gen_model, judge_name, method, strategy
        )

        results_by_level: Dict[int, List] = defaultdict(list)
        saved_results: Set[Tuple[str, float]] = set()
        items_since_save = 0
        last_save_time = time.time()
        save_interval_items = getattr(self.config, 'PHASE3_SAVE_INTERVAL_ITEMS', 50)
        save_interval_seconds = getattr(self.config, 'PHASE3_SAVE_INTERVAL_SECONDS', 120)

        total_images_to_process = self._count_images_to_process(
            dataset_name, gen_model, method, strategy, result_file, image_label_map
        )

        inner_pbar = None
        processed_count = 0
        start_time = time.time()
        if total_images_to_process > 0:
            inner_pbar = tqdm(
                total=total_images_to_process,
                desc=f"  → {gen_model[:10]}/{method[:10]}/{strategy[:8]}/{judge_name[:10]}",
                leave=False,
                unit="img",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%] {elapsed}<{remaining}'
            )

        is_llm_judge = isinstance(judge_model, JudgingModel)

        for level in self.config.OCCLUSION_LEVELS:
            images_to_process, labels_to_process = self._filter_images_to_process(
                dataset_name, gen_model, method, strategy, level,
                result_file, saved_results, image_label_map
            )

            if not images_to_process:
                continue

            if is_llm_judge:
                items_since_save, last_save_time, processed_count = self._evaluate_llm_level(
                    judge_model, images_to_process, labels_to_process, level,
                    gen_model, method, strategy, results_by_level, saved_results,
                    result_file, inner_pbar, processed_count, start_time,
                    total_images_to_process, items_since_save, last_save_time,
                    save_interval_items, save_interval_seconds
                )
            else:
                items_since_save, last_save_time, processed_count = self._evaluate_pytorch_level(
                    judge_model, images_to_process, labels_to_process, level,
                    gen_model, method, results_by_level, saved_results,
                    result_file, inner_pbar, processed_count, start_time,
                    total_images_to_process, items_since_save, last_save_time,
                    save_interval_items, save_interval_seconds
                )

        if results_by_level:
            self._save_results(results_by_level, result_file)
        if inner_pbar is not None:
            inner_pbar.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _image_id_from_path(path: Path, gen_model: str, method: str) -> str:
        """Extract the image ID from an occluded-image filename."""
        return path.stem.replace(f"{gen_model}-{method}-", "") or path.stem

    # ------------------------------------------------------------------
    # Image filtering helpers (single scan per level)
    # ------------------------------------------------------------------

    def _count_images_to_process(
        self,
        dataset_name: str,
        gen_model: str,
        method: str,
        strategy: str,
        result_file: Path,
        image_label_map: Dict[str, int]
    ) -> int:
        completed_items = self._get_completed_items(result_file)
        total = 0
        for level in self.config.OCCLUSION_LEVELS:
            for img_path in self._scan_occluded(dataset_name, gen_model, strategy, method, level):
                img_id = self._image_id_from_path(img_path, gen_model, method)
                if (img_id, float(level)) not in completed_items and img_id in image_label_map:
                    total += 1
        return total

    def _filter_images_to_process(
        self,
        dataset_name: str,
        gen_model: str,
        method: str,
        strategy: str,
        level: int,
        result_file: Path,
        saved_results: Set[Tuple[str, float]],
        image_label_map: Dict[str, int]
    ) -> Tuple[List[Path], List[int]]:
        # Use cached scan + cached completed items — no filesystem hits here
        completed_items = self._get_completed_items(result_file)
        occluded_images = self._scan_occluded(dataset_name, gen_model, strategy, method, level)

        images_to_process: List[Path] = []
        labels_to_process: List[int] = []

        for img_path in occluded_images:
            img_id = self._image_id_from_path(img_path, gen_model, method)
            result_key = (img_id, float(level))
            if result_key in completed_items or result_key in saved_results:
                continue
            if img_id not in image_label_map:
                continue
            images_to_process.append(img_path)
            labels_to_process.append(image_label_map[img_id])

        return images_to_process, labels_to_process

    # ------------------------------------------------------------------
    # LLM evaluation
    # ------------------------------------------------------------------

    def _evaluate_llm_level(
        self,
        judge_model: Any,
        images_to_process: List[Path],
        labels_to_process: List[int],
        level: int,
        gen_model: str,
        method: str,
        strategy: str,
        results_by_level: Dict,
        saved_results: Set,
        result_file: Path,
        inner_pbar: Optional[tqdm],
        processed_count: int,
        start_time: float,
        total_images_to_process: int,
        items_since_save: int,
        last_save_time: float,
        save_interval_items: int,
        save_interval_seconds: float
    ) -> Tuple[int, float, int]:
        batch_size = getattr(self.config, 'PHASE3_BATCH_SIZE_LLM', 32)
        num_batches = (len(images_to_process) + batch_size - 1) // batch_size

        from evaluation.judging.base_llm_judge import OLLAMA_KEEP_ALIVE
        if OLLAMA_KEEP_ALIVE is None or OLLAMA_KEEP_ALIVE == 0:
            max_batch_workers = 1
            logging.info("Sequential processing (keep_alive=None/0)")
        else:
            llm_cap = getattr(self.config, 'PHASE3_LLM_MAX_WORKERS', 4)
            max_batch_workers = min(llm_cap, num_batches)

        batch_executor = ThreadPoolExecutor(max_workers=max_batch_workers)
        completed_items = self._get_completed_items(result_file)

        try:
            batch_futures = []
            for i in range(0, len(images_to_process), batch_size):
                end_idx = min(i + batch_size, len(images_to_process))
                batch_paths = images_to_process[i:end_idx]
                batch_labels = labels_to_process[i:end_idx]
                batch_image_ids = [path.stem for path in batch_paths]
                batch_context = {
                    "occlusion_level": level,
                    "fill_strategy": strategy,
                    "gen_model": gen_model,
                    "method": method,
                }
                future = batch_executor.submit(
                    judge_model.predict_from_paths,
                    [str(p) for p in batch_paths],
                    batch_labels,
                    image_ids=batch_image_ids,
                    context=batch_context,
                    return_details=True,
                    shared_executor=None
                )
                batch_futures.append((future, batch_paths, batch_labels, i // batch_size))

            batch_results = {}
            future_to_batch = {
                future: (batch_paths, batch_labels, batch_idx)
                for future, batch_paths, batch_labels, batch_idx in batch_futures
            }

            for future in as_completed(future_to_batch.keys()):
                batch_paths, batch_labels, batch_idx = future_to_batch[future]
                try:
                    llm_timeout = getattr(self.config, 'PHASE3_LLM_BATCH_TIMEOUT', 300)
                    result = future.result(timeout=llm_timeout)
                    predictions = result[0] if isinstance(result, tuple) else result
                    batch_results[batch_idx] = (predictions, batch_paths, batch_labels)
                except FutureTimeoutError:
                    logging.error(f"Timeout on batch {batch_idx}")
                    batch_results[batch_idx] = (
                        np.full(len(batch_paths), -1, dtype=np.int64), batch_paths, batch_labels
                    )
                except Exception as e:
                    logging.error(f"Error on batch {batch_idx}: {e}")
                    batch_results[batch_idx] = (
                        np.full(len(batch_paths), -1, dtype=np.int64), batch_paths, batch_labels
                    )

            for batch_idx in sorted(batch_results.keys()):
                predictions, batch_paths, batch_labels = batch_results[batch_idx]
                items_since_save, last_save_time, processed_count = self._record_batch_results(
                    predictions, batch_paths, batch_labels, level, gen_model, method,
                    results_by_level, saved_results, completed_items, result_file,
                    inner_pbar, processed_count, start_time, total_images_to_process,
                    items_since_save, last_save_time, save_interval_items, save_interval_seconds
                )
        finally:
            batch_executor.shutdown(wait=True)

        return items_since_save, last_save_time, processed_count

    # ------------------------------------------------------------------
    # PyTorch evaluation — persistent prefetch pipeline
    # ------------------------------------------------------------------

    def _evaluate_pytorch_level(
        self,
        judge_model: Any,
        images_to_process: List[Path],
        labels_to_process: List[int],
        level: int,
        gen_model: str,
        method: str,
        results_by_level: Dict,
        saved_results: Set,
        result_file: Path,
        inner_pbar: Optional[tqdm],
        processed_count: int,
        start_time: float,
        total_images_to_process: int,
        items_since_save: int,
        last_save_time: float,
        save_interval_items: int,
        save_interval_seconds: float
    ) -> Tuple[int, float, int]:
        batch_size = getattr(self.config, 'PHASE3_BATCH_SIZE_PYTORCH', 512)
        completed_items = self._get_completed_items(result_file)

        # Build list of (paths, labels) batches
        batches = [
            (images_to_process[i:i + batch_size], labels_to_process[i:i + batch_size])
            for i in range(0, len(images_to_process), batch_size)
        ]

        if not batches:
            return items_since_save, last_save_time, processed_count

        # ----------------------------------------------------------
        # Prefetch pipeline:
        #   Use a queue that holds pre-loaded tensors so the GPU
        #   never waits for disk.  We pre-submit up to PREFETCH_AHEAD
        #   batches ahead of the current GPU batch.
        # ----------------------------------------------------------
        PREFETCH_AHEAD = getattr(self.config, 'PHASE3_PREFETCH_AHEAD', 3)

        prefetch_queue: queue.Queue = queue.Queue(maxsize=PREFETCH_AHEAD)
        stop_event = threading.Event()

        def prefetch_worker():
            for batch_paths, batch_labels in batches:
                if stop_event.is_set():
                    break
                images = self._load_images_batch(batch_paths)
                prefetch_queue.put((images, batch_paths, batch_labels))
            prefetch_queue.put(None)  # sentinel

        prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        prefetch_thread.start()

        try:
            while True:
                item = prefetch_queue.get()
                if item is None:
                    break
                batch_images, batch_paths, batch_labels = item

                predictions = self._evaluate_batch(batch_images, judge_model, batch_labels)
                del batch_images  # release memory immediately

                items_since_save, last_save_time, processed_count = self._record_batch_results(
                    predictions, batch_paths, batch_labels, level, gen_model, method,
                    results_by_level, saved_results, completed_items, result_file,
                    inner_pbar, processed_count, start_time, total_images_to_process,
                    items_since_save, last_save_time, save_interval_items, save_interval_seconds
                )
        finally:
            stop_event.set()
            prefetch_thread.join(timeout=10)
            if prefetch_thread.is_alive():
                logging.warning("Prefetch thread did not terminate within 10 s")

        return items_since_save, last_save_time, processed_count

    # ------------------------------------------------------------------
    # Result recording
    # ------------------------------------------------------------------

    def _record_batch_results(
        self,
        predictions: np.ndarray,
        batch_paths: List[Path],
        batch_labels: List[int],
        level: int,
        gen_model: str,
        method: str,
        results_by_level: Dict,
        saved_results: Set,
        completed_items: Set,
        result_file: Path,
        inner_pbar: Optional[tqdm],
        processed_count: int,
        start_time: float,
        total_images_to_process: int,
        items_since_save: int,
        last_save_time: float,
        save_interval_items: int,
        save_interval_seconds: float
    ) -> Tuple[int, float, int]:
        for j, (pred, true_label) in enumerate(zip(predictions, batch_labels)):
            img_id = self._image_id_from_path(batch_paths[j], gen_model, method)
            result_key = (img_id, float(level))

            if result_key in completed_items or result_key in saved_results:
                continue

            is_correct = 1 if (pred == true_label and pred >= 0) else 0
            results_by_level[level].append([img_id, level, is_correct])
            saved_results.add(result_key)
            self._mark_completed(result_file, result_key)
            items_since_save += 1

            if inner_pbar is not None:
                inner_pbar.update(1)
                processed_count += 1
                if processed_count > 10:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    if rate > 0:
                        remaining = (total_images_to_process - processed_count) / rate
                        inner_pbar.set_postfix(
                            {'ETA': f'{remaining:.0f}s', 'rate': f'{rate:.1f} img/s'}
                        )

        current_time = time.time()
        if items_since_save >= save_interval_items or (current_time - last_save_time) >= save_interval_seconds:
            if results_by_level:
                self._save_results(results_by_level, result_file)
                items_since_save = 0
                last_save_time = current_time
                results_by_level.clear()
                saved_results.clear()

        return items_since_save, last_save_time, processed_count

    # ------------------------------------------------------------------
    # Image loading — uses persistent thread pool
    # ------------------------------------------------------------------

    def _load_single_image(self, img_path: Path) -> torch.Tensor:
        return self.transform(Image.open(img_path).convert("RGB"))

    def _load_images_batch(self, image_paths: List[Path]) -> List[torch.Tensor]:
        if len(image_paths) == 1:
            return [self._load_single_image(image_paths[0])]

        if self.load_workers > 1:
            # Reuse the persistent executor — no thread creation overhead per batch
            futures = {
                self._load_executor.submit(self._load_single_image, p): i
                for i, p in enumerate(image_paths)
            }
            images = [None] * len(image_paths)
            for future in as_completed(futures):
                images[futures[future]] = future.result()
            return images

        return [self._load_single_image(p) for p in image_paths]

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def _evaluate_batch(
        self,
        batch_images: List[torch.Tensor],
        judge_model: Any,
        batch_labels: List[int] = None
    ) -> np.ndarray:
        try:
            batch_tensor = prepare_batch_tensor(
                batch_images,
                self.config.DEVICE,
                use_fp16=self.config.USE_FP16_INFERENCE,
                memory_format=torch.channels_last
            )

            with torch.inference_mode():
                if self.config.DEVICE == "cuda" and self.config.USE_FP16_INFERENCE:
                    with torch.amp.autocast(self.config.DEVICE, dtype=torch.float16):
                        outputs = judge_model(batch_tensor)
                elif self.config.DEVICE == "cuda":
                    with torch.amp.autocast(self.config.DEVICE):
                        outputs = judge_model(batch_tensor)
                else:
                    outputs = judge_model(batch_tensor)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                if isinstance(outputs, dict):
                    outputs = outputs.get('logits', outputs)

                predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            del batch_tensor
            # NOTE: empty_cache() intentionally removed — it stalls the GPU.
            # PyTorch manages VRAM automatically; only call it if you hit OOM.
            return predictions

        except Exception as e:
            logging.warning(f"Batch evaluation error: {e}")
            return np.full(len(batch_images), -1, dtype=np.int64)

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def _get_csv_lock(self, result_file: Path) -> Lock:
        key = str(result_file)
        if key not in self.csv_locks:
            self.csv_locks[key] = Lock()
        return self.csv_locks[key]

    def _save_results(self, results_by_level: Dict, result_file: Path):
        all_results = [
            row
            for level in sorted(results_by_level.keys())
            for row in results_by_level[level]
        ]
        if not all_results:
            return

        lock = self._get_csv_lock(result_file)
        with lock:
            self.file_manager.save_csv(
                result_file, all_results,
                header=["image_id", "occlusion_level", "is_correct"],
                append=result_file.exists()
            )

    # Kept for backward compatibility (used internally only)
    def _load_completed_items(self, result_file: Path) -> Set[Tuple[str, float]]:
        return self._get_completed_items(result_file)


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------

def main():
    from core._bootstrap import bootstrap_runner
    config, gpu_manager, file_manager, model_cache, get_cached_model = bootstrap_runner()

    runner = Phase3Runner(config, gpu_manager, file_manager, model_cache)
    runner.run(get_cached_model)


if __name__ == "__main__":
    main()