"""
Phase 3: Super-Fast Evaluation Runner.

Loads pre-generated occluded images from Phase 2 and tests them with judging models.
NO image generation - only loading and testing for maximum efficiency.
"""

import numpy as np
import torch
import logging
import time
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Set
from collections import defaultdict
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from core.gpu_manager import GPUManager
from core.file_manager import FileManager
from core.gpu_utils import prepare_batch_tensor
from core.phase1_runner import Phase1Runner
from core.phase2_runner import Phase2Runner
from data.loader import get_dataloader, get_default_transforms
from evaluation.judging.base import JudgingModel
from evaluation.judging.base_llm_judge import MAX_PARALLEL_WORKERS


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
        self.csv_locks = {}
    
    def run(self, get_cached_model_func):
        """Evaluate pre-generated occluded images with judging models."""
        dataset_name = self.config.DATASET_NAME
        
        if dataset_name not in self.config.DATASET_CONFIG:
            raise ValueError(
                f"Dataset '{dataset_name}' not found in DATASET_CONFIG. "
                f"Available datasets: {list(self.config.DATASET_CONFIG.keys())}"
            )
        
        logging.info(f"Starting Phase 3 - Dataset: {dataset_name}")
        
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
                for method in self.config.ATTRIBUTION_METHODS:
                    for strategy in self.config.FILL_STRATEGIES:
                        for judge_name in self.config.JUDGING_MODELS:
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
        
        logging.info(f"Phase 3 complete! Results saved to: {self.file_manager.get_result_dir(dataset_name)}")
    
    def _ensure_phase2_complete(self, dataset_name: str, get_cached_model_func):
        """Check if Phase 2 is complete, run it for missing items if needed."""
        batch_size = getattr(self.config, 'PHASE2_BATCH_SIZE', 128)
        dataloader = get_dataloader(dataset_name, batch_size=batch_size, shuffle=False)
        total_images = len(dataloader.dataset)
        
        missing_items = []
        for model_name in self.config.GENERATING_MODELS:
            for method_name in self.config.ATTRIBUTION_METHODS:
                for strategy in self.config.FILL_STRATEGIES:
                    for level in self.config.OCCLUSION_LEVELS:
                        occluded_images = self.file_manager.scan_occluded_images(
                            dataset_name, model_name, strategy, method_name, level
                        )
                        if len(occluded_images) < total_images:
                            missing_items.append((model_name, method_name, strategy, level))
                            break
        
        if missing_items:
            logging.info(f"Running Phase 2 for {len(missing_items)} missing combinations...")
            phase2 = Phase2Runner(self.config, self.gpu_manager, self.file_manager, self.model_cache)
            phase2.run(get_cached_model_func)
    
    def _load_dataset_labels(self, dataset_name: str) -> Dict[str, int]:
        """Load dataset and create image ID to label mapping."""
        batch_size = getattr(self.config, 'PHASE3_BATCH_SIZE_PYTORCH', 256)
        dataloader = get_dataloader(dataset_name, batch_size=batch_size, shuffle=False)
        image_label_map = {}
        global_idx = 0
        
        for batch_images, batch_labels in dataloader:
            for lbl in batch_labels:
                image_label_map[f"image_{global_idx:05d}"] = lbl.item()
                global_idx += 1
        
        return image_label_map
    
    def _load_judging_models(self, get_cached_model_func) -> Dict[str, Any]:
        """Load judging models."""
        judging_models = {
            name: get_cached_model_func(name) for name in self.config.JUDGING_MODELS
        }
        
        if self.config.USE_FP16_INFERENCE and self.config.DEVICE == "cuda" and self.gpu_manager.supports_fp16():
            for name, model in judging_models.items():
                if isinstance(model, JudgingModel):
                    continue
                try:
                    judging_models[name] = model.half()
                    logging.debug(f"Converted {name} to FP16 for Phase 3 inference")
                except Exception as e:
                    logging.warning(f"Failed to convert {name} to FP16, using FP32: {e}")
        
        return judging_models
    
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
        """Evaluate a specific combination (gen_model × method × strategy × judge)."""
        result_file = self.file_manager.get_result_file_path(
            dataset_name, gen_model, judge_name, method, strategy
        )
        completed_items = self._load_completed_items(result_file)
        
        # Initialize tracking variables
        results_by_level = defaultdict(list)
        saved_results = set()
        items_since_save = 0
        last_save_time = time.time()
        save_interval_items = 100
        save_interval_seconds = 30
        
        # Count total images and create progress bar
        total_images_to_process = self._count_images_to_process(
            dataset_name, gen_model, method, strategy, completed_items, image_label_map
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
        
        # Process each occlusion level
        is_llm_judge = isinstance(judge_model, JudgingModel)
        for level in self.config.OCCLUSION_LEVELS:
            completed_items = self._load_completed_items(result_file)
            images_to_process, labels_to_process = self._filter_images_to_process(
                dataset_name, gen_model, method, strategy, level,
                completed_items, saved_results, image_label_map
            )
            
            if not images_to_process:
                continue
            
            if is_llm_judge:
                items_since_save, last_save_time, processed_count = self._evaluate_llm_level(
                    judge_model, images_to_process, labels_to_process, level,
                    gen_model, method, results_by_level, saved_results,
                    completed_items, result_file, inner_pbar,
                    processed_count, start_time, total_images_to_process,
                    items_since_save, last_save_time, save_interval_items, save_interval_seconds
                )
            else:
                items_since_save, last_save_time, processed_count = self._evaluate_pytorch_level(
                    judge_model, images_to_process, labels_to_process, level,
                    gen_model, method, results_by_level, saved_results,
                    completed_items, result_file, inner_pbar,
                    processed_count, start_time, total_images_to_process,
                    items_since_save, last_save_time, save_interval_items, save_interval_seconds
                )
        
        # Final save and cleanup
        if results_by_level:
            self._save_results(results_by_level, result_file)
        if inner_pbar is not None:
            inner_pbar.close()
    
    def _count_images_to_process(
        self,
        dataset_name: str,
        gen_model: str,
        method: str,
        strategy: str,
        completed_items: Set[Tuple[str, float]],
        image_label_map: Dict[str, int]
    ) -> int:
        """Count total images that need processing."""
        total = 0
        for level in self.config.OCCLUSION_LEVELS:
            occluded_images = self.file_manager.scan_occluded_images(
                dataset_name, gen_model, strategy, method, level
            )
            for img_path in occluded_images:
                img_id = img_path.stem.replace(f"{gen_model}-{method}-", "")
                if (img_id, level) not in completed_items and img_id in image_label_map:
                    total += 1
        return total
    
    def _filter_images_to_process(
        self,
        dataset_name: str,
        gen_model: str,
        method: str,
        strategy: str,
        level: int,
        completed_items: Set[Tuple[str, float]],
        saved_results: Set[Tuple[str, float]],
        image_label_map: Dict[str, int]
    ) -> Tuple[List[Path], List[int]]:
        """Filter images that need processing for a specific level."""
        occluded_images = self.file_manager.scan_occluded_images(
            dataset_name, gen_model, strategy, method, level
        )
        
        images_to_process = []
        labels_to_process = []
        
        for img_path in occluded_images:
            img_id = img_path.stem.replace(f"{gen_model}-{method}-", "")
            result_key = (img_id, level)
            
            if result_key in completed_items or result_key in saved_results:
                continue
            if img_id not in image_label_map:
                continue
            
            images_to_process.append(img_path)
            labels_to_process.append(image_label_map[img_id])
        
        return images_to_process, labels_to_process
    
    def _evaluate_llm_level(
        self,
        judge_model: Any,
        images_to_process: List[Path],
        labels_to_process: List[int],
        level: int,
        gen_model: str,
        method: str,
        results_by_level: Dict,
        saved_results: Set,
        completed_items: Set,
        result_file: Path,
        inner_pbar: tqdm,
        processed_count: int,
        start_time: float,
        total_images_to_process: int,
        items_since_save: int,
        last_save_time: float,
        save_interval_items: int,
        save_interval_seconds: float
    ) -> Tuple[int, float, int]:
        """Evaluate a level using LLM judge (optimized pipeline processing)."""
        batch_size = getattr(self.config, 'PHASE3_BATCH_SIZE_LLM', 32)
        shared_executor = ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS)
        
        try:
            # Submit all batches immediately
            batch_futures = []
            for i in range(0, len(images_to_process), batch_size):
                end_idx = min(i + batch_size, len(images_to_process))
                batch_paths = images_to_process[i:end_idx]
                batch_labels = labels_to_process[i:end_idx]
                
                future = shared_executor.submit(
                    judge_model.predict_from_paths,
                    [str(path) for path in batch_paths],
                    batch_labels,
                    shared_executor=shared_executor
                )
                batch_futures.append((future, batch_paths, batch_labels, i // batch_size))
            
            # Process batches as they complete
            batch_results = {}
            future_to_batch = {
                future: (batch_paths, batch_labels, batch_idx)
                for future, batch_paths, batch_labels, batch_idx in batch_futures
            }
            
            for future in as_completed(future_to_batch.keys()):
                batch_paths, batch_labels, batch_idx = future_to_batch[future]
                try:
                    predictions = future.result()
                    batch_results[batch_idx] = (predictions, batch_paths, batch_labels)
                except Exception as e:
                    logging.error(f"Error processing batch {batch_idx}: {e}")
                    predictions = np.array([-1] * len(batch_paths), dtype=np.int64)
                    batch_results[batch_idx] = (predictions, batch_paths, batch_labels)
            
            # Record results in order
            for batch_idx in sorted(batch_results.keys()):
                predictions, batch_paths, batch_labels = batch_results[batch_idx]
                items_since_save, last_save_time, processed_count = self._record_batch_results(
                    predictions, batch_paths, batch_labels, level, gen_model, method,
                    results_by_level, saved_results, completed_items, result_file,
                    inner_pbar, processed_count, start_time, total_images_to_process,
                    items_since_save, last_save_time, save_interval_items, save_interval_seconds
                )
        finally:
            shared_executor.shutdown(wait=False)
        
        return items_since_save, last_save_time, processed_count
    
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
        completed_items: Set,
        result_file: Path,
        inner_pbar: tqdm,
        processed_count: int,
        start_time: float,
        total_images_to_process: int,
        items_since_save: int,
        last_save_time: float,
        save_interval_items: int,
        save_interval_seconds: float
    ) -> Tuple[int, float, int]:
        """Evaluate a level using PyTorch model (with prefetching)."""
        batch_size = getattr(self.config, 'PHASE3_BATCH_SIZE_PYTORCH', 512)
        next_batch_images = None
        next_batch_paths = None
        next_batch_labels = None
        next_future = None
        
        for i in range(0, len(images_to_process), batch_size):
            end_idx = min(i + batch_size, len(images_to_process))
            batch_paths = images_to_process[i:end_idx]
            batch_labels = labels_to_process[i:end_idx]
            
            # Use prefetched batch if available
            if next_batch_images is not None:
                batch_images = next_batch_images
                batch_paths = next_batch_paths
                batch_labels = next_batch_labels
            else:
                batch_images = self._load_images_batch(batch_paths)
            
            # Prefetch next batch
            if i + batch_size < len(images_to_process):
                next_end_idx = min(i + batch_size * 2, len(images_to_process))
                next_batch_paths = images_to_process[i + batch_size:next_end_idx]
                next_batch_labels = labels_to_process[i + batch_size:next_end_idx]
                if self.load_workers > 1:
                    executor = ThreadPoolExecutor(max_workers=1)
                    next_future = executor.submit(self._load_images_batch, next_batch_paths)
                else:
                    next_future = None
            else:
                next_future = None
            
            # Evaluate batch
            predictions = self._evaluate_batch(batch_images, judge_model, batch_labels)
            
            # Get prefetched batch if ready
            if next_future is not None:
                next_batch_images = next_future.result()
                executor.shutdown(wait=False)
            else:
                next_batch_images = None
            
            # Memory cleanup
            del batch_images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Record results
            items_since_save, last_save_time, processed_count = self._record_batch_results(
                predictions, batch_paths, batch_labels, level, gen_model, method,
                results_by_level, saved_results, completed_items, result_file,
                inner_pbar, processed_count, start_time, total_images_to_process,
                items_since_save, last_save_time, save_interval_items, save_interval_seconds
            )
        
        return items_since_save, last_save_time, processed_count
    
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
        inner_pbar: tqdm,
        processed_count: int,
        start_time: float,
        total_images_to_process: int,
        items_since_save: int,
        last_save_time: float,
        save_interval_items: int,
        save_interval_seconds: float
    ) -> Tuple[int, float, int]:
        """Record results for a batch and handle periodic saves."""
        # Process all images in batch
        for j, (pred, true_label) in enumerate(zip(predictions, batch_labels)):
            img_id = batch_paths[j].stem.replace(f"{gen_model}-{method}-", "")
            if not img_id:
                img_id = batch_paths[j].stem
            
            result_key = (img_id, level)
            if result_key in completed_items or result_key in saved_results:
                continue
            
            is_correct = 1 if pred == true_label else 0
            if pred < 0:
                is_correct = 0
            
            result_row = [img_id, level, is_correct]
            results_by_level[level].append(result_row)
            saved_results.add(result_key)
            items_since_save += 1
            
            # Update progress bar
            if inner_pbar is not None:
                inner_pbar.update(1)
                processed_count += 1
                if processed_count > 10:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    if rate > 0:
                        remaining = (total_images_to_process - processed_count) / rate
                        inner_pbar.set_postfix({'ETA': f'{remaining:.0f}s', 'rate': f'{rate:.1f} img/s'})
        
        # Periodic save - check once after processing entire batch (more efficient)
        current_time = time.time()
        time_since_last_save = current_time - last_save_time
        if items_since_save >= save_interval_items or time_since_last_save >= save_interval_seconds:
            if results_by_level:
                self._save_results(results_by_level, result_file)
                completed_items.update(self._load_completed_items(result_file))
                completed_items.update(saved_results)
                items_since_save = 0
                last_save_time = current_time
                results_by_level.clear()
                saved_results.clear()
        
        return items_since_save, last_save_time, processed_count
    
    def _load_single_image(self, img_path: Path) -> torch.Tensor:
        """Load a single image and convert to tensor."""
        pil_image = Image.open(img_path).convert("RGB")
        return self.transform(pil_image)
    
    def _load_images_batch(self, image_paths: List[Path]) -> List[torch.Tensor]:
        """Load images from disk and convert to tensors (parallel loading)."""
        if len(image_paths) == 1:
            return [self._load_single_image(image_paths[0])]
        
        if self.load_workers > 1:
            with ThreadPoolExecutor(max_workers=self.load_workers) as executor:
                futures = {
                    executor.submit(self._load_single_image, img_path): idx
                    for idx, img_path in enumerate(image_paths)
                }
                images = [None] * len(image_paths)
                for future in as_completed(futures):
                    idx = futures[future]
                    images[idx] = future.result()
        else:
            images = [self._load_single_image(img_path) for img_path in image_paths]
        
        return images
    
    def _evaluate_batch(
        self,
        batch_images: List[torch.Tensor],
        judge_model: Any,
        batch_labels: List[int] = None
    ) -> np.ndarray:
        """Evaluate a batch of images with the judging model."""
        try:
            if isinstance(judge_model, JudgingModel):
                batch_images_cpu = []
                has_gpu_images = False
                for img in batch_images:
                    if img.is_cuda:
                        has_gpu_images = True
                        batch_images_cpu.append(img.detach().cpu())
                    else:
                        batch_images_cpu.append(img)
                
                if has_gpu_images:
                    torch.cuda.synchronize()
                
                if batch_labels is not None:
                    return judge_model.predict(batch_images_cpu, true_labels=batch_labels)
                else:
                    return judge_model.predict(batch_images_cpu)
            
            # PyTorch models
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
                
                predictions_tensor = torch.argmax(outputs, dim=1)
                predictions = predictions_tensor.cpu().numpy()
            
            del batch_tensor
            return predictions
            
        except Exception as e:
            logging.warning(f"Batch evaluation error: {e}")
            return np.array([-1] * len(batch_images), dtype=np.int64)
    
    def _load_completed_items(self, result_file: Path) -> Set[Tuple[str, float]]:
        """Load already completed items from CSV file (thread-safe)."""
        if not result_file.exists():
            return set()
        
        lock = self._get_csv_lock(result_file)
        with lock:
            rows = self.file_manager.load_csv(result_file, skip_header=True)
        
        completed = set()
        for row in rows:
            if len(row) >= 2:
                try:
                    img_id = row[0]
                    occlusion_level = float(row[1])
                    completed.add((img_id, occlusion_level))
                except (ValueError, IndexError):
                    continue
        
        return completed
    
    def _get_csv_lock(self, result_file: Path) -> Lock:
        """Get or create a lock for a specific CSV file."""
        file_str = str(result_file)
        if file_str not in self.csv_locks:
            self.csv_locks[file_str] = Lock()
        return self.csv_locks[file_str]
    
    def _save_results(self, results_by_level: Dict, result_file: Path):
        """Save results to CSV file (thread-safe)."""
        all_results = []
        for level in sorted(results_by_level.keys()):
            all_results.extend(results_by_level[level])
        
        if not all_results:
            return
        
        header = ["image_id", "occlusion_level", "is_correct"]
        lock = self._get_csv_lock(result_file)
        with lock:
            self.file_manager.save_csv(
                result_file, all_results, header=header, append=result_file.exists()
            )


def main():
    """Simple main function to run Phase 3."""
    import sys
    import logging
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config
    from core.gpu_manager import GPUManager
    from core.file_manager import FileManager
    from models.loader import load_model
    from evaluation.judging.binary_llm_judge import BinaryLLMJudge
    from evaluation.judging.cosine_llm_judge import CosineSimilarityLLMJudge
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    gpu_manager = GPUManager()
    file_manager = FileManager(config.BASE_DIR)
    model_cache = {}
    
    def get_cached_model(name):
        if name not in model_cache:
            if name.endswith('-binary'):
                model_cache[name] = BinaryLLMJudge(name, config.DATASET_NAME, 0.0)
            elif name.endswith('-cosine'):
                model_cache[name] = CosineSimilarityLLMJudge(name, config.DATASET_NAME, 0.1, 0.8, "nomic-embed-text")
            else:
                model_cache[name] = load_model(name)
        return model_cache[name]
    
    runner = Phase3Runner(config, gpu_manager, file_manager, model_cache)
    runner.run(get_cached_model)


if __name__ == "__main__":
    main()
