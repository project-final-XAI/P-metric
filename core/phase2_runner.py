"""
Phase 2: Occlusion Evaluation Runner.

Evaluates heatmaps by progressively occluding pixels and measuring model accuracy degradation.
Uses pipelining to keep GPU busy while preparing next batches.
"""

import numpy as np
import torch
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future

from core.gpu_manager import GPUManager
from core.file_manager import FileManager
from core.progress_tracker import ProgressTracker
from core.gpu_utils import (
    get_memory_usage, clear_cache_if_needed, sync_and_clear,
    prepare_batch_tensor, warmup_gpu
)
from data.loader import get_dataloader
from evaluation.occlusion import apply_occlusion_batch
from evaluation.judging.base import JudgingModel


class Phase2Runner:
    """Handles Phase 2: Occlusion-based evaluation of heatmaps."""
    
    def __init__(
        self,
        config,
        gpu_manager: GPUManager,
        file_manager: FileManager,
        model_cache: Dict[str, Any]
    ):
        """
        Initialize Phase 2 runner.
        
        Args:
            config: Configuration object
            gpu_manager: GPU resource manager
            file_manager: File path manager
            model_cache: Shared model cache dictionary
        """
        self.config = config
        self.gpu_manager = gpu_manager
        self.file_manager = file_manager
        self.model_cache = model_cache
    
    def run(self, get_cached_model_func):
        """
        Evaluate heatmaps with occlusion for all judge-strategy combinations.
        
        Args:
            get_cached_model_func: Function to get cached model by name
        """
        dataset_name = self.config.DATASET_NAME
        
        # Validate dataset name
        if dataset_name not in self.config.DATASET_CONFIG:
            raise ValueError(
                f"Dataset '{dataset_name}' not found in DATASET_CONFIG. "
                f"Available datasets: {list(self.config.DATASET_CONFIG.keys())}"
            )
        
        try:
            # Initialize progress tracker for fast resume
            with ProgressTracker(
                self.file_manager,
                dataset_name,
                auto_save_interval=self.config.PROGRESS_AUTO_SAVE_INTERVAL,
                auto_save_time=self.config.PROGRESS_AUTO_SAVE_TIME
            ) as progress:
                completed_count = progress.get_completed_count()
                completed_str = f" | Resuming: {completed_count:,} done" if completed_count > 0 else ""
                
                logging.info(f"Starting Phase 2 - Dataset: {dataset_name}{completed_str}")
                
                # Load dataset and judging models
                image_label_map = self._load_dataset(dataset_name)
                judging_models = self._load_judging_models(get_cached_model_func)
                
                # Get and group heatmap files
                heatmap_groups = self._get_heatmap_groups(dataset_name)
                if not heatmap_groups:
                    return
                
                # Process each heatmap group
                self._process_heatmap_groups(
                    heatmap_groups, image_label_map, judging_models, progress, dataset_name
                )
                
                result_dir = self.file_manager.get_result_dir(dataset_name)
                logging.info(f"Phase 2 complete! Results saved to: {result_dir}")
        except Exception as e:
            logging.error(f"Phase 2 failed: {e}")
            raise
    
    def _load_dataset(self, dataset_name: str) -> Dict[str, Tuple[torch.Tensor, int]]:
        """
        Load dataset and create image-label mapping.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary mapping image IDs to (image, label) tuples
        """
        dataloader = get_dataloader(dataset_name, batch_size=1, shuffle=False)
        return {
            f"image_{i:05d}": (img, lbl.item())
            for i, (img, lbl) in enumerate(dataloader)
        }
    
    def _load_judging_models(self, get_cached_model_func) -> Dict[str, Any]:
        """
        Load judging models and convert to FP16 if enabled.
        
        Args:
            get_cached_model_func: Function to get cached model by name
            
        Returns:
            Dictionary mapping judge names to model instances
        """
        judging_models = {
            name: get_cached_model_func(name) for name in self.config.JUDGING_MODELS
        }
        
        # Convert PyTorch models to FP16 for faster inference (skip LLM judges)
        if self.config.USE_FP16_INFERENCE and self.config.DEVICE == "cuda" and self.gpu_manager.supports_fp16():
            for name, model in judging_models.items():
                if isinstance(model, JudgingModel):
                    logging.debug(f"Skipping FP16 conversion for {name} (not a PyTorch model)")
                    continue
                try:
                    judging_models[name] = model.half()
                    logging.debug(f"Converted {name} to FP16 for Phase 2 inference")
                except Exception as e:
                    logging.warning(f"Failed to convert {name} to FP16, using FP32: {e}")
        
        return judging_models
    
    def _get_heatmap_groups(self, dataset_name: str) -> Dict[Tuple[str, str], List[Path]]:
        """
        Get heatmap files grouped by generator model and method.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary mapping (gen_model, method) to list of heatmap file paths
        """
        heatmap_dir = self.file_manager.get_heatmap_dir(dataset_name)
        if not heatmap_dir.exists():
            logging.warning(f"No heatmaps found for {dataset_name}. Run Phase 1 first.")
            return {}
        
        all_files = list(heatmap_dir.glob("*.npy"))
        heatmap_files = [f for f in all_files if f.stem.endswith("_sorted")]
        
        if not heatmap_files:
            logging.warning(f"No heatmaps found for {dataset_name}. Run Phase 1 first.")
            return {}
        
        # Group by generator model and method
        groups = {}
        for heatmap_path in heatmap_files:
            parts = heatmap_path.stem.split('-')
            if len(parts) >= 3:
                gen_model, method = parts[0], parts[1]
                key = (gen_model, method)
                if key not in groups:
                    groups[key] = []
                groups[key].append(heatmap_path)
        
        return groups
    
    def _process_heatmap_groups(
        self,
        heatmap_groups: Dict[Tuple[str, str], List[Path]],
        image_label_map: Dict[str, Tuple[torch.Tensor, int]],
        judging_models: Dict[str, Any],
        progress: ProgressTracker,
        dataset_name: str
    ):
        """
        Process all heatmap groups with progress tracking.
        
        Args:
            heatmap_groups: Dictionary mapping (gen_model, method) to heatmap files
            image_label_map: Dictionary mapping image IDs to (image, label) tuples
            judging_models: Dictionary mapping judge names to model instances
            progress: Progress tracker instance
            dataset_name: Name of the dataset
        """
        with tqdm(total=len(heatmap_groups), desc="Phase 2 Progress", unit="group", leave=True) as pbar:
            for i, ((gen_model, method), group_files) in enumerate(heatmap_groups.items(), 1):
                pbar.set_description(
                    f"Phase 2 [{i}/{len(heatmap_groups)}] {gen_model[:20]}/{method[:20]}"
                )
                try:
                    self._evaluate_heatmap_group(
                        group_files, image_label_map, judging_models,
                        progress, dataset_name, gen_model, method
                    )
                except Exception as e:
                    logging.error(f"Error processing {gen_model}-{method}: {e}")
                finally:
                    pbar.update(1)
    
    def _evaluate_heatmap_group(
        self,
        heatmap_paths: List[Path],
        image_label_map: Dict[str, Tuple[torch.Tensor, int]],
        judging_models: Dict[str, Any],
        progress: ProgressTracker,
        dataset_name: str,
        gen_model: str,
        method: str
    ):
        """
        Evaluate a group of heatmaps (same generator and method).
        
        Args:
            heatmap_paths: List of heatmap file paths
            image_label_map: Dictionary mapping image IDs to (image, label) tuples
            judging_models: Dictionary mapping judge names to model instances
            progress: Progress tracker instance
            dataset_name: Name of the dataset
            gen_model: Generating model name
            method: Attribution method name
        """
        # Prepare batch data from heatmap files
        batch_data = self._prepare_batch_data(heatmap_paths, image_label_map)
        if not batch_data:
            return
        
        # Evaluate for each judge and fill strategy
        for judge_name, judge_model in judging_models.items():
            if judge_name == gen_model:
                continue  # Skip if judge is same as generator
            
            # Clear GPU cache if memory usage is high before starting new judge
            _, current_usage = get_memory_usage()
            if current_usage > 85.0:  # Raised threshold from 80 to 85
                clear_cache_if_needed(threshold_percent=85.0)
                # Skip sync_and_clear() - let it happen naturally
            
            for strategy in self.config.FILL_STRATEGIES:
                self._evaluate_strategy(
                    batch_data, judge_model, judge_name, strategy,
                    progress, dataset_name, gen_model, method
                )
    
    def _prepare_batch_data(
        self,
        heatmap_paths: List[Path],
        image_label_map: Dict[str, Tuple[torch.Tensor, int]]
    ) -> List[Dict]:
        """
        Prepare batch data from heatmap file paths.
        
        Args:
            heatmap_paths: List of heatmap file paths
            image_label_map: Dictionary mapping image IDs to (image, label) tuples
            
        Returns:
            List of dictionaries containing image data and sorted indices
        """
        batch_data = []
        
        for heatmap_path in heatmap_paths:
            try:
                # Parse filename: {gen_model}-{method}-{img_id}_sorted.npy
                parts = heatmap_path.stem.split('-')
                gen_model, method, img_id = parts[0], parts[1], '-'.join(parts[2:])
                
                # Remove '_sorted' suffix if present
                if img_id.endswith('_sorted'):
                    img_id = img_id[:-len('_sorted')]
                
                # Check if image exists in dataset
                if img_id not in image_label_map:
                    logging.warning(
                        f"Image {img_id} not found in dataset, skipping heatmap {heatmap_path.name}"
                    )
                    continue
                
                original_image, true_label = image_label_map[img_id]
                sorted_pixel_indices = np.load(heatmap_path)
                
                batch_data.append({
                    'gen_model': gen_model,
                    'method': method,
                    'img_id': img_id,
                    'image': original_image,
                    'label': true_label,
                    'sorted_indices': sorted_pixel_indices
                })
            except Exception as e:
                logging.warning(f"Error loading {heatmap_path}: {e}")
                continue
        
        return batch_data
    
    def _evaluate_strategy(
        self,
        batch_data: List[Dict],
        judge_model: Any,
        judge_name: str,
        strategy: str,
        progress: ProgressTracker,
        dataset_name: str,
        gen_model: str,
        method: str
    ):
        """
        Evaluate a specific judge/strategy combination for all occlusion levels.
        
        Uses pipelining to keep GPU busy: while GPU evaluates one batch,
        CPU prepares and occludes the next batches in parallel.
        
        Args:
            batch_data: List of dictionaries with image data and sorted indices
            judge_model: Judging model instance
            judge_name: Name of the judging model
            strategy: Fill strategy name
            progress: Progress tracker instance
            dataset_name: Name of the dataset
            gen_model: Generating model name
            method: Attribution method name
        """
        # Warm up GPU memory allocator
        if self.config.DEVICE == "cuda" and batch_data:
            sample_image = batch_data[0]['image'][0]
            warmup_gpu(
                self.config.DEVICE,
                sample_image.shape,
                use_fp16=self.config.USE_FP16_INFERENCE
            )
        
        # Results grouped by occlusion level
        results_by_level = defaultdict(list)
        occlusion_levels = list(self.config.OCCLUSION_LEVELS)
        show_inner_progress = len(occlusion_levels) > 10
        
        # Process occlusion levels with pipelining
        with ThreadPoolExecutor(max_workers=8) as executor:
            pipeline = OcclusionPipeline(
                batch_data, occlusion_levels, strategy, judge_name,
                self.config, self.gpu_manager, progress
            )
            
            for level_idx, occlusion_level in enumerate(
                tqdm(occlusion_levels, desc=f"  → {judge_name[:8]}/{strategy[:6]}",
                     leave=False, disable=not show_inner_progress)
            ):
                # Get occluded batch from pipeline (may be pre-computed)
                occluded_batch = pipeline.get_occluded_batch(level_idx, executor)
                
                if not occluded_batch:
                    continue
                
                masked_images, batch_labels, batch_info = occluded_batch
                
                # Evaluate batch on GPU
                predictions = self._evaluate_batch(masked_images, judge_model, batch_labels)
                del masked_images  # Free memory immediately
                
                # Record results
                batch_progress_items = []
                for pred_idx, (pred, info) in enumerate(zip(predictions, batch_info)):
                    if pred is None or pred < 0:
                        logging.error(
                            f"Skipping invalid prediction for {info['img_id']} at level {occlusion_level}"
                        )
                        continue
                    
                    is_correct = 1 if pred == batch_labels[pred_idx] else 0
                    results_by_level[occlusion_level].append([
                        info['img_id'], occlusion_level, is_correct
                    ])
                    
                    batch_progress_items.append((
                        info['gen_model'], info['method'], info['img_id'],
                        judge_name, strategy, occlusion_level
                    ))
                
                # Batch update progress tracker
                if batch_progress_items:
                    progress.mark_batch_completed(batch_progress_items)
        
        # Save results to file
        if results_by_level:
            self._save_strategy_results(
                results_by_level, dataset_name, gen_model,
                judge_name, method, strategy
            )
            progress.save()
    
    def _evaluate_batch(
        self,
        batch_images: List[torch.Tensor],
        judge_model: Any,
        batch_labels: List[int] = None
    ) -> np.ndarray:
        """
        Evaluate a batch of images with the judging model.
        
        Handles both PyTorch models and JudgingModel instances (LLM judges).
        Automatically splits large batches to fit GPU memory.
        
        Args:
            batch_images: List of image tensors
            judge_model: Judging model instance
            batch_labels: List of true labels (optional, used by some LLM judges)
            
        Returns:
            Array of predicted class indices
        """
        try:
            # Clear cache proactively if memory usage is high
            _, current_usage = get_memory_usage()
            if current_usage >= 80.0:
                clear_cache_if_needed(threshold_percent=80.0)
                _, current_usage = get_memory_usage()  # Re-check after clearing
            
            # Check temperature and adjust throttling
            self.gpu_manager.check_and_throttle()
            
            # Get optimal batch size (consolidated logic: includes memory, thermal, and base size)
            safe_batch_size = self.gpu_manager.get_optimal_inference_batch_size(current_usage)
            
            # Process in chunks if batch is too large
            if len(batch_images) > safe_batch_size:
                return self._evaluate_batch_in_chunks(
                    batch_images, judge_model, safe_batch_size, batch_labels
                )
            else:
                return self._evaluate_batch_chunk(batch_images, judge_model, batch_labels)
        
        except Exception as e:
            logging.warning(f"Batch evaluation error: {e}, falling back to single")
            return self._evaluate_single_fallback(batch_images, judge_model, batch_labels)
    
    def _evaluate_batch_in_chunks(
        self,
        batch_images: List[torch.Tensor],
        judge_model: Any,
        chunk_size: int,
        batch_labels: List[int] = None
    ) -> np.ndarray:
        """
        Evaluate large batch by splitting into smaller chunks.
        
        Args:
            batch_images: List of image tensors
            judge_model: Judging model instance
            chunk_size: Size of each chunk
            batch_labels: List of true labels (optional)
            
        Returns:
            Array of predicted class indices
        """
        all_predictions = []
        num_chunks = (len(batch_images) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(batch_images), chunk_size):
            chunk = batch_images[i:i + chunk_size]
            chunk_labels = batch_labels[i:i + chunk_size] if batch_labels else None
            chunk_predictions = self._evaluate_batch_chunk(chunk, judge_model, chunk_labels)
            all_predictions.extend(chunk_predictions)
            del chunk
            
            # Periodic GPU maintenance every 10 chunks
            chunk_idx = i // chunk_size
            if (chunk_idx > 0 and chunk_idx % 20 == 0):
                _, current_usage = get_memory_usage()
                if current_usage > 88.0:  # Only clear if critically high
                    clear_cache_if_needed(threshold_percent=88.0)
                self.gpu_manager.check_and_throttle()
        
        return np.array(all_predictions)
    
    def _evaluate_batch_chunk(
        self,
        batch_images: List[torch.Tensor],
        judge_model: Any,
        batch_labels: List[int] = None
    ) -> np.ndarray:
        """
        Evaluate a chunk of images (actual batch processing).
        
        Args:
            batch_images: List of image tensors
            judge_model: Judging model instance
            batch_labels: List of true labels (optional)
            
        Returns:
            Array of predicted class indices
        """
        # Handle LLM judges (JudgingModel instances)
        if isinstance(judge_model, JudgingModel):
            try:
                # Pass labels if available (for Binary LLM judge)
                if batch_labels is not None:
                    return judge_model.predict(batch_images, true_labels=batch_labels)
                else:
                    return judge_model.predict(batch_images)
            except Exception as e:
                logging.error(f"Error evaluating with JudgingModel: {e}")
                return np.array([-1] * len(batch_images), dtype=np.int64)
        
        # Handle PyTorch models
        try:
            # Stack images into batch tensor with GPU optimizations
            batch_tensor = prepare_batch_tensor(
                batch_images,
                self.config.DEVICE,
                use_fp16=self.config.USE_FP16_INFERENCE,
                memory_format=torch.channels_last
            )
            
            # Run inference
            with torch.inference_mode():
                if self.config.DEVICE == "cuda" and self.config.USE_FP16_INFERENCE:
                    with torch.amp.autocast(self.config.DEVICE, dtype=torch.float16):
                        outputs = judge_model(batch_tensor)
                elif self.config.DEVICE == "cuda":
                    with torch.amp.autocast(self.config.DEVICE):
                        outputs = judge_model(batch_tensor)
                else:
                    outputs = judge_model(batch_tensor)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                if isinstance(outputs, dict):
                    outputs = outputs['logits']
                
                # Get predictions
                predictions_tensor = torch.argmax(outputs, dim=1)
                predictions = predictions_tensor.cpu().numpy()
            
            del batch_tensor
            return predictions
        
        except RuntimeError as e:
            # Handle CUDA out of memory errors
            if "out of memory" in str(e).lower():
                logging.error(f"CUDA OOM error with batch size {len(batch_images)}")
                logging.error("Clearing cache and retrying with smaller batches...")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Retry with smaller batch size
                if len(batch_images) > 1:
                    mid = len(batch_images) // 2
                    pred1 = self._evaluate_batch_chunk(batch_images[:mid], judge_model)
                    pred2 = self._evaluate_batch_chunk(batch_images[mid:], judge_model)
                    return np.concatenate([pred1, pred2])
                else:
                    logging.error("Failed to evaluate even a single image!")
                    raise e
            else:
                raise e
    
    def _evaluate_single_fallback(
        self,
        batch_images: List[torch.Tensor],
        judge_model: Any,
        batch_labels: List[int] = None
    ) -> np.ndarray:
        """
        Fallback: evaluate images one at a time if batch processing fails.
        
        Args:
            batch_images: List of image tensors
            judge_model: Judging model instance
            batch_labels: List of true labels (optional)
            
        Returns:
            Array of predicted class indices
        """
        predictions = []
        
        for idx, img in enumerate(batch_images):
            try:
                if isinstance(judge_model, JudgingModel):
                    if batch_labels is not None:
                        pred = judge_model.predict([img], true_labels=[batch_labels[idx]])[0]
                    else:
                        pred = judge_model.predict([img])[0]
                    predictions.append(pred)
                else:
                    single_tensor = img.unsqueeze(0).to(self.config.DEVICE)
                    with torch.inference_mode():
                        if self.config.DEVICE == "cuda":
                            with torch.amp.autocast(self.config.DEVICE, dtype=torch.float16):
                                outputs = judge_model(single_tensor)
                        else:
                            outputs = judge_model(single_tensor)
                        
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        if isinstance(outputs, dict):
                            outputs = outputs['logits']
                        pred = torch.argmax(outputs, dim=1).item()
                    predictions.append(pred)
            except Exception as e2:
                logging.warning(f"Single evaluation failed: {e2}")
                predictions.append(-1)  # Invalid prediction marker
        
        return np.array(predictions, dtype=object)
    
    def _save_strategy_results(
        self,
        results_by_level: Dict,
        dataset_name: str,
        gen_model: str,
        judge_model: str,
        method: str,
        strategy: str
    ):
        """
        Save results for a specific strategy to CSV file.
        
        Args:
            results_by_level: Dictionary mapping occlusion level to list of results
            dataset_name: Name of the dataset
            gen_model: Generating model name
            judge_model: Judging model name
            method: Attribution method name
            strategy: Fill strategy name
        """
        result_file = self.file_manager.get_result_file_path(
            dataset_name, gen_model, judge_model, method, strategy
        )
        
        # Flatten results (sorted by occlusion level for consistent output)
        all_results = []
        for level in sorted(results_by_level.keys()):
            all_results.extend(results_by_level[level])
        
        # Skip if no results to save
        if not all_results:
            return
        
        # Save to CSV in one bulk write 
        header = ["image_id", "occlusion_level", "is_correct"]
        self.file_manager.save_csv(
            result_file, all_results, header=header, append=result_file.exists()
        )


class OcclusionPipeline:
    """
    Pipeline for preparing and occluding images in parallel.
    
    This class manages the pipelining logic: while GPU evaluates one batch,
    CPU prepares and occludes the next batches in parallel using ThreadPoolExecutor.
    """
    
    def __init__(
        self,
        batch_data: List[Dict],
        occlusion_levels: List[int],
        strategy: str,
        judge_name: str,
        config,
        gpu_manager: GPUManager,
        progress: ProgressTracker
    ):
        """
        Initialize occlusion pipeline.
        
        Args:
            batch_data: List of dictionaries with image data
            occlusion_levels: List of occlusion percentages to evaluate
            strategy: Fill strategy name
            judge_name: Name of the judging model
            config: Configuration object
            gpu_manager: GPU resource manager
            progress: Progress tracker instance
        """
        self.batch_data = batch_data
        self.occlusion_levels = occlusion_levels
        self.strategy = strategy
        self.judge_name = judge_name
        self.config = config
        self.gpu_manager = gpu_manager
        self.progress = progress
        
        # Pipeline state 
        self.pipeline_depth = min(8, len(occlusion_levels)) 
        self.occluded_cache = {}  # Cache of pre-computed occluded batches
        self.prepared_futures = {}  # Futures for data preparation
        
        # Enable pinned memory for faster CPU→GPU transfers (if CUDA available)
        self.use_pinned_memory = self.config.DEVICE == "cuda"
    
    def get_occluded_batch(
        self,
        level_idx: int,
        executor: ThreadPoolExecutor
    ) -> Tuple[List[torch.Tensor], List[int], List[Dict]]:
        """
        Get occluded batch for a specific occlusion level.
        
        Uses pipelining: if batch is already prepared, return it immediately.
        Otherwise, prepare it synchronously and start preparing next levels.
        
        Args:
            level_idx: Index of occlusion level
            executor: ThreadPoolExecutor for parallel processing
            
        Returns:
            Tuple of (masked_images, batch_labels, batch_info) or None
        """
        occlusion_level = self.occlusion_levels[level_idx]
        
        # Check cache first
        if level_idx in self.occluded_cache:
            return self.occluded_cache.pop(level_idx)
        
        # Prepare data for this level
        data_tuple = self._prepare_occlusion_data(occlusion_level)
        if data_tuple[0] is None:
            return None
        
        # Occlude batch
        occluded_batch = self._occlude_batch(occlusion_level, data_tuple)
        
        # Start preparing next levels in parallel
        self._start_preparing_next_levels(level_idx, executor)
        
        return occluded_batch
    
    def _prepare_occlusion_data(self, occlusion_level: int):
        """
        Prepare data for occlusion: filter out completed items.
        
        Args:
            occlusion_level: Occlusion percentage
            
        Returns:
            Tuple of (images, sorted_indices, labels, info) or None
        """
        
        uncompleted_data = self.progress.filter_batch_uncompleted(
            self.batch_data,
            self.judge_name,
            self.strategy,
            occlusion_level
        )
        
        if not uncompleted_data:
            return None, None, None, None
        
        # Extract data from filtered batch
        images_to_process = [data['image'][0] for data in uncompleted_data]
        sorted_indices_list = [data['sorted_indices'] for data in uncompleted_data]
        batch_labels = [data['label'] for data in uncompleted_data]
        batch_info = uncompleted_data  # Already filtered list
        
        return images_to_process, sorted_indices_list, batch_labels, batch_info
    
    def _occlude_batch(
        self,
        occlusion_level: int,
        data_tuple: Tuple
    ) -> Tuple[List[torch.Tensor], List[int], List[Dict]]:
        """
        Apply occlusion to a batch of images with enhanced GPU transfer.
        
        Uses pinned memory for faster CPU→GPU transfers when available.
        
        Args:
            occlusion_level: Occlusion percentage
            data_tuple: Tuple of (images, sorted_indices, labels, info)
            
        Returns:
            Tuple of (masked_images, batch_labels, batch_info)
        """
        images, sorted_indices, labels, info = data_tuple
        
        try:
            # Pin images in memory for faster CPU→GPU transfer (if enabled)
            if self.use_pinned_memory and images:
                # Check if images are on CPU and pin them
                for i, img in enumerate(images):
                    if img.device.type == 'cpu' and not img.is_pinned():
                        try:
                            images[i] = img.pin_memory()
                        except Exception:
                            pass  # Best effort: continue if pinning fails
            
            # Apply occlusion (already optimized with vectorization)
            masked_images = apply_occlusion_batch(
                images, sorted_indices, occlusion_level, self.strategy
            )
            
            # Apply GPU optimizations: channels_last + FP16
            if self.config.DEVICE == "cuda":
                for i, img in enumerate(masked_images):
                    # Channels-last memory format for better CNN performance
                    if img.ndim == 4:
                        try:
                            masked_images[i] = img.to(
                                memory_format=torch.channels_last, non_blocking=True
                            )
                        except Exception:
                            pass
                    # FP16 conversion for faster inference
                    if self.config.USE_FP16_INFERENCE and self.gpu_manager.supports_fp16():
                        masked_images[i] = masked_images[i].half()
            
            return masked_images, labels, info
        except Exception as e:
            logging.warning(f"Error occluding level {occlusion_level}: {e}")
            return None
    
    def _start_preparing_next_levels(
        self,
        current_idx: int,
        executor: ThreadPoolExecutor
    ):
        """
        Start preparing next occlusion levels in parallel.
        
        Args:
            current_idx: Current occlusion level index
            executor: ThreadPoolExecutor for parallel processing
        """
        # Calculate how many levels ahead we should prepare
        levels_ahead = self.pipeline_depth - len(self.prepared_futures)
        
        # Submit preparation tasks for future levels
        for offset in range(1, levels_ahead + 1):
            next_idx = current_idx + offset
            if next_idx >= len(self.occlusion_levels):
                break
            
            if next_idx in self.occluded_cache:
                continue  # Already prepared
            
            if next_idx not in self.prepared_futures:
                level = self.occlusion_levels[next_idx]
                # Submit data preparation task to thread pool
                future = executor.submit(self._prepare_occlusion_data, level)
                self.prepared_futures[next_idx] = future
        
        # Check completed futures and apply occlusion (CPU-bound work)
        for idx in list(self.prepared_futures.keys()):
            if idx <= current_idx:
                continue  # Skip past levels
            
            future = self.prepared_futures[idx]
            if future.done():
                try:
                    data_tuple = future.result()
                    del self.prepared_futures[idx]
                    
                    if data_tuple[0] is not None:
                        level = self.occlusion_levels[idx]
                        # Apply occlusion immediately to keep cache warm
                        occluded_batch = self._occlude_batch(level, data_tuple)
                        if occluded_batch:
                            self.occluded_cache[idx] = occluded_batch
                except Exception as e:
                    logging.warning(f"Error preparing level {idx}: {e}")
                    del self.prepared_futures[idx]

