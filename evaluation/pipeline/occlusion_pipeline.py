"""
Occlusion processing pipeline.

Simplifies the complex pipelining logic for occlusion evaluation.
"""

from typing import List, Dict, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, Future
from tqdm import tqdm
import torch
import numpy as np
import logging

from evaluation.occlusion import apply_occlusion_batch
from evaluation.pipeline.batch_preparer import BatchPreparer


class OcclusionPipeline:
    """
    Manages occlusion processing pipeline with async data preparation.
    
    Handles pipelining of data preparation and occlusion application
    to keep GPU continuously busy.
    """
    
    def __init__(
        self,
        batch_preparer: BatchPreparer,
        device: str,
        use_fp16: bool,
        supports_fp16: Callable[[], bool],
        pipeline_depth: int = 3
    ):
        """
        Initialize occlusion pipeline.
        
        Args:
            batch_preparer: BatchPreparer instance
            device: Device to use ("cuda" or "cpu")
            use_fp16: Whether FP16 is enabled
            supports_fp16: Function that returns True if FP16 is supported
            pipeline_depth: Number of levels to pipeline ahead
        """
        self.batch_preparer = batch_preparer
        self.device = device
        self.use_fp16 = use_fp16
        self.supports_fp16 = supports_fp16
        self.pipeline_depth = pipeline_depth
    
    def process_levels(
        self,
        batch_data: List[Dict],
        occlusion_levels: List[float],
        strategy: str,
        judge_name: str,
        show_progress: bool = False
    ):
        """
        Process all occlusion levels with pipelining.
        
        Yields tuples of (occlusion_level, masked_images, batch_labels, batch_info)
        for each level that has work to do.
        
        Args:
            batch_data: List of data dictionaries for batch
            occlusion_levels: List of occlusion levels to process
            strategy: Fill strategy name
            judge_name: Judging model name
            show_progress: Whether to show progress bar
        
        Yields:
            Tuples of (level, masked_images, batch_labels, batch_info)
        """
        # Limit pipeline depth to number of levels
        pipeline_depth = min(self.pipeline_depth, len(occlusion_levels))
        
        with ThreadPoolExecutor(max_workers=pipeline_depth + 2) as executor:
            # Queue of occluded batches ready for GPU (already processed)
            occluded_queue: Dict[int, Tuple] = {}
            # Queue of data preparation futures
            data_queue: List[Tuple[int, Future]] = []
            
            # Helper to occlude a batch
            def occlude_batch(level: float, data_tuple: Optional[Tuple]) -> Optional[Tuple]:
                """Apply occlusion to a prepared batch."""
                if data_tuple is None or data_tuple[0] is None:
                    return None
                
                images, sorted_indices, labels, info = data_tuple
                try:
                    masked = apply_occlusion_batch(images, sorted_indices, level, strategy)
                    
                    # Optimize for GPU if using CUDA
                    if self.device == "cuda":
                        for i, img in enumerate(masked):
                            if img.ndim == 4:
                                try:
                                    masked[i] = img.to(memory_format=torch.channels_last, non_blocking=True)
                                except Exception:
                                    pass
                            if self.use_fp16 and self.supports_fp16():
                                masked[i] = masked[i].half()
                    
                    return (masked, labels, info)
                except Exception as e:
                    logging.warning(f"Error occluding level {level}: {e}")
                    return None
            
            # Pre-prepare and pre-occlude first levels
            for i in range(min(pipeline_depth, len(occlusion_levels))):
                level = occlusion_levels[i]
                data_future = executor.submit(
                    self.batch_preparer.prepare_occlusion_level,
                    batch_data, level, judge_name, strategy
                )
                data_queue.append((i, data_future))
            
            # Process each level
            for idx, level in enumerate(tqdm(
                occlusion_levels,
                desc=f"  → {judge_name[:8]}/{strategy[:6]}",
                leave=False,
                disable=not show_progress
            )):
                # Check occlusion futures that completed
                self._check_completed_futures(occluded_queue)
                
                # Get occluded batch from queue if ready, otherwise process synchronously
                result = self._get_or_process_batch(
                    idx, level, occluded_queue, executor, occlude_batch, batch_data,
                    judge_name, strategy
                )
                
                if result is None:
                    continue
                
                masked_images, batch_labels, batch_info = result
                
                # Process completed data prep -> start occlusion in background
                self._process_data_queue(
                    data_queue, occluded_queue, idx, occlusion_levels, executor, occlude_batch
                )
                
                # Start preparing next levels to keep pipeline full
                self._fill_pipeline(
                    data_queue, occluded_queue, idx, occlusion_levels, executor, batch_data,
                    judge_name, strategy
                )
                
                if not masked_images:
                    continue
                
                yield level, masked_images, batch_labels, batch_info
    
    def _check_completed_futures(self, occluded_queue: Dict[int, Tuple]) -> None:
        """Check and resolve completed futures in occluded queue."""
        for occlude_idx in list(occluded_queue.keys()):
            item = occluded_queue[occlude_idx]
            if isinstance(item, Future):
                if item.done():
                    result = item.result()
                    if result is not None:
                        occluded_queue[occlude_idx] = result
                    else:
                        del occluded_queue[occlude_idx]
    
    def _get_or_process_batch(
        self,
        idx: int,
        level: float,
        occluded_queue: Dict[int, Tuple],
        executor: ThreadPoolExecutor,
        occlude_batch: Callable,
        batch_data: List[Dict],
        judge_name: str,
        strategy: str
    ) -> Optional[Tuple]:
        """Get batch from queue or process synchronously."""
        if idx in occluded_queue:
            item = occluded_queue.pop(idx)
            if isinstance(item, Future):
                if item.done():
                    result = item.result()
                    if result is None:
                        # Fallback to synchronous processing
                        return self._process_synchronously(
                            level, batch_data, judge_name, strategy, occlude_batch
                        )
                    return result
                else:
                    # Future not done yet - process synchronously
                    return self._process_synchronously(
                        level, batch_data, judge_name, strategy, occlude_batch
                    )
            else:
                # Already processed result tuple
                return item
        else:
            # Process synchronously (fallback)
            return self._process_synchronously(
                level, batch_data, judge_name, strategy, occlude_batch
            )
    
    def _process_synchronously(
        self,
        level: float,
        batch_data: List[Dict],
        judge_name: str,
        strategy: str,
        occlude_batch: Callable
    ) -> Optional[Tuple]:
        """Process a level synchronously."""
        data = self.batch_preparer.prepare_occlusion_level(
            batch_data, level, judge_name, strategy
        )
        if data is None or data[0] is None:
            return None
        return occlude_batch(level, data)
    
    def _process_data_queue(
        self,
        data_queue: List[Tuple[int, Future]],
        occluded_queue: Dict[int, Tuple],
        current_idx: int,
        occlusion_levels: List[float],
        executor: ThreadPoolExecutor,
        occlude_batch: Callable
    ) -> None:
        """Process completed data preparation futures."""
        for data_idx, data_future in list(data_queue):
            if data_future.done():
                data_queue.remove((data_idx, data_future))
                if data_idx not in occluded_queue and data_idx > current_idx:
                    level = occlusion_levels[data_idx]
                    occlude_future = executor.submit(occlude_batch, level, data_future.result())
                    occluded_queue[data_idx] = occlude_future
    
    def _fill_pipeline(
        self,
        data_queue: List[Tuple[int, Future]],
        occluded_queue: Dict[int, Tuple],
        current_idx: int,
        occlusion_levels: List[float],
        executor: ThreadPoolExecutor,
        batch_data: List[Dict],
        judge_name: str,
        strategy: str
    ) -> None:
        """Fill pipeline with next levels to keep it busy."""
        next_idx = current_idx + len(data_queue) + len([k for k in occluded_queue.keys() if k > current_idx]) + 1
        while len(data_queue) < self.pipeline_depth and next_idx < len(occlusion_levels):
            level = occlusion_levels[next_idx]
            data_future = executor.submit(
                self.batch_preparer.prepare_occlusion_level,
                batch_data, level, judge_name, strategy
            )
            data_queue.append((next_idx, data_future))
            next_idx += 1

