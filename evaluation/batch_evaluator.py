"""
Batch evaluation for judging models.

Handles efficient batch processing with memory management and error handling.
"""

from typing import List
import numpy as np
import torch
import logging

from evaluation.judging.base import JudgingModel
from evaluation.pipeline.gpu_pipeline import BatchSizeManager


class BatchEvaluator:
    """
    Evaluates batches of images using judging models.
    
    Handles batch size optimization, memory management, and error recovery.
    """
    
    def __init__(self, batch_size_manager: BatchSizeManager):
        """
        Initialize batch evaluator.
        
        Args:
            batch_size_manager: BatchSizeManager instance for batch size calculation
        """
        self.batch_size_manager = batch_size_manager
    
    def evaluate(
        self,
        batch_images: List[torch.Tensor],
        judge_model: JudgingModel
    ) -> np.ndarray:
        """
        Evaluate a batch of images with the judging model.
        
        Automatically handles batch size optimization, chunking, and error recovery.
        
        Args:
            batch_images: List of image tensors to evaluate
            judge_model: JudgingModel instance
        
        Returns:
            Array of predicted class indices (shape: [batch_size])
        """
        try:
            # Get optimal batch size based on current GPU conditions
            max_batch_size = self.batch_size_manager.get_optimal_batch_size()
            
            # Clear cache if memory usage is high
            self.batch_size_manager.clear_cache_if_needed(threshold_percent=80.0)
            
            # Process in chunks if batch is too large
            if len(batch_images) > max_batch_size:
                return self._evaluate_in_chunks(batch_images, judge_model, max_batch_size)
            else:
                return self._evaluate_chunk(batch_images, judge_model)
        
        except Exception as e:
            logging.warning(f"Batch evaluation error: {e}, falling back to single")
            return self._evaluate_single_fallback(batch_images, judge_model)
    
    def _evaluate_in_chunks(
        self,
        batch_images: List[torch.Tensor],
        judge_model: JudgingModel,
        max_batch_size: int
    ) -> np.ndarray:
        """
        Evaluate batch in smaller chunks.
        
        Args:
            batch_images: List of image tensors
            judge_model: JudgingModel instance
            max_batch_size: Maximum batch size per chunk
        
        Returns:
            Array of all predictions
        """
        all_predictions = []
        num_chunks = (len(batch_images) + max_batch_size - 1) // max_batch_size
        
        for i in range(0, len(batch_images), max_batch_size):
            chunk = batch_images[i:i + max_batch_size]
            # Process chunk - GPU works asynchronously, no sync needed
            chunk_predictions = self._evaluate_chunk(chunk, judge_model)
            all_predictions.extend(chunk_predictions)
            
            # Explicitly delete chunk to free memory immediately
            del chunk
            
            # Periodic cache clearing - only every 10 chunks or at end
            chunk_idx = i // max_batch_size
            if (chunk_idx > 0 and chunk_idx % 10 == 0) or (chunk_idx == num_chunks - 1):
                # Check memory usage before clearing
                if self.batch_size_manager.should_clear_cache(threshold_percent=85.0):
                    self.batch_size_manager.clear_cache_if_needed(threshold_percent=85.0)
                # Check temperature
                self.batch_size_manager.gpu_manager.check_and_throttle()
        
        return np.array(all_predictions)
    
    def _evaluate_chunk(
        self,
        batch_images: List[torch.Tensor],
        judge_model: JudgingModel
    ) -> np.ndarray:
        """
        Evaluate a chunk of images (actual batch processing).
        
        Args:
            batch_images: List of image tensors
            judge_model: JudgingModel instance
        
        Returns:
            Array of predicted class indices
        """
        try:
            # Use JudgingModel.predict() interface - handles all model-specific details
            predictions = judge_model.predict(batch_images)
            return predictions
        except RuntimeError as e:
            # Handle CUDA out of memory errors
            if "out of memory" in str(e).lower():
                logging.error(f"CUDA OOM error with batch size {len(batch_images)}")
                logging.error("Clearing cache and retrying with smaller batches...")
                
                # Emergency cache clear
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Retry with smaller batch size (divide by 2)
                if len(batch_images) > 1:
                    mid = len(batch_images) // 2
                    logging.info(f"Splitting batch into {mid} + {len(batch_images) - mid}")
                    pred1 = self._evaluate_chunk(batch_images[:mid], judge_model)
                    pred2 = self._evaluate_chunk(batch_images[mid:], judge_model)
                    return np.concatenate([pred1, pred2])
                else:
                    # Single image failed - this is serious
                    logging.error("Failed to evaluate even a single image!")
                    raise e
            else:
                raise e
    
    def _evaluate_single_fallback(
        self,
        batch_images: List[torch.Tensor],
        judge_model: JudgingModel
    ) -> np.ndarray:
        """
        Fallback to single-image evaluation if batch evaluation fails.
        
        Args:
            batch_images: List of image tensors
            judge_model: JudgingModel instance
        
        Returns:
            Array of predictions (may contain -1 for failed evaluations)
        """
        predictions = []
        for img in batch_images:
            try:
                # Use JudgingModel.predict() interface
                pred = judge_model.predict([img])[0]
                predictions.append(pred)
            except Exception as e2:
                logging.warning(f"Single evaluation failed: {e2}")
                predictions.append(-1)  # Invalid prediction marker
        return np.array(predictions, dtype=object)

