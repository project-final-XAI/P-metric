"""
Batch preparation utilities for occlusion evaluation.

Handles preparing batches of images for occlusion processing.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from pathlib import Path

from core.progress_tracker import ProgressTracker


class BatchPreparer:
    """
    Prepares batches of images for occlusion evaluation.
    
    Handles filtering completed items and organizing data for processing.
    """
    
    def __init__(self, progress: ProgressTracker):
        """
        Initialize batch preparer.
        
        Args:
            progress: ProgressTracker instance for checking completed items
        """
        self.progress = progress
    
    def prepare_occlusion_level(
        self,
        batch_data: List[Dict],
        occlusion_level: float,
        judge_name: str,
        strategy: str
    ) -> Optional[Tuple[List[torch.Tensor], List[np.ndarray], List[int], List[Dict]]]:
        """
        Prepare occlusion data for a specific level.
        
        Filters out already completed items and organizes data for processing.
        
        Args:
            batch_data: List of data dictionaries with keys:
                       - 'gen_model': generating model name
                       - 'method': attribution method name
                       - 'img_id': image identifier
                       - 'image': image tensor
                       - 'label': true label
                       - 'sorted_indices': sorted pixel indices
            occlusion_level: Occlusion level (percentage)
            judge_name: Judging model name
            strategy: Fill strategy name
        
        Returns:
            Tuple of (images_to_process, sorted_indices_list, batch_labels, batch_info)
            or None if no images to process
        """
        images_to_process = []
        sorted_indices_list = []
        batch_labels = []
        batch_info = []
        
        for data in batch_data:
            # Skip if already completed
            if self.progress.is_completed(
                data['gen_model'], data['method'], data['img_id'],
                judge_name, strategy, occlusion_level
            ):
                continue
            
            images_to_process.append(data['image'][0])
            sorted_indices_list.append(data['sorted_indices'])
            batch_labels.append(data['label'])
            batch_info.append(data)
        
        if not images_to_process:
            return None
        
        return images_to_process, sorted_indices_list, batch_labels, batch_info

