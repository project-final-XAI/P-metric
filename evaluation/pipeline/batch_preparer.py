"""
Batch preparation utilities for occlusion evaluation.

NOTE: This module is deprecated. Batch preparation is now handled directly
in Phase2Runner using CSVProgressChecker. This file is kept for backward
compatibility but is no longer used in the main codebase.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from pathlib import Path


class BatchPreparer:
    """
    DEPRECATED: Batch preparation is now handled in Phase2Runner.
    
    This class is kept for backward compatibility but is no longer used.
    """
    
    def __init__(self, progress_checker):
        """
        Initialize batch preparer.
        
        Args:
            progress_checker: CSVProgressChecker instance (or compatible interface)
        """
        self.progress_checker = progress_checker
    
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
            if self.progress_checker.is_completed(
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

