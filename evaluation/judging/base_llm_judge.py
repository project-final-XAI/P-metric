"""
Base class for LLM judges with shared functionality.

Provides common methods for all LLM-based judges to avoid code duplication.
"""

import torch
import tempfile
import os
import logging
from typing import List, Dict
from abc import abstractmethod

from evaluation.judging.LLM_judge_interface import LLMJudgeInterface
from evaluation.judging.llamavision import tensor_to_pil_image


# Maximum parallel workers for all LLM judges
MAX_PARALLEL_WORKERS = 6


class BaseLLMJudge(LLMJudgeInterface):
    """
    Base class for LLM judges providing shared functionality.
    
    Handles:
    - Loading class names from dataset
    - Loading ImageNet class mapping (synset ID -> readable name)
    - Converting tensors to temporary image files
    - Formatting class names for LLM prompts
    """
    
    def __init__(self, model_name: str, dataset_name: str = "imagenet"):
        """
        Initialize base LLM judge.
        
        Args:
            model_name: Ollama model name
            dataset_name: Dataset name
        """
        super().__init__(model_name)
        self.dataset_name = dataset_name
        
        # Load class names
        self.class_names = self._load_class_names()
        if not self.class_names:
            raise ValueError(f"Could not load class names for dataset: {dataset_name}")
        
        # Load ImageNet class mapping if needed
        self.class_name_mapping = self._load_class_mapping()
    
    def _load_class_names(self) -> List[str]:
        """
        Load class names for the dataset.
        
        Returns:
            List of class names (synset IDs for ImageNet, folder names for others)
        """
        from config import DATASET_CONFIG
        
        if self.dataset_name == "imagenet":
            dataset_path = DATASET_CONFIG.get("imagenet", {}).get("path")
            if dataset_path and os.path.exists(dataset_path):
                class_names = sorted([d for d in os.listdir(dataset_path) 
                                    if os.path.isdir(os.path.join(dataset_path, d))])
                if len(class_names) == 1000:
                    return class_names
            
            # Fallback: try ImageFolder
            try:
                from torchvision.datasets import ImageFolder
                if dataset_path and os.path.exists(dataset_path):
                    temp_dataset = ImageFolder(root=str(dataset_path))
                    return temp_dataset.classes
            except:
                pass
            
            logging.warning("Could not load ImageNet class names from dataset.")
            return []
        
        elif self.dataset_name in ["SIPaKMeD", "SIPaKMeD_cropped"]:
            dataset_path = DATASET_CONFIG.get(self.dataset_name, {}).get("path")
            if dataset_path and os.path.exists(dataset_path):
                class_names = sorted([d for d in os.listdir(dataset_path) 
                                    if os.path.isdir(os.path.join(dataset_path, d))])
                return class_names
        
        return []
    
    def _load_class_mapping(self) -> Dict[str, str]:
        """
        Load ImageNet class mapping (synset ID -> readable name).
        
        Returns:
            Dictionary mapping synset IDs to readable names, or empty dict
        """
        if self.dataset_name == "imagenet":
            try:
                from data.imagenet_class_mapping import get_cached_mapping
                mapping = get_cached_mapping()
                logging.info(f"Loaded ImageNet class mapping with {len(mapping)} entries")
                return mapping
            except Exception as e:
                logging.warning(f"Could not load ImageNet mapping: {e}")
                return {}
        return {}
    
    def _format_class_name(self, class_name: str) -> str:
        """
        Format class name for natural language prompt.
        
        Converts synset IDs to readable names for ImageNet.
        
        Args:
            class_name: Raw class name (e.g., 'n01440764' or 'Dyskeratotic')
            
        Returns:
            Human-readable name (e.g., 'tench' or 'Dyskeratotic')
        """
        # Try to get readable name from mapping (for ImageNet synsets)
        if class_name in self.class_name_mapping:
            readable_name = self.class_name_mapping[class_name]
            # Format for LLM: take first part if there's a comma
            from data.imagenet_class_mapping import format_class_for_llm
            return format_class_for_llm(readable_name)
        
        # Replace underscores with spaces for other datasets
        return class_name.replace('_', ' ')
    
    def _tensor_to_temp_file(self, tensor: torch.Tensor) -> str:
        """
        Convert torch tensor to temporary image file.
        
        Args:
            tensor: Image tensor (C, H, W) - normalized
            
        Returns:
            Path to temporary image file
        """
        pil_image = tensor_to_pil_image(tensor)
        temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(temp_fd)
        pil_image.save(temp_path, 'JPEG')
        return temp_path
    
    @abstractmethod
    def predict(self, images: List[torch.Tensor], **kwargs):
        """
        Predict classes for given images.
        
        Must be implemented by subclasses.
        """
        pass

