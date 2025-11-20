"""
LlamaVision judge implementation using Ollama.

Asks yes/no questions for each ImageNet class and returns class indices.
"""

import ollama
import torch
import numpy as np
from typing import List
import tempfile
import os
from PIL import Image
import logging

from evaluation.judging.LLM_judge_interface import LLMJudgeInterface


def tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    """
    Convert normalized torch tensor to PIL Image.
    
    Args:
        tensor: Normalized tensor (C, H, W) with ImageNet normalization (can be on GPU or CPU)
        
    Returns:
        PIL Image in RGB format
    """
    # Move tensor to CPU first to avoid device mismatch issues
    img_tensor = tensor.cpu().clone()
    
    # Denormalize ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Denormalize
    img_tensor = img_tensor * std + mean
    
    # Clamp to [0, 1]
    img_tensor = torch.clamp(img_tensor, 0, 1)
    
    # Convert to numpy and scale to 0-255
    img_array = (img_tensor.numpy() * 255).astype(np.uint8)
    
    # Convert from CHW to HWC
    img_array = np.transpose(img_array, (1, 2, 0))
    
    # Convert to PIL Image
    return Image.fromarray(img_array)


class LlamaVisionJudge(LLMJudgeInterface):
    """
    LLM-based judge using Ollama's vision models.
    
    For each image, asks the LLM to identify the real class directly
    and matches the response to the dataset's class names.
    """
    
    def __init__(self, model_name: str, dataset_name: str = "imagenet"):
        """
        Initialize LlamaVision judge.
        
        Args:
            model_name: Ollama model name (e.g., "llama3.2-vision")
            dataset_name: Dataset name to get class names from
        """
        super().__init__(model_name)
        self.dataset_name = dataset_name
        self.class_names = self._load_class_names()
        
        if not self.class_names:
            raise ValueError(f"Could not load class names for dataset: {dataset_name}")
        
        logging.info(f"LlamaVisionJudge initialized with {len(self.class_names)} classes")
    
    def _load_class_names(self) -> List[str]:
        """Load class names for the dataset."""
        from config import DATASET_CONFIG
        
        if self.dataset_name == "imagenet":
            # Try to load from dataset directory
            dataset_path = DATASET_CONFIG.get("imagenet", {}).get("path")
            if dataset_path and os.path.exists(dataset_path):
                class_names = sorted([d for d in os.listdir(dataset_path) 
                                    if os.path.isdir(os.path.join(dataset_path, d))])
                if len(class_names) == 1000:
                    return class_names
            
            # Fallback: Load from torchvision if available
            try:
                # Try to get class names from ImageFolder if dataset exists
                from torchvision.datasets import ImageFolder
                if dataset_path and os.path.exists(dataset_path):
                    temp_dataset = ImageFolder(root=str(dataset_path))
                    return temp_dataset.classes
            except:
                pass
            
            logging.warning("Could not load ImageNet class names from dataset. "
                          "Please ensure ImageNet dataset is properly set up.")
            return []
        
        elif self.dataset_name in ["SIPaKMeD", "SIPaKMeD_cropped"]:
            dataset_path = DATASET_CONFIG.get(self.dataset_name, {}).get("path")
            if dataset_path and os.path.exists(dataset_path):
                class_names = sorted([d for d in os.listdir(dataset_path) 
                                    if os.path.isdir(os.path.join(dataset_path, d))])
                return class_names
        
        return []
    
    def _tensor_to_temp_file(self, tensor: torch.Tensor) -> str:
        """
        Convert torch tensor to temporary image file.
        
        Args:
            tensor: Normalized tensor (C, H, W)
            
        Returns:
            Path to temporary image file
        """
        # Convert tensor to PIL Image
        pil_image = tensor_to_pil_image(tensor)
        
        # Save to temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(temp_fd)  # Close file descriptor, keep file
        
        pil_image.save(temp_path, 'JPEG')
        return temp_path
    
    def predict(self, images: List[torch.Tensor], **kwargs) -> np.ndarray:
        """
        Predict classes for given images by asking the LLM to identify the real class.
        
        For each image, asks the LLM "What class do you see?" and matches the response
        to the dataset's class names.
        
        Args:
            images: List of image tensors (C, H, W) - normalized ImageNet format
            **kwargs: Additional parameters (e.g., 'true_label' for context)
            
        Returns:
            Array of predicted class indices (shape: [batch_size])
        """
        if len(images) == 0:
            return np.array([], dtype=np.int64)
        
        predictions = []
        
        # Ask the LLM to identify the real class of the image directly
        # This is more efficient than asking about all classes
        dataset_type = "ImageNet" if self.dataset_name == "imagenet" else self.dataset_name
        prompt = (
            f"Look at this image. What {dataset_type} class do you see? "
            "Answer with just the class name or synset ID (e.g., 'tench' or 'n01440764'). "
            "If you don't recognize any class from this dataset, say 'none'."
        )
        
        for img_tensor in images:
            try:
                # Convert tensor to temporary file
                temp_image_path = self._tensor_to_temp_file(img_tensor)
                
                try:
                    # Call Ollama API
                    response = ollama.chat(
                        model=self.model_name,
                        messages=[
                            {
                                'role': 'user',
                                'content': prompt,
                                'images': [temp_image_path]
                            }
                        ]
                    )
                    
                    response_text = response.message.content.strip()
                    
                    # Match response to class name
                    class_idx = self._match_class_name(response_text)
                    
                    if class_idx == -1:
                        # No match found - return 0 as fallback
                        logging.debug(f"No class match found in LLM response: {response_text[:100]}")
                        predictions.append(0)
                    else:
                        predictions.append(class_idx)
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)
            
            except Exception as e:
                logging.error(f"Error predicting with LlamaVision: {e}")
                predictions.append(-1)  # Invalid prediction marker
        
        return np.array(predictions, dtype=np.int64)
    
    def _match_class_name(self, response_text: str) -> int:
        """
        Match LLM response text to a class name or synset ID.
        
        Args:
            response_text: LLM response (e.g., "tench" or "n01440764")
            
        Returns:
            Class index if found, -1 otherwise
        """
        response_lower = response_text.lower().strip()
        
        # Check for "none" or "no class"
        if response_lower in ['none', 'no class', 'nothing', 'unknown']:
            return -1
        
        # Try exact match first
        for idx, class_name in enumerate(self.class_names):
            if response_lower == class_name.lower():
                return idx
        
        # Try partial match (response contains class name)
        for idx, class_name in enumerate(self.class_names):
            class_lower = class_name.lower()
            if class_lower in response_lower or response_lower in class_lower:
                return idx
        
        # Try to extract synset ID pattern (n01440764)
        import re
        synset_match = re.search(r'n\d+', response_lower)
        if synset_match:
            synset_id = synset_match.group()
            for idx, class_name in enumerate(self.class_names):
                if synset_id.lower() == class_name.lower():
                    return idx
        
        return -1