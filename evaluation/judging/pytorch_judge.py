"""
PyTorch judging model implementation.

Wraps PyTorch models to implement the JudgingModel interface.
"""

import torch
import numpy as np
from typing import List, Union, Any
import logging

from evaluation.judging.base import JudgingModel


class PyTorchJudgingModel(JudgingModel):
    """
    PyTorch model wrapper for judging.
    
    Handles PyTorch-specific details like device placement,
    precision conversion, and output format normalization.
    """
    
    def __init__(self, model: torch.nn.Module, model_name: str, device: str = "cuda"):
        """
        Initialize PyTorch judging model.
        
        Args:
            model: PyTorch model (already on device, in eval mode)
            model_name: Name/identifier of the model
            device: Device where model is located ("cuda" or "cpu")
        """
        super().__init__(model_name)
        self.model = model
        self.device = device
    
    def predict(
        self,
        images: Union[List[torch.Tensor], torch.Tensor],
        **kwargs
    ) -> np.ndarray:
        """
        Predict classes for given images.
        
        Args:
            images: List of image tensors (C, H, W) or batched tensor (B, C, H, W)
            **kwargs: Additional parameters (ignored for PyTorch models)
        
        Returns:
            Array of predicted class indices (shape: [batch_size])
        """
        # Handle list of tensors
        if isinstance(images, list):
            if len(images) == 0:
                return np.array([], dtype=np.int64)
            # Stack into batch tensor
            batch_tensor = torch.stack(images)
        else:
            batch_tensor = images
        
        # Ensure tensor is on correct device
        if batch_tensor.device.type != self.device:
            batch_tensor = batch_tensor.to(self.device, non_blocking=True)
        
        # Convert to channels_last format if supported (better GPU performance)
        if batch_tensor.ndim == 4:
            try:
                batch_tensor = batch_tensor.to(memory_format=torch.channels_last, non_blocking=True)
            except Exception:
                pass
        
        # Run inference
        with torch.inference_mode():
            outputs = self.model(batch_tensor)
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if isinstance(outputs, dict):
                outputs = outputs.get('logits', outputs)
            
            # Get predictions
            predictions_tensor = torch.argmax(outputs, dim=1)
            predictions = predictions_tensor.cpu().numpy()
        
        return predictions
    
    def __repr__(self) -> str:
        return f"PyTorchJudgingModel(model_name='{self.model_name}', device='{self.device}')"

