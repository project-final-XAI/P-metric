"""
Base class for judging models.

Provides a unified interface for different types of judging models
(PyTorch models, LLMs, etc.) to enable easy extensibility.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Any, Optional
import numpy as np


class JudgingModel(ABC):
    """
    Abstract base class for judging models.
    
    All judging models must implement the predict() method which takes
    images and returns predictions. Additional parameters can be passed
    for model-specific needs (e.g., prompts for LLMs).
    """
    
    def __init__(self, model_name: str):
        """
        Initialize judging model.
        
        Args:
            model_name: Name/identifier of the model
        """
        self.model_name = model_name
    
    @abstractmethod
    def predict(
        self,
        images: Union[List, np.ndarray, Any],
        **kwargs
    ) -> np.ndarray:
        """
        Predict classes for given images.
        
        Args:
            images: Input images (format depends on model type)
            **kwargs: Additional parameters for model-specific needs
                     (e.g., 'true_label' for LLM prompts, 'prompt' for custom prompts)
        
        Returns:
            Array of predicted class indices (shape: [batch_size])
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"

