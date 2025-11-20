"""
Registry for judging model types.

Allows registration and retrieval of judging model factories.
"""

from typing import Dict, Callable, Optional, Any
import logging

from evaluation.judging.base import JudgingModel
from evaluation.judging.pytorch_judge import PyTorchJudgingModel
from evaluation.judging.llamavision import LlamaVisionJudge


# Registry mapping model names to factory functions
_JUDGING_MODEL_REGISTRY: Dict[str, Callable[[str], JudgingModel]] = {}


def register_judging_model(
    model_name: str,
    factory: Callable[[str], JudgingModel]
) -> None:
    """
    Register a judging model factory.
    
    Args:
        model_name: Name/identifier of the model
        factory: Function that takes model_name and returns JudgingModel instance
    """
    _JUDGING_MODEL_REGISTRY[model_name] = factory
    logging.debug(f"Registered judging model: {model_name}")


def get_judging_model(model_name: str) -> Optional[JudgingModel]:
    """
    Get a judging model instance by name.
    
    Args:
        model_name: Name/identifier of the model
    
    Returns:
        JudgingModel instance or None if not found
    """
    factory = _JUDGING_MODEL_REGISTRY.get(model_name)
    if factory is None:
        return None
    return factory(model_name)


def is_registered(model_name: str) -> bool:
    """
    Check if a model is registered.
    
    Args:
        model_name: Name/identifier of the model
    
    Returns:
        True if model is registered, False otherwise
    """
    return model_name in _JUDGING_MODEL_REGISTRY


def list_registered_models() -> list:
    """
    List all registered model names.
    
    Returns:
        List of registered model names
    """
    return list(_JUDGING_MODEL_REGISTRY.keys())


def create_pytorch_judging_model_factory(
    model_loader: Callable[[str], Any],
    device: str,
    use_fp16: bool = False,
    supports_fp16: Callable[[], bool] = lambda: True
) -> Callable[[str], JudgingModel]:
    """
    Create a factory function for PyTorch judging models.
    
    Args:
        model_loader: Function that loads a PyTorch model given a model name
        device: Device to use ("cuda" or "cpu")
        use_fp16: Whether to convert models to FP16
        supports_fp16: Function that returns True if FP16 is supported
    
    Returns:
        Factory function that creates PyTorchJudgingModel instances
    """
    def factory(model_name: str) -> PyTorchJudgingModel:
        # Load the PyTorch model
        model = model_loader(model_name)
        
        # Convert to FP16 if enabled and supported
        if use_fp16 and device == "cuda" and supports_fp16():
            try:
                model = model.half()
                logging.debug(f"Converted {model_name} to FP16 for judging")
            except Exception as e:
                logging.warning(f"Failed to convert {model_name} to FP16, using FP32: {e}")
        
        return PyTorchJudgingModel(model, model_name, device)
    
    return factory


def create_llamavision_judge_factory(
    dataset_name: str = "imagenet"
) -> Callable[[str], JudgingModel]:
    """
    Create a factory function for LlamaVision judging models.
    
    Args:
        dataset_name: Dataset name to use for class names (default: "imagenet")
    
    Returns:
        Factory function that creates LlamaVisionJudge instances
    """
    def factory(model_name: str) -> LlamaVisionJudge:
        return LlamaVisionJudge(model_name=model_name, dataset_name=dataset_name)
    
    return factory

