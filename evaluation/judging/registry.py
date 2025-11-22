"""
Registry for judging model types.

Allows registration and retrieval of judging model factories.
"""

from typing import Dict, Callable, Optional, Any
import logging

from evaluation.judging.base import JudgingModel
from evaluation.judging.pytorch_judge import PyTorchJudgingModel
from evaluation.judging.llamavision import LlamaVisionJudge
from evaluation.judging.binary_llm_judge import BinaryLLMJudge
from evaluation.judging.cosine_llm_judge import CosineSimilarityLLMJudge


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


def create_binary_llm_judge_factory(
    dataset_name: str = "imagenet",
    temperature: float = 0.0
) -> Callable[[str], JudgingModel]:
    """
    Create a factory function for Binary LLM judging models.
    
    Asks yes/no questions for each category.
    
    Args:
        dataset_name: Dataset name to use for class names (default: "imagenet")
        temperature: LLM temperature (default: 0.0 for deterministic)
    
    Returns:
        Factory function that creates BinaryLLMJudge instances
    """
    def factory(model_name: str) -> BinaryLLMJudge:
        return BinaryLLMJudge(
            model_name=model_name,
            dataset_name=dataset_name,
            temperature=temperature
        )
    
    return factory


def create_cosine_llm_judge_factory(
    dataset_name: str = "imagenet",
    temperature: float = 0.1,
    similarity_threshold: float = 0.8,
    embedding_model: str = "nomic-embed-text"
) -> Callable[[str], JudgingModel]:
    """
    Create a factory function for Cosine Similarity LLM judging models.
    
    Asks open-ended questions and computes cosine similarity with class names.
    
    Args:
        dataset_name: Dataset name to use for class names (default: "imagenet")
        temperature: LLM temperature (default: 0.1 for mostly deterministic)
        similarity_threshold: Minimum cosine similarity to accept (default: 0.8)
        embedding_model: Ollama embedding model name (default: "nomic-embed-text")
    
    Returns:
        Factory function that creates CosineSimilarityLLMJudge instances
    """
    def factory(model_name: str) -> CosineSimilarityLLMJudge:
        return CosineSimilarityLLMJudge(
            model_name=model_name,
            dataset_name=dataset_name,
            temperature=temperature,
            similarity_threshold=similarity_threshold,
            embedding_model=embedding_model
        )
    
    return factory

