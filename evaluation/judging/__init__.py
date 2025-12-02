"""
Judging model abstraction for evaluation.

Supports different types of judging models (PyTorch, LLM, etc.)
through a unified interface.
"""

from evaluation.judging.base import JudgingModel
from evaluation.judging.pytorch_judge import PyTorchJudgingModel
from evaluation.judging.base_llm_judge import BaseLLMJudge
from evaluation.judging.binary_llm_judge import BinaryLLMJudge
from evaluation.judging.cosine_llm_judge import CosineSimilarityLLMJudge
__all__ = [
    'JudgingModel',
    'PyTorchJudgingModel',
    'BaseLLMJudge',
    'BinaryLLMJudge',
    'CosineSimilarityLLMJudge',
]

