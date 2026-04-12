"""
Shared utilities for all attribution methods.

Single source of truth for DEVICE and model caching.
"""

import torch
import config
from typing import Any, Callable, TypeVar

DEVICE = torch.device(
    config.DEVICE
    if hasattr(config, "DEVICE")
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

T = TypeVar("T")
_MODEL_CACHE: dict[str, Any] = {}


def get_cached_model(key: str, loader_fn: Callable[[], T]) -> T:
    """Return a cached model, calling loader_fn() on first access.

    Args:
        key:       Unique cache key (e.g. "u2net", "dinov2_hf").
        loader_fn: Callable that returns a ready-to-use model.

    Returns:
        The cached model instance.
    """
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = loader_fn()
    return _MODEL_CACHE[key]
