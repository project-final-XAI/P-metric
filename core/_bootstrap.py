"""
Shared bootstrap logic for phase runner CLI entry points.

Eliminates duplicated setup code across phase1/phase2/phase3 main() functions.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

from core.gpu_manager import GPUManager
from core.file_manager import FileManager


def bootstrap_runner() -> tuple:
    """Set up the common runtime context for any phase runner CLI entry.

    Returns:
        (config_module, gpu_manager, file_manager, model_cache, get_cached_model_func)
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config
    from models.loader import load_model
    from evaluation.judging.binary_llm_judge import BinaryLLMJudge
    from evaluation.judging.cosine_llm_judge import CosineSimilarityLLMJudge
    from evaluation.judging.classid_llm_judge import ClassIdLLMJudge

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    gpu_manager = GPUManager()
    file_manager = FileManager(config.BASE_DIR)
    model_cache: Dict[str, Any] = {}

    def get_cached_model(name: str):
        if name not in model_cache:
            if name.endswith('-binary'):
                model_cache[name] = BinaryLLMJudge(name, config.DATASET_NAME, 0.0)
            elif name.endswith('-cosine'):
                model_cache[name] = CosineSimilarityLLMJudge(
                    name, config.DATASET_NAME, 0.1, 0.8, "nomic-embed-text",
                )
            elif name.endswith('-classid'):
                model_cache[name] = ClassIdLLMJudge(name, config.DATASET_NAME, 0.0)
            else:
                model_cache[name] = load_model(name)
        return model_cache[name]

    return config, gpu_manager, file_manager, model_cache, get_cached_model
