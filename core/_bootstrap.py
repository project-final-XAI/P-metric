"""
Shared bootstrap logic for phase runner CLI entry points.

Handles Dependency Injection (DI) and wiring of managers, handlers,
and providers before handing execution over to the Runners.
"""

import sys
import logging
from pathlib import Path


def _setup_environment():
    """Ensure the project root is in PYTHONPATH and setup logging."""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    )


def bootstrap_phase1():
    """
    Bootstrap Phase 1 (Heatmap Generation) with full DI wiring.
    """
    _setup_environment()
    import config as config_module

    from core.gpu_manager import GPUManager
    from core.file_manager import FileManager
    from data.loader import get_dataset_handler
    from models.loader import get_model_provider
    from core.phase1_runner import Phase1Runner

    # 1. Initialize Core Utilities
    gpu_manager = GPUManager()
    file_manager = FileManager(config_module.BASE_DIR)

    # 2. Initialize Data and Model Factories based on configuration
    dataset_handler = get_dataset_handler(config_module.DATASET_NAME)
    model_provider = get_model_provider(config_module.DATASET_NAME)

    # 3. Inject dependencies and return the orchestrated Runner
    return Phase1Runner(
        config=config_module,
        gpu_manager=gpu_manager,
        file_manager=file_manager,
        dataset_handler=dataset_handler,
        model_provider=model_provider
    )

def bootstrap_phase2():
    """
    Bootstrap Phase 2 (Occlusion Generation) with full DI wiring.
    """
    _setup_environment()
    import config as config_module

    from core.gpu_manager import GPUManager
    from core.file_manager import FileManager
    from data.loader import get_dataset_handler
    from models.loader import get_model_provider
    from core.phase2_runner import Phase2Runner

    # 1. Initialize Core Utilities
    gpu_manager = GPUManager()
    file_manager = FileManager(config_module.BASE_DIR)

    # 2. Initialize Data and Model Factories based on configuration
    dataset_handler = get_dataset_handler(config_module.DATASET_NAME)
    model_provider = get_model_provider(config_module.DATASET_NAME)

    # 3. Inject correct architecture dependencies (Matching clean Phase 1 DI)
    return Phase2Runner(
        config=config_module,
        gpu_manager=gpu_manager,
        file_manager=file_manager,
        dataset_handler=dataset_handler,
        model_provider=model_provider
    )


def bootstrap_phase3():
    """
    Bootstrap Phase 3 judging.
    """
    _setup_environment()
    import config as config_module

    from core.gpu_manager import GPUManager
    from core.file_manager import FileManager
    from data.loader import get_dataset_handler
    from models.loader import get_model_provider
    from core.phase3_runner import Phase3Runner

    # 1. Initialize Core Utilities
    gpu_manager = GPUManager()
    file_manager = FileManager(config_module.BASE_DIR)

    # 2. Initialize Data and Model Factories based on configuration
    dataset_handler = get_dataset_handler(config_module.DATASET_NAME)
    model_provider = get_model_provider(config_module.DATASET_NAME)

    # 3. Inject correct architecture dependencies (Matching clean Phase 1 DI)
    return Phase3Runner(
        config=config_module,
        gpu_manager=gpu_manager,
        file_manager=file_manager,
        dataset_handler=dataset_handler,
        model_provider=model_provider
    )


def bootstrap_phase4():
    """
    Bootstrap Phase 3 judging.
    """
    _setup_environment()
    import config as config_module

    from core.gpu_manager import GPUManager
    from core.file_manager import FileManager
    from data.loader import get_dataset_handler
    from models.loader import get_model_provider
    from core.phase4_runner import Phase4Runner

    # 1. Initialize Core Utilities
    gpu_manager = GPUManager()
    file_manager = FileManager(config_module.BASE_DIR)

    # 2. Initialize Data and Model Factories based on configuration
    dataset_handler = get_dataset_handler(config_module.DATASET_NAME)
    model_provider = get_model_provider(config_module.DATASET_NAME)

    # 3. Inject correct architecture dependencies (Matching clean Phase 1 DI)
    return Phase4Runner(
        config=config_module,
        gpu_manager=gpu_manager,
        file_manager=file_manager,
        dataset_handler=dataset_handler,
        model_provider=model_provider
    )