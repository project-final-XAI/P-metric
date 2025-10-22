# config.py
"""
Central configuration file for the CROSS-XAI project.
Edit this file to change experiment parameters like models, datasets,
and XAI methods without altering the core logic.
"""

from pathlib import Path

import torch
from torchvision.models import vit_b_16

# -----------------
# Project Paths
# -----------------
# Base path of the project
BASE_DIR = Path(__file__).parent

# Directory to store datasets
DATA_DIR = BASE_DIR / "data"

# Directory to save generated heatmaps (Phase 1 output)
HEATMAP_DIR = BASE_DIR / "results" / "heatmaps"

# Directory to save evaluation results (Phase 2 output)
RESULTS_DIR = BASE_DIR / "results" / "evaluation"

# Directory to save final plots and metrics (Phase 3 output)
ANALYSIS_DIR = BASE_DIR / "results" / "analysis"

# -----------------
# Hardware Configuration
# -----------------
MAX_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# For monitor:              'nvidia-smi'

# -----------------
# Dataset Configuration
# -----------------
# To add a new dataset, add its configuration here and implement its
# loading logic in utils/data_loader.py
DATASET_CONFIG = {
    "imagenet": {
        "path": DATA_DIR / "imagenet",
        "num_classes": 1000
    },
    # Example for future extension:
    # "sipakmed": {
    #     "path": DATA_DIR / "sipakmed",
    #     "num_classes": 5
    # }
}

# -----------------
# Model Configuration
# -----------------
# List of models to be used as generators (F) and judges (J).
# To add a new model, simply add its name to the list.
# The `model_loader` will handle fetching from torchvision/timm.
# For custom models, you can define a path.
GENERATING_MODELS = [
    "resnet50",
    "mobilenet_v2",
    "vgg16",
    "vit_b_16",
    "swin_t",
]

JUDGING_MODELS = [
    "resnet50",
    "vit_b_16",
    "swin_t",
]

# -----------------
# XAI (Attribution) Methods Configuration
# -----------------
# List of attribution methods (A) to evaluate.
# The name here should correspond to a function in `modules/attribution_generator.py`
ATTRIBUTION_METHODS = [
    "saliency",
    "integrated_gradients",
    "guided_backprop",
    "grad_cam",
    "saliency_mask",
    "integrated_gradients_mask",
    "guided_backprop_mask",
    "gradcam_mask_once",
    "vit_gradcam_token",
    "inputxgradient_mask",
    "guided_gradcam_mask",
    "smoothgrad_mask",
    "gradientshap_mask",
    "xrai_mask",
    "occlusion_mask",
    "naive_occ_mask",
]

# -----------------
# Occlusion Configuration
# -----------------
# Occlusion levels (P) in percentage of pixels to remove.
OCCLUSION_LEVELS = list(range(5, 100, 5))

# Fill strategies (S) for occluded regions.
# The name should correspond to a strategy in `modules/occlusion_evaluator.py`
FILL_STRATEGIES = [
    "gray",
    "blur",
    "random_noise",
    "black",
    "mean",
    "white",
]
