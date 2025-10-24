"""
Central configuration for CROSS-XAI experiment.

Edit this file to change experiment parameters like models, datasets,
and XAI methods without altering the core logic.
"""

from pathlib import Path
import torch

# -----------------
# Project Paths
# -----------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
HEATMAP_DIR = BASE_DIR / "results" / "heatmaps"
RESULTS_DIR = BASE_DIR / "results" / "evaluation"
ANALYSIS_DIR = BASE_DIR / "results" / "analysis"

# -----------------
# Hardware Configuration
# -----------------
MAX_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------
# Dataset Configuration
# -----------------
DATASET_CONFIG = {
    "imagenet": {
        "path": DATA_DIR / "imagenet",
        "num_classes": 1000
    }
}

# -----------------
# Model Configuration
# -----------------
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
# Attribution Methods Configuration
# -----------------
ATTRIBUTION_METHODS = [
    "saliency",
    "inputxgradient",
    "smoothgrad",
    "guided_backprop",
    "integrated_gradients",
    "gradientshap",
    "occlusion",
    "xrai",
    "grad_cam",
    "guided_gradcam",
    "random_baseline",
]

# -----------------
# Occlusion Configuration
# -----------------
OCCLUSION_LEVELS = list(range(5, 100, 5))

FILL_STRATEGIES = [
    "gray",
    "blur",
    "random_noise",
    "black",
    "mean",
    "white",
]