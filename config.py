"""
Central configuration for CROSS-XAI experiment.

Edit this file to change experiment parameters like models, datasets,
and XAI methods without altering the core logic.
"""


# The code crush some time - why? I don't know!! butttt this line make the code works so its make me happy - so pls even if you are very curious what will happen if you will delete this line - get over this, thx ;)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Some time there is some warns - again, why? I don't know!! but I don't like them so that's line of code make them gone
import  warnings
warnings.filterwarnings("ignore")

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
HEATMAP_BATCH_SIZE = 4

# -----------------
# Storage Optimization
# -----------------
# Save full heatmaps (True) or only sorted indices (False)
# Setting to False saves ~50% disk space (~4GB for 22K heatmaps)
# but you won't be able to visualize heatmaps or do further analysis
SAVE_HEATMAPS = False

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
    # "vgg16",
    "vit_b_16",
    # "swin_t",
]

JUDGING_MODELS = [
    "resnet50",
    # "vit_b_16",
    "mobilenet_v2"
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
    "occlusion",
    "gradientshap",
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
    # "mean",
    # "white",
]