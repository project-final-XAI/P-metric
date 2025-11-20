"""
Central configuration for CROSS-XAI experiment.

Edit this file to change experiment parameters like models, datasets,
and XAI methods without altering the core logic.
"""

# The code crush some time - why? I don't know!! butttt this line make the code works so its make me happy - so pls even if you are very curious what will happen if you will delete this line - get over this, thx ;)
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
TORCH_LOGS = ""
os.environ["TORCHDYNAMO_VERBOSE"] = "0"

# Some time there is some warns - again, why? I don't know!! but I don't like them so that's line of code make them gone
import warnings

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
MAX_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HEATMAP_BATCH_SIZE = 12

# -----------------
# Performance Optimization
# -----------------
# Enable FP16 inference for faster computation (Phase 2 judging models)
USE_FP16_INFERENCE = True

# Enable torch.compile for model optimization (PyTorch 2.0+)
USE_TORCH_COMPILE = True

# Set optimal matmul precision for Ampere/Ada GPUs
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')  # Use TensorFloat-32 for better performance

# -----------------
# Progress Tracking Configuration
# -----------------
# Auto-save progress every N completed items (0 to disable)
PROGRESS_AUTO_SAVE_INTERVAL = 100
# Auto-save progress every N seconds (0 to disable)
PROGRESS_AUTO_SAVE_TIME = 300  # 5 minutes

# -----------------
# Dataset Configuration
# -----------------
DATASET_CONFIG = {
    "imagenet": {
        "path": DATA_DIR / "imagenet",
        "num_classes": 1000
    },
    "SIPaKMeD": {
        "path": DATA_DIR / "SIPaKMeD",
        "num_classes": 5
    },
    "SIPaKMeD_cropped": {
        "path": DATA_DIR / "SIPaKMeD_cropped",
        "num_classes": 5
    }
}

DATASET_NAME = "imagenet"

# -----------------
# Model Configuration
# -----------------
GENERATING_MODELS = [
    "resnet50",
    # "mobilenet_v2",
    # "vgg16",
    # "vit_b_16",
    # "swin_t",
    # "sipakmed_resnet50.pth",
    # "sipakmed_cropped_ResNet50.pth",

]

JUDGING_MODELS = [
    # "resnet50",
    # "vit_b_16",
    # "mobilenet_v2",
    # "swin_t",
    # "sipakmed_efficientnetB0.pth"
    # "sipakmed_cropped_efficientnet.pth",
    "llama3.2-vision",
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
    # "c3f",
]

# -----------------
# Occlusion Configuration
# -----------------
OCCLUSION_LEVELS = list(range(5, 100, 5))

FILL_STRATEGIES = [
    "gray",
    # "blur",
    # "random_noise",
    # "black",
    # "mean",
    # "white",
]
