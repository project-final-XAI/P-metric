"""
Central configuration for CROSS-XAI experiment.

Edit this file to change experiment parameters like models, datasets,
and XAI methods without altering the core logic.
"""

import os
import warnings
from pathlib import Path
import torch

# -----------------
# Environment Setup
# -----------------
# Set KMP_DUPLICATE_LIB_OK to avoid library conflicts on some systems
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configure PyTorch CUDA memory allocator for better performance
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Suppress torch dynamo verbose output
os.environ["TORCHDYNAMO_VERBOSE"] = "0"

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

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
MAX_WORKERS = 8  # Number of data loader workers
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HEATMAP_BATCH_SIZE = 12

# -----------------
# Performance Optimization
# -----------------
# Enable FP16 inference for faster computation (Phase 2 judging models)
# Requires GPU with compute capability >= 7.0 (Volta+)
USE_FP16_INFERENCE = True

# Enable torch.compile for model optimization (PyTorch 2.0+)
# Currently disabled in code for maximum compatibility
USE_TORCH_COMPILE = True

# Set optimal matmul precision for Ampere/Ada GPUs
# Uses TensorFloat-32 (TF32) for better performance on modern GPUs
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

# -----------------
# Progress Tracking Configuration
# -----------------
# Auto-save progress every N completed items (0 to disable)

PROGRESS_AUTO_SAVE_INTERVAL = 200

# Auto-save progress every N seconds (0 to disable)

PROGRESS_AUTO_SAVE_TIME = 600  # 10 minutes

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

# Current dataset to use
DATASET_NAME = "imagenet"

# -----------------
# Model Configuration
# -----------------
# Models used for generating attribution heatmaps (Phase 1)
GENERATING_MODELS = [
    "resnet50",
    # "mobilenet_v2",
    # "vgg16",
    # "vit_b_16",
    # "swin_t",
    # "sipakmed_resnet50.pth",
    # "sipakmed_cropped_ResNet50.pth",
]

# Models used for evaluating occluded images (Phase 2)
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
# Occlusion levels (percentages) to evaluate
OCCLUSION_LEVELS = list(range(5, 100, 5))

# Fill strategies for occluded pixels
FILL_STRATEGIES = [
    "gray",
    # "blur",
    # "random_noise",
    # "black",
    # "mean",
    # "white",
]
