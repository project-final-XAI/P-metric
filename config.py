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
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# Remove deprecated variable to avoid warnings
if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
    del os.environ["PYTORCH_CUDA_ALLOC_CONF"]

# Suppress torch dynamo verbose output
os.environ["TORCHDYNAMO_VERBOSE"] = "0"

# Suppress httpx (ollama) verbose logging
os.environ["HTTPX_LOG_LEVEL"] = "WARNING"

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
MODELS_DIR = BASE_DIR / "models"

# -----------------
# Hardware Configuration
# -----------------
MAX_WORKERS = 8  # Number of data loader workers
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HEATMAP_BATCH_SIZE = 12
PHASE2_BATCH_SIZE = 256  # Batch size for Phase 2 occlusion processing (GPU)
PHASE2_SAVE_WORKERS = 8  # Number of workers for parallel image saving (CPU)
PHASE3_BATCH_SIZE_PYTORCH = 512  # Batch size for Phase 3 PyTorch model evaluation (GPU) - increased for better GPU utilization
PHASE3_BATCH_SIZE_LLM = 1  # Batch size for Phase 3 LLM model evaluation (CPU/API) - smaller batches = continuous processing, no gaps
PHASE3_LOAD_WORKERS = 8  # Number of workers for parallel image loading in Phase 3
PHASE3_SAVE_INTERVAL_ITEMS = 50   # Checkpoint to CSV after this many new results
PHASE3_SAVE_INTERVAL_SECONDS = 120  # … or after this many seconds, whichever comes first
PHASE3_LLM_BATCH_TIMEOUT = 300   # Per-batch timeout (seconds) for LLM judges
PHASE3_PREFETCH_AHEAD = 3        # Prefetch depth for PyTorch evaluation pipeline
PHASE3_LLM_MAX_WORKERS = 4       # Max parallel workers for LLM batch evaluation

# Input image size for occlusion masks (must match the dataloader transform)
OCCLUSION_IMAGE_SHAPE = (224, 224)

# -----------------
# GPU VRAM Tier Thresholds (GB)
# -----------------
# Used by GPUManager to scale batch sizes based on available GPU memory.
VRAM_TIER_HIGH = 22    # >= this: "very high" tier
VRAM_TIER_MID = 16     # >= this: "high" tier
VRAM_TIER_LOW = 8      # >= this: "standard" tier; below: "low" tier

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
        "path": DATA_DIR / "SIPaKMed_cropped",
        "num_classes": 5
    }
}

# Current dataset to use
DATASET_NAME = "imagenet"
# DATASET_NAME = "SIPaKMeD_cropped"

# -----------------
# Model Configuration
# -----------------
# Models used for generating attribution heatmaps (Phase 1)
GENERATING_MODELS = [
    "resnet50",
    "mobilenet_v2",
    "vgg16",

    # "vit_b_16",
    # "swin_t",

    # "sipakmed_cropped_efficientnet.pth",
    # "sipakmed_cropped_ResNet50.pth",
]

# Models used for evaluating occluded images (Phase 2)
JUDGING_MODELS = [
    "resnet50",
    "mobilenet_v2",
    "vgg16",

    # "vit_b_16",
    # "swin_t",

    # "sipakmed_cropped_efficientnet.pth",
    # "sipakmed_cropped_ResNet50.pth",

    # "llama3.2-vision-binary",
    # "llama3.2-vision-cosine",
    # "llama3.2-vision-classid",
]

# -----------------
# Attribution Methods Configuration
# -----------------
ATTRIBUTION_METHODS = [
    # --- Model-dependent (need the classifier) ---
    "saliency",
    "inputxgradient",
    # "smoothgrad",
    "guided_backprop",
    "integrated_gradients",
    "occlusion",
    "gradientshap",
    "xrai",
    "grad_cam",
    "guided_gradcam",
    "random_baseline",

    # --- Model-independent (DINO / U2Net) ---
    "dinov2_attention",
    # "dinov2_PC1",
    # "dinov2_PC_EV",
    # "dinov2_PC_L2",
    # "dinov2_COMBO_FIXED",
    # "dinov2_ENT",
    "dinov2_COMBO_ENT_SMOOTH",
    "U2Net-Saliency",
    "u2net_dino_fusion",

    # --- Continuous wrappers ---
    # "saliency_continuous",
    "inputxgradient_continuous",
    # "guided_backprop_continuous",
    # "integrated_gradients_continuous",
    # "gradientshap_continuous",
    # "occlusion_continuous",
    # "xrai_continuous",
    # "grad_cam_continuous",
    # "guided_gradcam_continuous",
    # "random_baseline_continuous",
    # "u2net_saliency_continuous",
    # "u2net_dino_fusion_continuous",

    # --- U2Net underlay + XAI fill ---
    # "saliency_u2net_fill",
    "inputxgradient_u2net_fill",
    # "guided_backprop_u2net_fill",
    # "integrated_gradients_u2net_fill",
    # "gradientshap_u2net_fill",
    # "occlusion_u2net_fill",
    # "xrai_u2net_fill",
    # "grad_cam_u2net_fill",
    # "guided_gradcam_u2net_fill",
    # "random_baseline_u2net_fill",
]

# -----------------
# DINOv2 Configuration
# -----------------
# Shared DINO model used by both DINO methods and U2Net+DINO fusion.
DINO_MODEL_NAME = "facebook/dinov2-with-registers-base"
# ViT-B with registers exposes 4 register tokens.
DINO_NUM_REGISTERS = 4
# "eager" is required for output_attentions (SDPA returns None → DINO XAI methods crash).
DINO_ATTN_IMPLEMENTATION = "eager"

# -----------------
# Occlusion Configuration
# -----------------
# Occlusion levels (percentages) to evaluate
OCCLUSION_LEVELS = list[int](range(0, 105, 5))
# OCCLUSION_LEVELS =  [20, 40, 60, 80, 95,100]

# Fill strategies for occluded pixels
FILL_STRATEGIES = [
    "gray",
    "blur",
    "random_noise",
    "black",
    "mean",
    "white",
]

# -----------------
# Heatmap Visualization Configuration
# -----------------
# Colormap for regular heatmap visualization (Phase 1)
# Options: "hot", "jet", "viridis", "rainbow", "turbo"
HEATMAP_COLORMAP = "hot"
