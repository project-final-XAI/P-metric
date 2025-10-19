# config.py

import torch
from torchvision import models
from torchvision.models import (
    ResNet18_Weights, ResNet50_Weights, MobileNet_V2_Weights,
    ViT_B_16_Weights, vit_b_16
)

# ============================================================
# CONFIGURATION
# ============================================================
# Masking mode for Stage 2 occlusion implementation:
#  - "fixed":         replace masked pixels with DEFAULT_MASK_COLOR (after normalizing)
#  - "imagenet_mean": replace masked pixels with the normalized ImageNet mean (neutral baseline)
#  - "random":        replace masked pixels with random noise (strong baseline)
DEFAULT_MASK_COLOR = (1., 1., 1.)  # grey
MASK_MODE = "random"

# -------------------- MODEL SELECTION --------------------
MODEL_CONFIG = {
    "resnet18": {
        "model_func": models.resnet18,
        "weights": ResNet18_Weights.DEFAULT,
        "gradcam_layer": "layer4",
        "model_type": "cnn",
    },
    "resnet50": {
        "model_func": models.resnet50,
        "weights": ResNet50_Weights.DEFAULT,
        "gradcam_layer": "layer4",
        "model_type": "cnn",
    },
    "mobilenet_v2": {
        "model_func": models.mobilenet_v2,
        "weights": MobileNet_V2_Weights.DEFAULT,
        "gradcam_layer": "features.18",
        "model_type": "cnn",
    },
    "vit_b_16": {
        "model_func": vit_b_16,
        "weights": ViT_B_16_Weights.DEFAULT,
        "gradcam_layer": "encoder.layers.encoder_layer_11.ln_1",
        "model_type": "vit",
    },
}

# Choose which model to run: "resnet18", "resnet50", "mobilenet_v2", or "vit_b_16"
MODEL_NAME = "mobilenet_v2"

CNN_MODEL_FUNC = MODEL_CONFIG[MODEL_NAME]["model_func"]
CNN_WEIGHTS = MODEL_CONFIG[MODEL_NAME]["weights"]
GRADCAM_TARGET_LAYER = MODEL_CONFIG[MODEL_NAME]["gradcam_layer"]
MODEL_TYPE = MODEL_CONFIG[MODEL_NAME]["model_type"]

# Occlusion hyperparameters (Captum Occlusion for Stage 1)
OCCL_WINDOW = 24
OCCL_STRIDE = 12
OCCL_BATCH_SIZE = 16

# Expected Grad-CAM (averaging Grad-CAM across noisy baselines) — CNN only
EG_NUM_BASELINES = 32
EG_ALPHA_RANGE   = (0.30, 1.00)
EG_NOISE_IMG_STD = 0.03
EG_SMOOTH        = True
EG_KERNEL_SIZE   = 7
EG_SIGMA         = 2.0

# ============================================================
# DEVICE & NORMALIZATION CONSTANTS
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization constants (used for denorm, mask color normalization, etc.)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

# Paths & thresholds (edit as needed)
INPUT_FOLDER   = r"./images/imagenet"
HEATMAP_FOLDER = r"./images/heatmaps-mobilenet_v2"
OUTPUT_FOLDER  = r"./images/mobilenet_v2\random"
THRESHOLDS     = [round(i * 0.05, 2) for i in range(1, 20)]  # 5%..95%