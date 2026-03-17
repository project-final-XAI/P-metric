import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from scipy.ndimage import gaussian_filter
# import config
from attribution.base import AttributionMethod

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_PATCH_SIZE, _ATTN_SIZE = 14, (518, 518)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)


class Dinov2AttentionMethod(AttributionMethod):
    _transform = transforms.Compose([
        transforms.Resize(_ATTN_SIZE), transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    def __init__(self) -> None:
        super().__init__("dinov2_attention")
        self.use_registers = True #getattr(config, "DINO_ATTENTION_USE_REGISTERS", True)
        self.smooth_sigma = 0.0 #getattr(config, "DINO_ATTENTION_SMOOTH_SIGMA", 0.0)
        self._dino_model = None

    def _ensure_model(self):
        if self._dino_model is None:
            model_name = "dinov2_vits14_reg" if self.use_registers else "dinov2_vits14"
            self._dino_model = torch.hub.load("facebookresearch/dinov2", model_name).to(DEVICE).eval()

    def compute(self, model, images, targets):
        self._ensure_model()
        B, _, H, W = images.shape

        # FIX: Ensure mean/std are on the same device as the input images
        device = images.device
        mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)

        # Now this operation will work regardless of CPU or GPU
        imgs = F.interpolate((images * std + mean), size=_ATTN_SIZE, mode="bilinear", align_corners=False)
        imgs = (imgs - mean) / std

        with torch.no_grad():
            # Some torch.hub versions of DINOv2 require the input to be exactly on the model's device
            attn = self._dino_model.get_last_selfattention(imgs.to(DEVICE))

        attn_cls = attn[:, :, 0, :].mean(dim=1)  # Average heads
        grid_h, grid_w = _ATTN_SIZE[0] // _PATCH_SIZE, _ATTN_SIZE[1] // _PATCH_SIZE
        patch_attn = attn_cls[:, -(grid_h * grid_w):].reshape(B, grid_h, grid_w).cpu().numpy()

        heatmaps = []
        for i in range(B):
            m = _normalize(patch_attn[i])
            if self.smooth_sigma > 0:
                m = _normalize(gaussian_filter(m, sigma=self.smooth_sigma))
            heatmaps.append(cv2.resize(m, (W, H)))

        return torch.from_numpy(np.stack(heatmaps)).float()