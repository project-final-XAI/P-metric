"""
attribution_methods.py
======================
Concrete ModelIndependentMethod implementations for:
  - DinoAttribution        : DINOv2 ViT-L/14+reg, 3-signal blend, guided filter at out_H, TTA flip
  - Dino448Attribution     : same model but guided filter runs at full 448px for sharper boundaries
  - U2NetAttribution       : U2Net salient-object detection, native 320px inference
  - DinoU2NetAttribution   : 50/50 average of DINO + U2Net (blended at input H×W by default)
  - DinoU2NetSumAttribution: clamped sum (bounded OR) of DINO + U2Net
  - DinoU2Net320Attribution: hybrid blended on U2Net's native 320px grid
  - Dino448U2NetAttribution: DINO-448 downsampled to 320 then averaged with U2Net

All are drop-in subclasses of ModelIndependentMethod and integrate with the
AttributionMethod.compute() lifecycle:
  raw attribution → normalize [0,1] → upscale to original resolution.

Prerequisites
-------------
pip install torch torchvision scipy Pillow numpy
"""

from __future__ import annotations

from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms

from attribution.base import ModelIndependentMethod
import config

MODELS_DIR = getattr(config, "MODELS_DIR", None)
WEIGHTS_PATH = MODELS_DIR / "u2net.pth"

# ---------------------------------------------------------------------------
# Shared constants  (mirror stage1_all_methods.py)
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_PATCH_SIZE = 14          # DINOv2 ViT-L/14 patch size
_DINO_SIZE  = 448         # inference resolution → 32×32 patch grid
_U2NET_SIZE = 320         # U2Net native inference resolution
_GF_RADIUS  = 8           # guided-filter radius (at 224px; scaled automatically)
_GF_EPS     = 1e-2        # guided-filter regularisation

# ---------------------------------------------------------------------------
# Device + lazy model cache
# ---------------------------------------------------------------------------

DEVICE = torch.device(
    config.DEVICE
    if hasattr(config, "DEVICE")
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

_MODEL_CACHE: dict[str, nn.Module] = {}   # "dino" | "u2net" → loaded model


def _ensure_dino() -> nn.Module:
    """Load and cache DINOv2 ViT-L/14+reg (called once on first use)."""
    if "dino" not in _MODEL_CACHE:
        print(f"[AviMethods] Loading dinov2_vitl14_reg on {DEVICE} …")
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", verbose=False)
        _MODEL_CACHE["dino"] = m.to(DEVICE).eval()
    return _MODEL_CACHE["dino"]


from attribution.model_independent.unet_based import _ensure_u2net


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _norm01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8) if hi > lo else np.zeros_like(arr)


def _guided_filter(
    guide: torch.Tensor,   # (B, C, H, W)  RGB in [0, 1]
    src:   torch.Tensor,   # (B, 1, H, W)
    r:     int   = _GF_RADIUS,
    eps:   float = _GF_EPS,
) -> torch.Tensor:
    """Joint / guided bilateral filter implemented with box convolutions."""
    # Convert guide to greyscale intensity
    I = (0.299 * guide[:, 0:1]
       + 0.587 * guide[:, 1:2]
       + 0.114 * guide[:, 2:3])

    def box(t: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(t, kernel_size=2 * r + 1, stride=1, padding=r)

    mean_I  = box(I)
    mean_p  = box(src)
    var_I   = box(I * I)   - mean_I * mean_I
    cov_Ip  = box(I * src) - mean_I * mean_p
    a       = cov_Ip / (var_I + eps)
    b       = mean_p - a * mean_I
    return (box(a) * I + box(b)).clamp(0.0, 1.0)


def _map_entropy(scores_01: np.ndarray, N: int) -> float:
    """Normalised Shannon entropy of a score distribution."""
    p = scores_01.astype(np.float64) + 1e-10
    p /= p.sum()
    return float(-np.sum(p * np.log(p)) / np.log(N))


# ---------------------------------------------------------------------------
# DINO signal extraction
# ---------------------------------------------------------------------------

def _dino_extract(
    dino_model: nn.Module,
    x: torch.Tensor,  # (1, 3, H, W)  — already at DINO inference size
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single forward pass through the last DINOv2 block.
    Returns three per-patch salience signals, each shape (N,) in [0, 1]:
      s_attn : entropy-weighted CLS→patch attention
      s_norm : patch-token feature-norm
      s_pop  : patch popularity (mean cross-patch attention)
    """
    H, W = x.shape[-2], x.shape[-1]
    N    = (H // _PATCH_SIZE) * (W // _PATCH_SIZE)
    holder: dict[str, torch.Tensor] = {}

    def _hook(module, input, output):
        B, T, C = input[0].shape
        qkv = module.qkv(input[0])
        qkv = qkv.reshape(B, T, 3, module.num_heads,
                           C // module.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        attn = (q @ k.transpose(-2, -1)) * ((C // module.num_heads) ** -0.5)
        holder["attn"] = attn.softmax(dim=-1).detach()

    handle = dino_model.blocks[-1].attn.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            out = dino_model.forward_features(x)
    finally:
        handle.remove()

    # ── patch-token feature norms ──────────────────────────────────────────
    if isinstance(out, dict) and "x_norm_patchtokens" in out:
        patches = out["x_norm_patchtokens"][0].float()
    elif isinstance(out, dict) and "patch_tokens" in out:
        patches = out["patch_tokens"][0].float()
    else:
        raise ValueError("Cannot extract patch tokens from forward_features().")

    s_norm = _norm01(patches.norm(dim=-1).cpu().numpy().astype(np.float32))

    # ── entropy-weighted CLS attention ────────────────────────────────────
    attn        = holder["attn"][0]           # (heads, T, T)
    T_           = attn.shape[-1]
    patch_start = T_ - N
    cls_rows    = attn[:, 0, patch_start:]    # (heads, N)

    head_w = np.array([
        max(0.0, 1.0 - _map_entropy(
            _norm01(cls_rows[h].cpu().numpy().astype(np.float32)), N))
        for h in range(cls_rows.shape[0])
    ], dtype=np.float32)
    denom  = head_w.sum()
    head_w = head_w / denom if denom > 1e-8 else np.ones_like(head_w) / len(head_w)

    hw_t   = torch.from_numpy(head_w).to(attn.device)
    s_attn = _norm01((cls_rows * hw_t[:, None]).sum(0).cpu().numpy().astype(np.float32))

    # ── patch popularity ──────────────────────────────────────────────────
    p2p   = attn[:, patch_start:, patch_start:]
    s_pop = _norm01(p2p.sum(dim=1).mean(dim=0).cpu().numpy().astype(np.float32))

    return s_attn, s_norm, s_pop


def _dino_single_pass(
    dino_model: nn.Module,
    x_dino:    torch.Tensor,   # (1, 3, DINO_SIZE, DINO_SIZE)
    guide_rgb: torch.Tensor,   # (1, 3, out_H, out_W)  — guide for GF
    gf_radius: int,
) -> torch.Tensor:
    """One forward pass → guided-filtered heatmap (1, 1, out_H, out_W)."""
    gh     = x_dino.shape[-2] // _PATCH_SIZE
    gw     = x_dino.shape[-1] // _PATCH_SIZE
    N      = gh * gw
    out_H, out_W = guide_rgb.shape[-2], guide_rgb.shape[-1]

    s_attn, s_norm, s_pop = _dino_extract(dino_model, x_dino)

    w_attn = max(0.0, 1.0 - _map_entropy(s_attn, N))
    w_norm = max(0.0, 1.0 - _map_entropy(s_norm, N))
    w_pop  = max(0.0, 1.0 - _map_entropy(s_pop,  N))
    denom  = w_attn + w_norm + w_pop
    wa, wn, wp = ((1/3, 1/3, 1/3) if denom < 1e-8
                  else (w_attn/denom, w_norm/denom, w_pop/denom))

    scores = _norm01(wa * s_attn + wn * s_norm + wp * s_pop)
    hm     = torch.from_numpy(scores).float().reshape(1, 1, gh, gw).to(x_dino.device)
    hm     = F.interpolate(hm, size=(out_H, out_W), mode="bilinear", align_corners=False)
    hm     = _guided_filter(guide_rgb, hm, r=gf_radius)
    lo, hi = hm.min(), hm.max()
    return (hm - lo) / (hi - lo + 1e-8)


# U2Net architecture classes removed; imported _ensure_u2net directly from unet_based.


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

_imagenet_normalize = transforms.Normalize(
    mean=_IMAGENET_MEAN,
    std=_IMAGENET_STD,
)

def _denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalisation, return RGB in [0, 1]."""
    mean = torch.tensor(_IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor(_IMAGENET_STD,  device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def _resize_batch(x: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# DinoAttribution
# ---------------------------------------------------------------------------

class DinoAttribution(ModelIndependentMethod):
    """
    DINOv2 ViT-L/14+registers saliency.

    Signal pipeline (per image):
      1. Resize to dino_size (448) → 32×32 patch grid.
      2. Extract s_attn (entropy-weighted CLS attention),
                 s_norm (patch feature norms),
                 s_pop  (patch popularity / cross-patch attention).
      3. Entropy-weight the three signals and blend.
      4. Bilinear upsample 32×32 → input H×W.
      5. Joint guided filter using the original RGB as guide.
      6. Test-time flip (horizontal) — average original + h-flipped.
    """

    def __init__(
        self,
        dino_size:  int   = _DINO_SIZE,
        gf_radius:  int   = _GF_RADIUS,
        gf_eps:     float = _GF_EPS,
        tta_flip:   bool  = True,
    ):
        super().__init__(name="dino")
        self.dino_size = dino_size
        self.gf_radius = gf_radius
        self.gf_eps    = gf_eps
        self.tta_flip  = tta_flip

    def _get_model(self) -> nn.Module:
        return _ensure_dino()

    def compute_independent(self, images: torch.Tensor) -> torch.Tensor:
        paths = getattr(self, "current_paths", None)
        if paths is not None:
            model = self._get_model()
            heatmaps = []
            to_tensor = transforms.ToTensor()
            for path in paths:
                img_pil_orig = Image.open(path).convert("RGB")
                img_pil = img_pil_orig.resize((224, 224), Image.LANCZOS)
                guide_224 = to_tensor(img_pil).unsqueeze(0).to(DEVICE)
                x_448 = to_tensor(img_pil_orig.resize((self.dino_size, self.dino_size), Image.LANCZOS)).unsqueeze(0).to(DEVICE)
                x_448_norm = _imagenet_normalize(x_448)
                hm = _dino_single_pass(model, x_448_norm, guide_224, self.gf_radius)
                if self.tta_flip:
                    xi_f = torch.flip(x_448_norm, dims=[-1])
                    gi_f = torch.flip(guide_224, dims=[-1])
                    hm_f = _dino_single_pass(model, xi_f, gi_f, self.gf_radius)
                    hm_f = torch.flip(hm_f, dims=[-1])
                    hm   = (hm + hm_f) * 0.5
                    lo, hi_v = hm.min(), hm.max()
                    hm   = (hm - lo) / (hi_v - lo + 1e-8)
                heatmaps.append(hm.squeeze().to(images.device))
            return torch.stack(heatmaps)

        B, C, H, W = images.shape
        guide = _denormalize_imagenet(images)
        x_dino_rgb = _resize_batch(guide, self.dino_size)
        x_dino     = x_dino_rgb  # Unnormalized to match stage1_all_methods.py
        gf_r = round(self.gf_radius * H / self.dino_size)

        model = self._get_model()
        heatmaps = []
        for i in range(B):
            xi = x_dino[i:i+1]
            gi = guide[i:i+1]

            hm = _dino_single_pass(model, xi, gi, gf_r)

            if self.tta_flip:
                xi_f = torch.flip(xi, dims=[-1])
                gi_f = torch.flip(gi, dims=[-1])
                hm_f = _dino_single_pass(model, xi_f, gi_f, gf_r)
                hm_f = torch.flip(hm_f, dims=[-1])
                hm   = (hm + hm_f) * 0.5
                lo, hi_v = hm.min(), hm.max()
                hm   = (hm - lo) / (hi_v - lo + 1e-8)

            heatmaps.append(hm.squeeze())

        return torch.stack(heatmaps)


# ---------------------------------------------------------------------------
# U2NetAttribution
# ---------------------------------------------------------------------------

class U2NetAttribution(ModelIndependentMethod):
    """
    U²-Net salient-object detection as an attribution signal.
    """

    def __init__(
        self,
        u2net_size:   int   = _U2NET_SIZE,
        smooth_sigma: float = 1.5,
    ):
        super().__init__(name="u2net")
        self.u2net_size   = u2net_size
        self.smooth_sigma = smooth_sigma

    def _get_model(self) -> nn.Module:
        return _ensure_u2net()

    def compute_independent(self, images: torch.Tensor) -> torch.Tensor:
        paths = getattr(self, "current_paths", None)
        if paths is not None:
            u2net = self._get_model()
            model_device = next(u2net.parameters()).device
            heatmaps = []
            _no_amp = (
                torch.amp.autocast("cuda", enabled=False)
                if model_device.type == "cuda"
                else nullcontext()
            )
            for path in paths:
                img_pil_orig = Image.open(path).convert("RGB")
                img_320 = img_pil_orig.resize((self.u2net_size, self.u2net_size), Image.BILINEAR)
                inp = np.array(img_320, dtype=np.float32) / 255.0
                mean_np = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std_np  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                inp = (inp - mean_np) / std_np
                tensor = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(model_device)
                with torch.no_grad(), _no_amp:
                    pred = u2net(tensor)
                alpha = pred.squeeze().cpu().numpy().astype(np.float64)
                if self.smooth_sigma > 0:
                    alpha = gaussian_filter(alpha, sigma=self.smooth_sigma)
                alpha = _norm01(alpha).astype(np.float32)
                heatmaps.append(torch.from_numpy(alpha).to(images.device))
            return torch.stack(heatmaps)

        B, C, H, W = images.shape
        u2net = self._get_model()
        model_device = next(u2net.parameters()).device

        # Reverse ImageNet normalization to get RGB [0,1]
        mean = torch.tensor(_IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
        rgb01 = (images * std + mean).clamp(0, 1)

        heatmaps = []
        _no_amp = (
            torch.amp.autocast("cuda", enabled=False)
            if model_device.type == "cuda"
            else nullcontext()
        )

        for i in range(B):
            # 1. Convert to uint8 numpy array exactly as the user's snippet inputs expect
            img_np = (rgb01[i].cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

            # 2. Resize and normalize using PIL and numpy as in stage1_all_methods.py
            pil = Image.fromarray(img_np).resize((self.u2net_size, self.u2net_size), Image.BILINEAR)
            inp = np.array(pil, dtype=np.float32) / 255.0
            
            mean_np = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std_np  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            inp = (inp - mean_np) / std_np
            
            tensor = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(model_device)

            # 3. Model inference
            with torch.no_grad(), _no_amp:
                pred = u2net(tensor)

            # 4. Extract alpha, map to float64, and post-process
            alpha = pred.squeeze().cpu().numpy().astype(np.float64)
            if self.smooth_sigma > 0:
                alpha = gaussian_filter(alpha, sigma=self.smooth_sigma)
            alpha = _norm01(alpha).astype(np.float32)

            heatmaps.append(torch.from_numpy(alpha).to(images.device))

        return torch.stack(heatmaps)


# ---------------------------------------------------------------------------
# DinoU2NetAttribution  (50/50 raw-heatmap average)
# ---------------------------------------------------------------------------

class DinoU2NetAttribution(ModelIndependentMethod):
    """
    Hybrid: element-wise 50/50 average of DINO and U²-Net raw heatmaps.
    """

    def __init__(self, blend_size: int | None = None):
        super().__init__(name="u2net+dino")
        self._dino  = DinoAttribution()
        self._u2net = U2NetAttribution()
        self.blend_size = blend_size

    def compute_independent(self, images: torch.Tensor) -> torch.Tensor:
        B, _, H, W = images.shape
        blend_h = blend_w = (H if self.blend_size is None else self.blend_size)

        hm_dino = self._dino.compute_independent(images).unsqueeze(1)
        if hm_dino.shape[-2:] != (blend_h, blend_w):
            hm_dino = F.interpolate(hm_dino, size=(blend_h, blend_w), mode="bilinear", align_corners=False)
        hm_dino = hm_dino.squeeze(1)

        hm_u2 = self._u2net.compute_independent(images).unsqueeze(1)
        if hm_u2.shape[-2:] != (blend_h, blend_w):
            hm_u2 = F.interpolate(hm_u2, size=(blend_h, blend_w), mode="bilinear", align_corners=False)
        hm_u2 = hm_u2.squeeze(1)

        hm_d_np = hm_dino.cpu().numpy().astype(np.float32)
        hm_u_np = hm_u2.cpu().numpy().astype(np.float32)

        blended = []
        for i in range(B):
            combined = _norm01(hm_d_np[i] + hm_u_np[i]).astype(np.float32)
            blended.append(torch.from_numpy(combined).to(images.device))

        return torch.stack(blended)


# ---------------------------------------------------------------------------
# Dino448Attribution  — guided filter at full 448px
# ---------------------------------------------------------------------------

class Dino448Attribution(DinoAttribution):
    """
    Identical to DinoAttribution except the guided filter runs at the full
    DINO inference resolution (448px) instead of the input resolution.
    """

    def __init__(
        self,
        dino_size:  int   = _DINO_SIZE,
        gf_radius:  int   = _GF_RADIUS,
        gf_eps:     float = _GF_EPS,
        tta_flip:   bool  = True,
    ):
        super().__init__(dino_size, gf_radius, gf_eps, tta_flip)
        self.name = "dino448"

    def compute_independent(self, images: torch.Tensor) -> torch.Tensor:
        paths = getattr(self, "current_paths", None)
        if paths is not None:
            model = self._get_model()
            heatmaps = []
            to_tensor = transforms.ToTensor()
            for path in paths:
                img_pil_orig = Image.open(path).convert("RGB")
                guide_448 = to_tensor(img_pil_orig.resize((self.dino_size, self.dino_size), Image.LANCZOS)).unsqueeze(0).to(DEVICE)
                x_448_norm = _imagenet_normalize(guide_448)
                base_out_size = 224
                gf_r = round(self.gf_radius * self.dino_size / base_out_size)
                hm = _dino_single_pass(model, x_448_norm, guide_448, gf_r)
                if self.tta_flip:
                    xi_f = torch.flip(x_448_norm, dims=[-1])
                    gi_f = torch.flip(guide_448, dims=[-1])
                    hm_f = _dino_single_pass(model, xi_f, gi_f, gf_r)
                    hm_f = torch.flip(hm_f, dims=[-1])
                    hm   = (hm + hm_f) * 0.5
                    lo, hi_v = hm.min(), hm.max()
                    hm   = (hm - lo) / (hi_v - lo + 1e-8)
                heatmaps.append(hm.squeeze().to(images.device))
            return torch.stack(heatmaps)

        B = images.shape[0]
        guide = _denormalize_imagenet(images)
        guide_hi   = _resize_batch(guide, self.dino_size)
        x_dino     = guide_hi  # Unnormalized to match stage1_all_methods.py

        base_out_size = 224
        gf_r = round(self.gf_radius * self.dino_size / base_out_size)

        model = self._get_model()
        heatmaps = []
        for i in range(B):
            xi = x_dino[i:i+1]
            gi = guide_hi[i:i+1]

            hm = _dino_single_pass(model, xi, gi, gf_r)

            if self.tta_flip:
                xi_f = torch.flip(xi, dims=[-1])
                gi_f = torch.flip(gi, dims=[-1])
                hm_f = _dino_single_pass(model, xi_f, gi_f, gf_r)
                hm_f = torch.flip(hm_f, dims=[-1])
                hm   = (hm + hm_f) * 0.5
                lo, hi_v = hm.min(), hm.max()
                hm   = (hm - lo) / (hi_v - lo + 1e-8)

            heatmaps.append(hm.squeeze())

        return torch.stack(heatmaps)


# ---------------------------------------------------------------------------
#DinoU2NetSumAttribution  — clamped sum (bounded OR)
# ---------------------------------------------------------------------------
class DinoU2NetSumAttribution(ModelIndependentMethod):
    """
    Hybrid: element-wise sum of DINO and U²-Net heatmaps, clamped to [0, 1].
    """

    def __init__(self, blend_size: int | None = None):
        super().__init__(name="dino+u2net_sum")
        self._dino      = DinoAttribution()
        self._u2net     = U2NetAttribution()
        self.blend_size = blend_size

    def compute_independent(self, images: torch.Tensor) -> torch.Tensor:
        B, _, H, W = images.shape
        blend_h = blend_w = (H if self.blend_size is None else self.blend_size)

        def _to_blend(hm: torch.Tensor) -> torch.Tensor:
            hm = hm.unsqueeze(1)
            if hm.shape[-2:] != (blend_h, blend_w):
                hm = F.interpolate(hm, size=(blend_h, blend_w), mode="bilinear", align_corners=False)
            return hm.squeeze(1)

        hm_dino = _to_blend(self._dino.compute_independent(images))
        hm_u2   = _to_blend(self._u2net.compute_independent(images))

        return torch.clamp(hm_dino + hm_u2, 0.0, 1.0)


# ---------------------------------------------------------------------------
# DinoU2Net320Attribution  — blend on U2Net's native 320px grid
# ---------------------------------------------------------------------------

class DinoU2Net320Attribution(ModelIndependentMethod):
    """
    Hybrid: 50/50 average of DINO and U²-Net heatmaps on U2Net's native 320×320 grid.
    """

    def __init__(self):
        super().__init__(name="u2net+dino_320")
        self._dino  = DinoAttribution()
        self._u2net = U2NetAttribution()

    def compute_independent(self, images: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]

        # Pull DINO maps and align to U2Net native footprint
        hm_dino = self._dino.compute_independent(images).unsqueeze(1)
        hm_dino = F.interpolate(hm_dino, size=(_U2NET_SIZE, _U2NET_SIZE), mode="bilinear", align_corners=False).squeeze(1)

        # Pull U2Net maps (already 320x320)
        hm_u2 = self._u2net.compute_independent(images)

        hm_d_np = hm_dino.cpu().numpy().astype(np.float32)
        hm_u_np = hm_u2.cpu().numpy().astype(np.float32)

        blended = []
        for i in range(B):
            combined = _norm01(hm_d_np[i] + hm_u_np[i]).astype(np.float32)
            blended.append(torch.from_numpy(combined).to(images.device))

        return torch.stack(blended)


# ---------------------------------------------------------------------------
# Dino448U2NetAttribution — DINO-448 downsampled to 320 then averaged with U2Net
# ---------------------------------------------------------------------------

class Dino448U2NetAttribution(ModelIndependentMethod):
    """
    Hybrid: DINO-448 downsampled to 320px then averaged 50/50 with U²-Net.
    """

    def __init__(self):
        super().__init__(name="dino448+u2net")
        self._dino448 = Dino448Attribution()
        self._u2net   = U2NetAttribution()

    def compute_independent(self, images: torch.Tensor) -> torch.Tensor:
        paths = getattr(self, "current_paths", None)
        if paths is not None:
            self._dino448.current_paths = paths
            self._u2net.current_paths = paths
        B = images.shape[0]

        # Pull high-resolution DINO map (448x448) and downsample to 320x320
        hm_dino = self._dino448.compute_independent(images).unsqueeze(1)
        hm_dino = F.interpolate(hm_dino, size=(_U2NET_SIZE, _U2NET_SIZE), mode="bilinear", align_corners=False).squeeze(1)

        # Pull U2Net maps (320x320)
        hm_u2 = self._u2net.compute_independent(images)

        hm_d_np = hm_dino.cpu().numpy().astype(np.float32)
        hm_u_np = hm_u2.cpu().numpy().astype(np.float32)

        blended = []
        for i in range(B):
            combined = _norm01(hm_d_np[i] + hm_u_np[i]).astype(np.float32)
            blended.append(torch.from_numpy(combined).to(images.device))

        return torch.stack(blended)