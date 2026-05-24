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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torchvision import transforms

from attribution.base import ModelIndependentMethod
import config

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


def _ensure_u2net() -> nn.Module:
    """Load and cache U2Net from config.U2NET_WEIGHTS (called once on first use)."""
    if "u2net" not in _MODEL_CACHE:
        weights = config.U2NET_WEIGHTS
        print(f"[AviMethods] Loading U2Net from {weights} on {DEVICE} …")
        m = U2NET(3, 1)
        m.load_state_dict(torch.load(weights, map_location=DEVICE, weights_only=True))
        _MODEL_CACHE["u2net"] = m.to(DEVICE).eval()
    return _MODEL_CACHE["u2net"]


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


# ---------------------------------------------------------------------------
# U2Net architecture  (verbatim from stage1_all_methods.py)
# ---------------------------------------------------------------------------

class _REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super().__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=dirate, dilation=dirate)
        self.bn_s1   = nn.BatchNorm2d(out_ch)
        self.relu    = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.bn_s1(self.conv_s1(x)))

def _up(src, tgt):
    return F.interpolate(src, size=tgt.shape[2:], mode="bilinear", align_corners=False)

class _RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = _REBNCONV(in_ch, out_ch)
        self.rebnconv1  = _REBNCONV(out_ch,    mid_ch); self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.rebnconv2  = _REBNCONV(mid_ch,    mid_ch); self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.rebnconv3  = _REBNCONV(mid_ch,    mid_ch); self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.rebnconv4  = _REBNCONV(mid_ch,    mid_ch); self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.rebnconv5  = _REBNCONV(mid_ch,    mid_ch); self.pool5 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.rebnconv6  = _REBNCONV(mid_ch,    mid_ch)
        self.rebnconv7  = _REBNCONV(mid_ch,    mid_ch, dirate=2)
        self.rebnconv6d = _REBNCONV(mid_ch*2,  mid_ch); self.rebnconv5d = _REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv4d = _REBNCONV(mid_ch*2,  mid_ch); self.rebnconv3d = _REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv2d = _REBNCONV(mid_ch*2,  mid_ch); self.rebnconv1d = _REBNCONV(mid_ch*2, out_ch)
    def forward(self, x):
        xi = self.rebnconvin(x)
        e1=self.rebnconv1(xi);               e2=self.rebnconv2(self.pool1(e1))
        e3=self.rebnconv3(self.pool2(e2));   e4=self.rebnconv4(self.pool3(e3))
        e5=self.rebnconv5(self.pool4(e4));   e6=self.rebnconv6(self.pool5(e5))
        e7=self.rebnconv7(e6)
        d=self.rebnconv6d(torch.cat((_up(e7,e6),e6),1))
        d=self.rebnconv5d(torch.cat((_up(d,e5), e5),1))
        d=self.rebnconv4d(torch.cat((_up(d,e4), e4),1))
        d=self.rebnconv3d(torch.cat((_up(d,e3), e3),1))
        d=self.rebnconv2d(torch.cat((_up(d,e2), e2),1))
        d=self.rebnconv1d(torch.cat((_up(d,e1), e1),1))
        return d + xi

class _RSU6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = _REBNCONV(in_ch, out_ch)
        self.rebnconv1  = _REBNCONV(out_ch,   mid_ch); self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.rebnconv2  = _REBNCONV(mid_ch,   mid_ch); self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.rebnconv3  = _REBNCONV(mid_ch,   mid_ch); self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.rebnconv4  = _REBNCONV(mid_ch,   mid_ch); self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.rebnconv5  = _REBNCONV(mid_ch,   mid_ch)
        self.rebnconv6  = _REBNCONV(mid_ch,   mid_ch, dirate=2)
        self.rebnconv5d = _REBNCONV(mid_ch*2, mid_ch); self.rebnconv4d = _REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv3d = _REBNCONV(mid_ch*2, mid_ch); self.rebnconv2d = _REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv1d = _REBNCONV(mid_ch*2, out_ch)
    def forward(self, x):
        xi=self.rebnconvin(x)
        e1=self.rebnconv1(xi);             e2=self.rebnconv2(self.pool1(e1))
        e3=self.rebnconv3(self.pool2(e2)); e4=self.rebnconv4(self.pool3(e3))
        e5=self.rebnconv5(self.pool4(e4)); e6=self.rebnconv6(e5)
        d=self.rebnconv5d(torch.cat((_up(e6,e5),e5),1))
        d=self.rebnconv4d(torch.cat((_up(d,e4), e4),1))
        d=self.rebnconv3d(torch.cat((_up(d,e3), e3),1))
        d=self.rebnconv2d(torch.cat((_up(d,e2), e2),1))
        d=self.rebnconv1d(torch.cat((_up(d,e1), e1),1))
        return d + xi

class _RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = _REBNCONV(in_ch, out_ch)
        self.rebnconv1  = _REBNCONV(out_ch,   mid_ch); self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.rebnconv2  = _REBNCONV(mid_ch,   mid_ch); self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.rebnconv3  = _REBNCONV(mid_ch,   mid_ch); self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.rebnconv4  = _REBNCONV(mid_ch,   mid_ch)
        self.rebnconv5  = _REBNCONV(mid_ch,   mid_ch, dirate=2)
        self.rebnconv4d = _REBNCONV(mid_ch*2, mid_ch); self.rebnconv3d = _REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv2d = _REBNCONV(mid_ch*2, mid_ch); self.rebnconv1d = _REBNCONV(mid_ch*2, out_ch)
    def forward(self, x):
        xi=self.rebnconvin(x)
        e1=self.rebnconv1(xi);             e2=self.rebnconv2(self.pool1(e1))
        e3=self.rebnconv3(self.pool2(e2)); e4=self.rebnconv4(self.pool3(e3))
        e5=self.rebnconv5(e4)
        d=self.rebnconv4d(torch.cat((_up(e5,e4),e4),1))
        d=self.rebnconv3d(torch.cat((_up(d,e3), e3),1))
        d=self.rebnconv2d(torch.cat((_up(d,e2), e2),1))
        d=self.rebnconv1d(torch.cat((_up(d,e1), e1),1))
        return d + xi

class _RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = _REBNCONV(in_ch, out_ch)
        self.rebnconv1  = _REBNCONV(out_ch,   mid_ch); self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.rebnconv2  = _REBNCONV(mid_ch,   mid_ch); self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.rebnconv3  = _REBNCONV(mid_ch,   mid_ch)
        self.rebnconv4  = _REBNCONV(mid_ch,   mid_ch, dirate=2)
        self.rebnconv3d = _REBNCONV(mid_ch*2, mid_ch); self.rebnconv2d = _REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv1d = _REBNCONV(mid_ch*2, out_ch)
    def forward(self, x):
        xi=self.rebnconvin(x)
        e1=self.rebnconv1(xi);             e2=self.rebnconv2(self.pool1(e1))
        e3=self.rebnconv3(self.pool2(e2)); e4=self.rebnconv4(e3)
        d=self.rebnconv3d(torch.cat((_up(e4,e3),e3),1))
        d=self.rebnconv2d(torch.cat((_up(d,e2), e2),1))
        d=self.rebnconv1d(torch.cat((_up(d,e1), e1),1))
        return d + xi

class _RSU4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = _REBNCONV(in_ch,    out_ch)
        self.rebnconv1  = _REBNCONV(out_ch,   mid_ch, dirate=1)
        self.rebnconv2  = _REBNCONV(mid_ch,   mid_ch, dirate=2)
        self.rebnconv3  = _REBNCONV(mid_ch,   mid_ch, dirate=4)
        self.rebnconv4  = _REBNCONV(mid_ch,   mid_ch, dirate=8)
        self.rebnconv3d = _REBNCONV(mid_ch*2, mid_ch, dirate=4)
        self.rebnconv2d = _REBNCONV(mid_ch*2, mid_ch, dirate=2)
        self.rebnconv1d = _REBNCONV(mid_ch*2, out_ch, dirate=1)
    def forward(self, x):
        xi=self.rebnconvin(x)
        e1=self.rebnconv1(xi); e2=self.rebnconv2(e1)
        e3=self.rebnconv3(e2); e4=self.rebnconv4(e3)
        d=self.rebnconv3d(torch.cat((e4,e3),1))
        d=self.rebnconv2d(torch.cat((d,  e2),1))
        d=self.rebnconv1d(torch.cat((d,  e1),1))
        return d + xi

class U2NET(nn.Module):
    """U²-Net full-size salient-object detection model."""
    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()
        self.stage1 = _RSU7(in_ch,  32,  64);  self.pool12 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.stage2 = _RSU6(64,     32, 128);  self.pool23 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.stage3 = _RSU5(128,    64, 256);  self.pool34 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.stage4 = _RSU4(256,   128, 512);  self.pool45 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.stage5 = _RSU4F(512,  256, 512);  self.pool56 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.stage6 = _RSU4F(512,  256, 512)
        self.stage5d = _RSU4F(1024, 256, 512)
        self.stage4d = _RSU4(1024,  128, 256)
        self.stage3d = _RSU5(512,    64, 128)
        self.stage2d = _RSU6(256,    32,  64)
        self.stage1d = _RSU7(128,    16,  64)
        self.side1   = nn.Conv2d( 64, out_ch, 3, padding=1)
        self.side2   = nn.Conv2d( 64, out_ch, 3, padding=1)
        self.side3   = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4   = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5   = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6   = nn.Conv2d(512, out_ch, 3, padding=1)
        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        h1=self.stage1(x);               h2=self.stage2(self.pool12(h1))
        h3=self.stage3(self.pool23(h2)); h4=self.stage4(self.pool34(h3))
        h5=self.stage5(self.pool45(h4)); h6=self.stage6(self.pool56(h5))
        h5d=self.stage5d(torch.cat((_up(h6,h5), h5),1))
        h4d=self.stage4d(torch.cat((_up(h5d,h4),h4),1))
        h3d=self.stage3d(torch.cat((_up(h4d,h3),h3),1))
        h2d=self.stage2d(torch.cat((_up(h3d,h2),h2),1))
        h1d=self.stage1d(torch.cat((_up(h2d,h1),h1),1))
        d1=self.side1(h1d)
        d2=_up(self.side2(h2d),d1); d3=_up(self.side3(h3d),d1)
        d4=_up(self.side4(h4d),d1); d5=_up(self.side5(h5d),d1)
        d6=_up(self.side6(h6),   d1)
        return torch.sigmoid(self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1)))


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
        B, C, H, W = images.shape
        guide = _denormalize_imagenet(images)
        x_dino_rgb = _resize_batch(guide, self.dino_size)
        x_dino     = _imagenet_normalize(x_dino_rgb)
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
        x_u2 = _resize_batch(images, self.u2net_size)

        with torch.no_grad():
            preds = self._get_model()(x_u2)

        heatmaps = []
        for i in range(preds.shape[0]):
            hm = preds[i, 0].cpu().numpy().astype(np.float64)
            if self.smooth_sigma > 0:
                hm = gaussian_filter(hm, sigma=self.smooth_sigma)
            hm = _norm01(hm).astype(np.float32)
            heatmaps.append(torch.from_numpy(hm))

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
            blended.append(torch.from_numpy(combined))

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
        B = images.shape[0]
        guide = _denormalize_imagenet(images)
        guide_hi   = _resize_batch(guide, self.dino_size)
        x_dino     = _imagenet_normalize(guide_hi)

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
            blended.append(torch.from_numpy(combined))

        return torch.stack(blended)


# ---------------------------------------------------------------------------
# Dino448U2NetAttribution — DINO-448 downsampled to 320 then averaged with U2Net
# ---------------------------------------------------------------------------

class Dino448U2NetAttribution(ModelIndependentMethod):
    """
    Hybrid: DINO-448 downsampled to 320px then averaged 50/50 with U²-Net.
    """

    def __init__(self):
        super().__init__(name="dino448_u2net_avg_320")
        self._dino448 = Dino448Attribution()
        self._u2net   = U2NetAttribution()

    def compute_independent(self, images: torch.Tensor) -> torch.Tensor:
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
            blended.append(torch.from_numpy(combined))

        return torch.stack(blended)