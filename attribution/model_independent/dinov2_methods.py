"""DINOv2-based attribution methods — unified 7-method suite.

All methods use ``facebook/dinov2-with-registers-base`` (ViT-B/14, registers
enabled) loaded via HuggingFace ``transformers``.

Configuration flags (all optional, read from ``config`` module):
  DINO_PCA_N_COMPONENTS    int    3
  DINO_PCA_BORDER          int    1
  DINO_PCA_CENTER_FRAC     float  0.4
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms

try:
    from transformers import AutoModel
except ImportError as exc:
    raise ImportError("Please install transformers: pip install transformers") from exc

import config
from attribution.base import ModelIndependentMethod
from attribution._shared import DEVICE, get_cached_model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "dinov2_vitl14_reg"
MODEL_HUB = "facebookresearch/dinov2"
_PATCH_SIZE           = 14
_DEFAULT_INPUT_SIZE   = int(getattr(config, "DINO_INPUT_SIZE", 224))
_IMAGENET_MEAN        = (0.485, 0.456, 0.406)
_IMAGENET_STD         = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _ensure_model():
    def _load():
        m          = torch.hub.load(MODEL_HUB, MODEL_NAME).to(DEVICE).eval()
        n_expected = getattr(config, "DINO_NUM_REGISTERS", 4)
        # torch.hub DINO models do not expose HuggingFace-style `model.config`.
        # Prefer native attributes when available; otherwise skip strict check.
        n_actual = getattr(
            m,
            "num_register_tokens",
            getattr(m, "num_registers", n_expected),
        )
        if n_actual != n_expected:
            raise ValueError(
                f"DINO register count mismatch: config={n_expected}, model={n_actual}. "
                "Update DINO_NUM_REGISTERS or DINO_MODEL_NAME in config.py."
            )
        return m.to(DEVICE).eval()
    return get_cached_model("dinov2_hf", _load)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _normalize_dino_size(size: int | None) -> int:
    s = int(_DEFAULT_INPUT_SIZE if size is None else size)
    if s <= 0:
        raise ValueError(f"DINO input size must be positive, got {s}.")
    if s % _PATCH_SIZE != 0:
        raise ValueError(f"DINO input size must be divisible by {_PATCH_SIZE}, got {s}.")
    return s


def _preprocess(img_pil: Image.Image, input_size: int) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    return transform(img_pil).unsqueeze(0).to(DEVICE)


def _tensor_to_pil(images: torch.Tensor) -> list[Image.Image]:
    mean     = torch.tensor(_IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std      = torch.tensor(_IMAGENET_STD,  device=images.device).view(1, 3, 1, 1)
    imgs_u8  = ((images * std + mean).clamp(0, 1) * 255).byte().cpu()
    return [Image.fromarray(imgs_u8[i].permute(1, 2, 0).numpy()) for i in range(imgs_u8.shape[0])]


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def _forward_once(model, img_pil: Image.Image, input_size: int) -> dict:
    """Single DINOv2 forward pass; returns all signals needed by every method.

    Token layout: [CLS, reg_1, …, reg_R, patch_1, …, patch_N]
    """
    img_t  = _preprocess(img_pil, input_size)
    grid_h = img_t.shape[2] // _PATCH_SIZE
    grid_w = img_t.shape[3] // _PATCH_SIZE
    N      = grid_h * grid_w

    attn_last = None
    hidden = None

    # Primary path: HuggingFace-style API (supports output_attentions)
    try:
        with torch.no_grad():
            outputs = model(img_t, output_attentions=True)
        if getattr(outputs, "attentions", None) is not None:
            hidden = outputs.last_hidden_state[0].float()
            attn_last = outputs.attentions[-1][0].float()  # (H, T, T)
    except TypeError:
        # torch.hub DINO does not accept output_attentions kwarg.
        outputs = None

    # Fallback path: torch.hub DINO (capture attention via hook + forward_features)
    if attn_last is None or hidden is None:
        holder: dict[str, torch.Tensor] = {}

        def _hook(module, module_input, module_output):
            B, T, C = module_input[0].shape
            qkv = module.qkv(module_input[0])
            qkv = qkv.reshape(B, T, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            attn = (q @ k.transpose(-2, -1)) * ((C // module.num_heads) ** -0.5)
            holder["attn"] = attn.softmax(dim=-1).detach()

        handle = model.blocks[-1].attn.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                out = model.forward_features(img_t)
        finally:
            handle.remove()

        if "attn" not in holder:
            raise RuntimeError("Failed to capture DINO attention via hook.")
        attn_last = holder["attn"][0].float()  # (H, T, T)

        # Prefer normalized patch/CLS tokens exposed by torch.hub model.
        if isinstance(out, dict) and "x_norm_patchtokens" in out:
            patch_tokens = out["x_norm_patchtokens"][0].float()
            cls_raw = out.get("x_norm_clstoken", None)
            if cls_raw is None:
                raise ValueError("forward_features output missing x_norm_clstoken.")
            cls_token = cls_raw[0].float() if cls_raw.ndim == 2 else cls_raw.float()
        elif isinstance(out, dict) and "patch_tokens" in out:
            patch_tokens = out["patch_tokens"][0].float()
            cls_raw = out.get("cls_token", None)
            if cls_raw is None:
                raise ValueError("forward_features output missing cls_token.")
            cls_token = cls_raw[0].float() if cls_raw.ndim == 2 else cls_raw.float()
        else:
            raise ValueError("Cannot extract patch/cls tokens from forward_features().")

        if patch_tokens.shape[0] != N:
            raise ValueError(f"Patch-token count mismatch: got {patch_tokens.shape[0]}, expected {N}")
    else:
        n_prefix = 1 + getattr(config, "DINO_NUM_REGISTERS", 4)
        cls_token = hidden[0]
        patch_tokens = hidden[n_prefix : n_prefix + N]

    attn_raw  = attn_last[:, 0, -N:].mean(dim=0)        # (N,)
    lo, hi    = attn_raw.min(), attn_raw.max()
    attn_cls  = (attn_raw - lo) / (hi - lo + 1e-8)

    W_orig, H_orig = img_pil.size
    return dict(
        patch_tokens=patch_tokens,
        cls_token=cls_token,
        attn_last=attn_last,
        attn_cls=attn_cls,
        grid_h=grid_h,
        grid_w=grid_w,
        dino_input_size=input_size,
        W_orig=W_orig,
        H_orig=H_orig,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    return ((scores - lo) / (hi - lo)).astype(np.float32) if hi - lo > 1e-8 \
           else np.zeros_like(scores, dtype=np.float32)


def _map_entropy(scores_01: np.ndarray) -> float:
    """Normalized Shannon entropy of a [0,1] array; 1 = flat, 0 = spike."""
    N = len(scores_01)
    p = scores_01.astype(np.float64) + 1e-10
    p /= p.sum()
    return float(np.clip(-np.sum(p * np.log(p)) / np.log(N), 0.0, 1.0))


def _fix_polarity_cls(
    scores: np.ndarray,
    patch_tokens: torch.Tensor,
    cls_token: torch.Tensor,
) -> np.ndarray:
    """Flip component sign if it correlates negatively with CLS cosine similarity."""
    pt      = F.normalize(patch_tokens.float(), dim=1)
    cls     = F.normalize(cls_token.float().unsqueeze(0), dim=1)
    cos_sim = (pt @ cls.T).squeeze(1).cpu().numpy().astype(np.float32)

    scores  = scores.astype(np.float32)
    s_n = (scores  - scores.mean())  / (scores.std()  + 1e-8)
    c_n = (cos_sim - cos_sim.mean()) / (cos_sim.std() + 1e-8)
    return -scores if (s_n * c_n).mean() < 0 else scores


def _guided_filter(guide_rgb: torch.Tensor, src: torch.Tensor, r: int, eps: float) -> torch.Tensor:
    """Edge-aware guided filter (single-channel source, RGB guide)."""
    I = 0.299 * guide_rgb[:, 0:1] + 0.587 * guide_rgb[:, 1:2] + 0.114 * guide_rgb[:, 2:3]

    def box(t: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(t, kernel_size=2 * r + 1, stride=1, padding=r)

    mean_I, mean_p = box(I), box(src)
    a = (box(I * src) - mean_I * mean_p) / (box(I * I) - mean_I * mean_I + eps)
    b = mean_p - a * mean_I
    return (box(a) * I + box(b)).clamp(0.0, 1.0)


def _gaussian_smooth(hm: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply separable Gaussian blur to a (1, 1, H, W) tensor."""
    if sigma <= 0:
        return hm
    half   = math.ceil(3.0 * sigma)
    ksize  = 2 * half + 1
    coords = torch.arange(ksize, dtype=torch.float32, device=hm.device) - half
    g      = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel = torch.outer(g / g.sum(), g / g.sum()).view(1, 1, ksize, ksize)
    return F.conv2d(hm, kernel, padding=half)


def _renorm_tensor(t: torch.Tensor) -> torch.Tensor:
    lo, hi = t.min(), t.max()
    return (t - lo) / (hi - lo + 1e-8) if (hi - lo).abs() > 1e-8 else torch.zeros_like(t)


def _to_output_tensor(
    scores: np.ndarray, grid_h: int, grid_w: int, H_out: int, W_out: int,
) -> np.ndarray:
    """Upsample (N,) patch scores to (H_out, W_out) and renormalize."""
    t = torch.from_numpy(scores).reshape(1, 1, grid_h, grid_w).float()
    t = F.interpolate(t, size=(H_out, W_out), mode="bilinear", align_corners=False).squeeze()
    return _renorm_tensor(t).numpy()


# ---------------------------------------------------------------------------
# PCA helpers
# ---------------------------------------------------------------------------

def _run_pca(patch_tokens: torch.Tensor, n_components: int) -> tuple[np.ndarray, PCA]:
    feat = patch_tokens.cpu().numpy().astype(np.float32)
    feat -= feat.mean(axis=0, keepdims=True)
    n    = min(n_components, feat.shape[0], feat.shape[1])
    pca  = PCA(n_components=n, whiten=False)
    return pca.fit_transform(feat), pca


def _pca_soft_base(fwd: dict, n_components: int, border: int, center_frac: float) -> np.ndarray:
    """Best PCA component (center-border selection + CLS polarity), normalized to [0,1]."""
    patch_tokens = fwd["patch_tokens"]
    cls_token    = fwd["cls_token"]
    grid_h, grid_w = fwd["grid_h"], fwd["grid_w"]

    pcs, _ = _run_pca(patch_tokens, n_components)

    # Build masks
    bm = np.zeros((grid_h, grid_w), dtype=bool)
    bm[:border, :] = bm[-border:, :] = bm[:, :border] = bm[:, -border:] = True
    border_flat = bm.flatten()

    cm = np.zeros((grid_h, grid_w), dtype=bool)
    h0 = max(int(grid_h * (0.5 - center_frac / 2)), 0)
    h1 = min(int(grid_h * (0.5 + center_frac / 2)), grid_h)
    w0 = max(int(grid_w * (0.5 - center_frac / 2)), 0)
    w1 = min(int(grid_w * (0.5 + center_frac / 2)), grid_w)
    cm[h0:h1, w0:w1] = True
    center_flat = cm.flatten()

    # Select best component by center-border contrast
    best_score, best_comp = -np.inf, 0
    for i in range(pcs.shape[1]):
        for sign in (+1.0, -1.0):
            cand_n = _normalize_scores(sign * pcs[:, i])
            score  = cand_n[center_flat].mean() - cand_n[border_flat].mean()
            if score > best_score:
                best_score, best_comp = score, i

    raw = _fix_polarity_cls(pcs[:, best_comp].copy(), patch_tokens, cls_token)
    return _normalize_scores(raw)

# ---------------------------------------------------------------------------
# Score extractors
# ---------------------------------------------------------------------------

def _scores_attn(fwd: dict) -> np.ndarray:
    return fwd["attn_cls"].cpu().numpy().astype(np.float32)


def _scores_pc_eigenweighted(fwd: dict, n_components: int) -> np.ndarray:
    pcs, pca = _run_pca(fwd["patch_tokens"], n_components)
    evr      = pca.explained_variance_ratio_
    weights  = (evr / evr.sum()).astype(np.float32) if evr.sum() > 1e-12 \
               else np.full(pcs.shape[1], 1.0 / pcs.shape[1], dtype=np.float32)

    combined = sum(
        weights[k] * _normalize_scores(_fix_polarity_cls(pcs[:, k].copy(), fwd["patch_tokens"], fwd["cls_token"]))
        for k in range(pcs.shape[1])
    )
    return _normalize_scores(combined.astype(np.float32))


def _scores_pc_l2(fwd: dict, n_components: int) -> np.ndarray:
    pcs, _ = _run_pca(fwd["patch_tokens"], n_components)
    return _normalize_scores(np.linalg.norm(pcs, ord=2, axis=1).astype(np.float32))


def _scores_dino_trisignal(fwd: dict) -> np.ndarray:
    patch_tokens = fwd["patch_tokens"]
    attn_last    = fwd["attn_last"]
    N            = fwd["grid_h"] * fwd["grid_w"]

    s_norm = _normalize_scores(patch_tokens.norm(dim=-1).cpu().numpy().astype(np.float32))

    cls_rows = attn_last[:, 0, -N:]  # (H, N)
    head_w   = np.array([
        max(0.0, 1.0 - _map_entropy(_normalize_scores(cls_rows[h].cpu().numpy().astype(np.float32))))
        for h in range(cls_rows.shape[0])
    ], dtype=np.float32)
    if head_w.sum() < 1e-8:
        head_w = np.full_like(head_w, 1.0 / len(head_w))
    else:
        head_w /= head_w.sum()
    s_attn = _normalize_scores(
        (cls_rows * torch.from_numpy(head_w).to(attn_last.device, dtype=attn_last.dtype)[:, None])
        .sum(dim=0).cpu().numpy().astype(np.float32)
    )

    s_pop = _normalize_scores(attn_last[:, -N:, -N:].sum(dim=1).mean(dim=0).cpu().numpy().astype(np.float32))

    weights = np.array([1.0 - _map_entropy(s) for s in (s_attn, s_norm, s_pop)], dtype=np.float32)
    if weights.sum() < 1e-8:
        weights = np.full(3, 1.0 / 3.0)
    else:
        weights /= weights.sum()

    return _normalize_scores(weights[0] * s_attn + weights[1] * s_norm + weights[2] * s_pop)


def _scores_combo_fixed(attn: np.ndarray, pc1: np.ndarray) -> np.ndarray:
    return (0.5 * attn + 0.5 * pc1).astype(np.float32)


def _scores_combo_entropy(attn: np.ndarray, pc1: np.ndarray) -> np.ndarray:
    w_attn = 1.0 - _map_entropy(attn)
    w_pc1  = 1.0 - _map_entropy(pc1)
    denom  = w_attn + w_pc1
    if denom < 1e-8:
        w_attn, w_pc1 = 0.5, 0.5
    else:
        w_attn, w_pc1 = w_attn / denom, w_pc1 / denom
    return (w_attn * attn + w_pc1 * pc1).astype(np.float32)


def _apply_smooth_gate(
    base: np.ndarray,
    pc1: np.ndarray,
    grid_h: int,
    grid_w: int,
    sigma: float,
    gamma: float,
    alpha: float,
) -> np.ndarray:
    """Shared post-processing: Gaussian smooth → PC1 gate → alpha blend."""
    hm     = torch.from_numpy(base).float().reshape(1, 1, grid_h, grid_w).to(DEVICE)
    hm     = _renorm_tensor(_gaussian_smooth(hm, sigma))
    pc_map = torch.from_numpy(pc1).float().reshape(1, 1, grid_h, grid_w).to(DEVICE)
    gated  = hm * (pc_map ** gamma)
    hm     = alpha * gated + (1.0 - alpha) * hm
    return _normalize_scores(hm.reshape(-1).cpu().numpy().astype(np.float32))


# ===========================================================================
# Base class
# ===========================================================================

class Dinov2AllMethodsBase(ModelIndependentMethod):
    """Shared base for all DINOv2 attribution methods."""

    def __init__(self, method_name: str, dino_input_size: int | None = None) -> None:
        super().__init__(method_name)
        self.n_components    = getattr(config, "DINO_PCA_N_COMPONENTS", 1)
        self.border          = getattr(config, "DINO_PCA_BORDER",       1)
        self.center_frac     = getattr(config, "DINO_PCA_CENTER_FRAC",  0.4)
        self.dino_input_size = _normalize_dino_size(dino_input_size)
        self._model          = None

    def _get_model(self):
        if self._model is None:
            self._model = _ensure_model()
        return self._model

    def _shared_forward(self, img_pil: Image.Image) -> tuple[dict, np.ndarray]:
        fwd     = _forward_once(self._get_model(), img_pil, self.dino_input_size)
        pc_base = _pca_soft_base(fwd, self.n_components, self.border, self.center_frac)
        return fwd, pc_base

    def _scores_from_fwd(self, fwd: dict, pc_base: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _heatmap_single(self, img_pil: Image.Image) -> np.ndarray:
        fwd, pc_base = self._shared_forward(img_pil)
        scores       = self._scores_from_fwd(fwd, pc_base)
        return _to_output_tensor(scores, fwd["grid_h"], fwd["grid_w"], fwd["H_orig"], fwd["W_orig"])

    def compute_independent(self, images: torch.Tensor) -> torch.Tensor:
        B, C, H, W = images.shape
        heatmaps   = np.stack([self._heatmap_single(p) for p in _tensor_to_pil(images)], axis=0)
        if heatmaps.shape[1] != H or heatmaps.shape[2] != W:
            ht       = torch.from_numpy(heatmaps).unsqueeze(1).float()
            heatmaps = F.interpolate(ht, size=(H, W), mode="bilinear", align_corners=False).squeeze(1).numpy()
        return torch.from_numpy(heatmaps).float()


# ===========================================================================
# Attribution methods
# ===========================================================================

class Dinov2AttnMethod(Dinov2AllMethodsBase):
    """Method 1 — DINO_ATTN: head-averaged CLS-to-patch self-attention."""

    def __init__(self, dino_input_size: int | None = None) -> None:
        super().__init__("dinov2_attn", dino_input_size=dino_input_size)

    def _scores_from_fwd(self, fwd: dict, pc_base: np.ndarray) -> np.ndarray:
        return _scores_attn(fwd)


class Dinov2Pc1Method(Dinov2AllMethodsBase):
    """Method 2 — DINO_PC1: best soft PCA component."""

    def __init__(self, dino_input_size: int | None = None) -> None:
        super().__init__("dinov2_pc1", dino_input_size=dino_input_size)

    def _scores_from_fwd(self, fwd: dict, pc_base: np.ndarray) -> np.ndarray:
        return pc_base


class Dinov2PcEigenweightedMethod(Dinov2AllMethodsBase):
    """Method 3 — DINO_PC_EV: explained-variance-weighted PC1+PC2+PC3."""

    def __init__(self, dino_input_size: int | None = None) -> None:
        super().__init__("dinov2_pc_ev", dino_input_size=dino_input_size)

    def _scores_from_fwd(self, fwd: dict, pc_base: np.ndarray) -> np.ndarray:
        return _scores_pc_eigenweighted(fwd, self.n_components)


class Dinov2PcL2Method(Dinov2AllMethodsBase):
    """Method 4 — DINO_PC_L2: L2 norm of patch projections onto PC1+PC2+PC3."""

    def __init__(self, dino_input_size: int | None = None) -> None:
        super().__init__("dinov2_pc_l2", dino_input_size=dino_input_size)

    def _scores_from_fwd(self, fwd: dict, pc_base: np.ndarray) -> np.ndarray:
        return _scores_pc_l2(fwd, self.n_components)


class Dinov2ComboFixedMethod(Dinov2AllMethodsBase):
    """Method 5 — COMBO_FIXED: equal-weight blend of attention and PC1."""

    def __init__(self, dino_input_size: int | None = None) -> None:
        super().__init__("dinov2_combo_fixed", dino_input_size=dino_input_size)

    def _scores_from_fwd(self, fwd: dict, pc_base: np.ndarray) -> np.ndarray:
        return _scores_combo_fixed(_scores_attn(fwd), pc_base)


class Dinov2ComboEntropyMethod(Dinov2AllMethodsBase):
    """Method 6 — COMBO_ENT: entropy-adaptive blend of attention and PC1."""

    def __init__(self, dino_input_size: int | None = None) -> None:
        super().__init__("dinov2_combo_ent", dino_input_size=dino_input_size)

    def _scores_from_fwd(self, fwd: dict, pc_base: np.ndarray) -> np.ndarray:
        return _scores_combo_entropy(_scores_attn(fwd), pc_base)


class Dinov2ComboEntSmoothMethod(Dinov2AllMethodsBase):
    """Method 7 — COMBO_ENT_SMOOTH: entropy-weighted attn+PC1 with Gaussian smoothing and PC1 gate.

    Args:
        sigma:  Gaussian std-dev in patch units (0 = disabled).
        gamma:  PC1 gate exponent.
        alpha:  Blend weight for the gated map (1.0 = fully gated).
    """

    def __init__(
        self,
        sigma: float = 2.0,
        gamma: float = 0.7,
        alpha: float = 0.45,
        dino_input_size: int | None = None,
    ) -> None:
        super().__init__("dinov2_combo_ent_smooth", dino_input_size=dino_input_size)
        self.sigma = sigma
        self.gamma = gamma
        self.alpha = alpha

    def _scores_from_fwd(self, fwd: dict, pc_base: np.ndarray) -> np.ndarray:
        base = _scores_combo_entropy(_scores_attn(fwd), pc_base)
        return _apply_smooth_gate(base, pc_base, fwd["grid_h"], fwd["grid_w"], self.sigma, self.gamma, self.alpha)


class Dinov2TriSignalGuidedMethod(Dinov2AllMethodsBase):
    """DINO tri-signal (attn + norm + popularity) with guided filter and flip TTA."""

    def __init__(
        self,
        dino_input_size: int | None = None,
        output_size: int | None = None,
        hi_res_guided_filter: bool = False,
        guided_filter_radius: int = 8,
        guided_filter_eps: float = 1e-2,
        use_flip_tta: bool = True,
    ) -> None:
        super().__init__("dinov2_trisignal_guided", dino_input_size=dino_input_size)
        self.output_size = _normalize_dino_size(output_size) if output_size is not None else 224
        self.hi_res_guided_filter = bool(hi_res_guided_filter)
        self.guided_filter_radius = max(0, int(guided_filter_radius))
        self.guided_filter_eps    = float(guided_filter_eps)
        self.use_flip_tta         = bool(use_flip_tta)

    def _extract_signals(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exact stage1 extraction: hook qkv-attention + forward_features patches."""
        dino_model = self._get_model()
        H, W = x.shape[-2], x.shape[-1]
        N = (H // _PATCH_SIZE) * (W // _PATCH_SIZE)
        holder = {}

        def _hook(module, module_input, module_output):
            B, T, C = module_input[0].shape
            qkv = module.qkv(module_input[0])
            qkv = qkv.reshape(B, T, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            attn = (q @ k.transpose(-2, -1)) * ((C // module.num_heads) ** -0.5)
            holder["attn"] = attn.softmax(dim=-1).detach()

        handle = dino_model.blocks[-1].attn.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                out = dino_model.forward_features(x)
        finally:
            handle.remove()

        if isinstance(out, dict) and "x_norm_patchtokens" in out:
            patches = out["x_norm_patchtokens"][0].float()
        elif isinstance(out, dict) and "patch_tokens" in out:
            patches = out["patch_tokens"][0].float()
        else:
            raise ValueError("Cannot extract patch tokens from forward_features().")

        norms = patches.norm(dim=-1).cpu().numpy().astype(np.float32)
        s_norm = _normalize_scores(norms)

        attn = holder["attn"][0]
        T = attn.shape[-1]
        patch_start = T - N
        cls_rows = attn[:, 0, patch_start:]

        head_w = []
        for h in range(cls_rows.shape[0]):
            row = cls_rows[h].cpu().numpy().astype(np.float32)
            row_n = _normalize_scores(row)
            head_w.append(max(0.0, 1.0 - _map_entropy(row_n)))
        head_w = np.array(head_w, dtype=np.float32)
        denom = head_w.sum()
        head_w = head_w / denom if denom > 1e-8 else np.ones_like(head_w) / len(head_w)

        hw_t = torch.from_numpy(head_w).to(attn.device)
        scores = (cls_rows * hw_t[:, None]).sum(0).cpu().numpy().astype(np.float32)
        s_attn = _normalize_scores(scores)

        p2p = attn[:, patch_start:, patch_start:]
        pop = p2p.sum(dim=1).mean(dim=0).cpu().numpy().astype(np.float32)
        s_pop = _normalize_scores(pop)
        return s_attn, s_norm, s_pop

    def _heatmap_core(self, x: torch.Tensor, guide: torch.Tensor, out_size: int, gf_radius: int) -> torch.Tensor:
        """Exact stage1 core: entropy-weighted tri-signal + guided filter."""
        N = (self.dino_input_size // _PATCH_SIZE) ** 2
        s_attn, s_norm, s_pop = self._extract_signals(x)

        w_attn = max(0.0, 1.0 - _map_entropy(s_attn))
        w_norm = max(0.0, 1.0 - _map_entropy(s_norm))
        w_pop = max(0.0, 1.0 - _map_entropy(s_pop))
        denom = w_attn + w_norm + w_pop
        wa, wn, wp = ((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0) if denom < 1e-8
                      else (w_attn / denom, w_norm / denom, w_pop / denom))

        scores = _normalize_scores(wa * s_attn + wn * s_norm + wp * s_pop)
        gh = self.dino_input_size // _PATCH_SIZE
        hm = torch.from_numpy(scores).float().reshape(1, 1, gh, gh).to(x.device)
        hm = F.interpolate(hm, size=(out_size, out_size), mode="bilinear", align_corners=False)
        if gf_radius > 0:
            hm = _guided_filter(guide, hm, gf_radius, self.guided_filter_eps)
        return _renorm_tensor(hm)

    def _single_pass(self, img_pil: Image.Image) -> np.ndarray:
        to_tensor = transforms.ToTensor()
        x = to_tensor(img_pil.resize((self.dino_input_size, self.dino_input_size), Image.LANCZOS)).unsqueeze(0).to(DEVICE)

        if self.hi_res_guided_filter:
            out_size = self.dino_input_size
            guide = x
            gf_radius = round(self.guided_filter_radius * out_size / self.output_size) if self.output_size > 0 else self.guided_filter_radius
        else:
            out_size = self.output_size
            guide = to_tensor(img_pil.resize((out_size, out_size), Image.LANCZOS)).unsqueeze(0).to(DEVICE)
            gf_radius = self.guided_filter_radius

        with torch.no_grad():
            hm = self._heatmap_core(x, guide, out_size, gf_radius)
            if self.use_flip_tta:
                hm_flip = self._heatmap_core(
                    torch.flip(x, dims=[-1]),
                    torch.flip(guide, dims=[-1]),
                    out_size,
                    gf_radius,
                )
                hm_flip = torch.flip(hm_flip, dims=[-1])
                hm = 0.5 * hm + 0.5 * hm_flip
                hm = _renorm_tensor(hm)

        return hm.squeeze().cpu().numpy().astype(np.float32)

    def _heatmap_single(self, img_pil: Image.Image) -> np.ndarray:
        hm = self._single_pass(img_pil)
        # Framework expects final map at original image resolution.
        if hm.shape[0] == img_pil.size[1] and hm.shape[1] == img_pil.size[0]:
            return hm
        hm_t = torch.from_numpy(hm).unsqueeze(0).unsqueeze(0).float()
        hm_t = F.interpolate(hm_t, size=(img_pil.size[1], img_pil.size[0]), mode="bilinear", align_corners=False)
        return _renorm_tensor(hm_t.squeeze()).cpu().numpy().astype(np.float32)