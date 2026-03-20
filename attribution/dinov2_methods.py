"""
DINOv2-based attribution methods — unified 6-method suite.

Architecture
------------
All six methods share a single forward pass through DINOv2.  The shared
base is the **soft PCA map**: the best-scoring principal component (among
the first ``DINO_PCA_N_COMPONENTS``, default 3) selected by center-vs-border
contrast scoring, normalized to [0, 1] with NO binarization and NO Gaussian
blur.  This raw continuous map is then used as the PC1 signal by every
method that needs it, avoiding redundant model calls.

    ┌─────────────────────────────────────────────────────────────────────┐
    │ Step 0  One DINOv2 forward pass → patch_tokens [N,D], cls [D],     │
    │          attention [heads, N]                                        │
    │ Step 1  PCA(n_components) → select best component by center-border  │
    │          scoring → soft [0,1] map  (shared base for all methods)    │
    │                                                                      │
    │  Method              PC base?  Attn?  Extra blend logic             │
    │  ──────────────────  ────────  ─────  ─────────────────────────     │
    │  1. DINO_ATTN        –         ✓      head-averaged CLS attention    │
    │  2. DINO_PC1         ✓         –      soft PC1 only                  │
    │  3. DINO_PC_EV       ✓         –      PC1+PC2+PC3 × explained var    │
    │  4. DINO_PC_L2       ✓         –      L2 norm of PC1+PC2+PC3         │
    │  5. COMBO_FIXED      ✓         ✓      0.5 × attn + 0.5 × PC1         │
    │  6. COMBO_ENT        ✓         ✓      entropy-adaptive attn + PC1    │
    └─────────────────────────────────────────────────────────────────────┘

Polarity correction
-------------------
For all PCA-based methods, the sign of each component is determined by
Pearson correlation with per-patch cosine similarity to the CLS token.
The CLS token is DINOv2's global image summary; patches most similar to it
are empirically the foreground regions.  A negative correlation means the
component is pointing "away" from foreground — it is flipped.  This works
for any foreground-to-background area ratio (close-ups, medical images, etc.)
and replaces the old center-vs-border polarity heuristic used for sign only
(center-vs-border is still used for *component selection*, not sign).

Why no binarization?
--------------------
The normalized PCA component values already form a continuous [0, 1]
importance map.  Applying a hard quantile threshold would collapse the
natural gradations between "strongly foreground" and "weakly foreground"
patches.  The soft map is passed directly into each method's blending logic.

Configuration flags (all optional, read from ``config`` module)
---------------------------------------------------------------
  DINO_PCA_USE_REGISTERS   bool   True   – use dinov2_vits14_reg
  DINO_PCA_N_COMPONENTS    int    3      – components to evaluate for selection
  DINO_PCA_BORDER          int    1      – border ring width for scoring
  DINO_PCA_CENTER_FRAC     float  0.4    – center fraction for scoring
  DEVICE                   str           – torch device (auto-detected if absent)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms

import config
from attribution.base import AttributionMethod


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEVICE = torch.device(
    config.DEVICE
    if hasattr(config, "DEVICE")
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

_PATCH_SIZE = 14
_ATTN_IMAGE_SIZE = (518, 518)   # 518 = 14 × 37 — optimal for DINOv2 patch grid
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

# One cache per model variant so both can coexist in the same process
_MODEL_CACHE: dict[str, object] = {}   # "std" | "reg" → torch.hub model


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _ensure_model(use_registers: bool):
    """Load and cache the torch.hub DINOv2 model (called once on first use).

    Uses the torch.hub API (facebookresearch/dinov2) which exposes both
    ``forward_features`` and ``get_last_selfattention`` — required by all
    six methods.

    Args:
        use_registers: If True load ``dinov2_vits14_reg`` (4 register tokens),
                       otherwise ``dinov2_vits14``.

    Returns:
        The cached, eval-mode DINOv2 model on DEVICE.
    """
    key = "reg" if use_registers else "std"
    if key not in _MODEL_CACHE:
        name = "dinov2_vits14_reg" if use_registers else "dinov2_vits14"
        print(f"[Dinov2Methods] Loading {name} on {DEVICE} …")
        m = torch.hub.load("facebookresearch/dinov2", name, verbose=False)
        m.to(DEVICE).eval()
        _MODEL_CACHE[key] = m
    return _MODEL_CACHE[key]


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------

_transform = transforms.Compose([
    transforms.Resize(_ATTN_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def _tensor_to_pil(images: torch.Tensor) -> list[Image.Image]:
    """Un-normalize an ImageNet-normalized float32 batch tensor to PIL images.

    Args:
        images: (B, C, H, W) float32 tensor, ImageNet-normalized.

    Returns:
        List of B RGB PIL Images.
    """
    mean       = torch.tensor(_IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std        = torch.tensor(_IMAGENET_STD,  device=images.device).view(1, 3, 1, 1)
    imgs_rgb   = (images * std + mean).clamp(0, 1)
    imgs_uint8 = (imgs_rgb * 255).byte().cpu()
    return [
        Image.fromarray(imgs_uint8[i].permute(1, 2, 0).numpy())
        for i in range(imgs_uint8.shape[0])
    ]


# ---------------------------------------------------------------------------
# Shared forward pass — extracts everything needed by all 6 methods at once
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shared forward pass — extracts everything needed by all 6 methods at once
# ---------------------------------------------------------------------------

def _forward_once(model, img_pil: Image.Image) -> dict:
    """Run one DINOv2 forward pass and return all signals needed by all methods.

    Uses a forward hook on the last block's QKV projection to extract the
    self-attention matrix, bypassing the lack of `get_last_selfattention` in DINOv2.
    """
    img_t = _transform(img_pil).unsqueeze(0).to(DEVICE)  # (1, C, 518, 518)
    grid_h = img_t.shape[2] // _PATCH_SIZE
    grid_w = img_t.shape[3] // _PATCH_SIZE
    N = grid_h * grid_w

    # 1. Register a hook to intercept the QKV output of the very last block
    saved_qkv = []

    def qkv_hook(module, input, output):
        saved_qkv.append(output.detach())

    last_block = model.blocks[-1]
    handle = last_block.attn.qkv.register_forward_hook(qkv_hook)

    with torch.no_grad():
        # --- patch tokens + CLS -------------------------------------------
        out = model.forward_features(img_t)

    # Remove the hook immediately after the forward pass so it doesn't linger
    handle.remove()

    # 2. Parse the token outputs
    if isinstance(out, dict) and "x_norm_patchtokens" in out:
        patch_tokens = out["x_norm_patchtokens"][0].float()  # (N, D)
        cls_token = out["x_norm_clstoken"][0].float().squeeze(0)  # (D,)
    else:
        # Fallback: raw tensor [1, T, D] — skip non-patch prefix tokens
        all_tok = (out[0] if torch.is_tensor(out) else out["x_prenorm"][0]).float()
        n_prefix = all_tok.shape[0] - N
        patch_tokens = all_tok[n_prefix:]
        cls_token = all_tok[0]

    # 3. Manually compute the attention matrix from the intercepted QKV
    qkv = saved_qkv[0]  # Shape: (1, T, 3 * D)
    B, T, three_D = qkv.shape
    D = three_D // 3
    num_heads = last_block.attn.num_heads

    # Reshape and split into Q, K, V: (1, num_heads, T, head_dim)
    qkv = qkv.reshape(B, T, 3, num_heads, D // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    # Compute attention scores: Softmax((Q * scale) @ K^T)
    scale = (D // num_heads) ** -0.5
    q = q * scale
    attn_matrix = q @ k.transpose(-2, -1)  # (1, num_heads, T, T)
    attn_matrix = attn_matrix.softmax(dim=-1)

    # Extract the CLS-to-patch attention
    # CLS token is at index 0. Patch tokens are the last N tokens.
    attn_raw = attn_matrix[0, :, 0, -N:].mean(dim=0)  # Average across heads -> (N,)

    # Normalize to [0, 1]
    lo, hi = attn_raw.min(), attn_raw.max()
    attn_cls = (attn_raw - lo) / (hi - lo + 1e-8)

    W_orig, H_orig = img_pil.size
    return dict(
        patch_tokens=patch_tokens,
        cls_token=cls_token,
        attn_cls=attn_cls,
        grid_h=grid_h,
        grid_w=grid_w,
        W_orig=W_orig,
        H_orig=H_orig,
    )

# ---------------------------------------------------------------------------
# Polarity correction
# ---------------------------------------------------------------------------

def _fix_polarity_cls(
    scores: np.ndarray,
    patch_tokens: torch.Tensor,
    cls_token: torch.Tensor,
) -> np.ndarray:
    """Flip a PCA component if it correlates negatively with CLS similarity.

    DINOv2's CLS token is the global image summary.  Patches whose features
    are most cosine-similar to the CLS token are empirically the foreground
    regions.  We compute the Pearson correlation between the raw PC scores and
    per-patch cosine similarity to CLS via z-score normalization (numerically
    identical to np.corrcoef but stays float32 and handles constant inputs
    safely via an eps guard).  If the correlation is negative, the component
    points "away from foreground" and is flipped.

    This approach works for any foreground/background area ratio — it makes no
    assumption about the object being centered or small.

    Args:
        scores:       (N,) float32 raw PC projection values (any sign).
        patch_tokens: (N, D) float32 patch feature tensor on DEVICE.
        cls_token:    (D,)   float32 CLS feature tensor on DEVICE.

    Returns:
        (N,) float32 sign-corrected scores.
    """
    pt      = F.normalize(patch_tokens.float(), dim=1)
    cls     = F.normalize(cls_token.float().unsqueeze(0), dim=1)
    cos_sim = (pt @ cls.T).squeeze(1).cpu().numpy().astype(np.float32)

    scores  = scores.astype(np.float32)
    s_n = (scores  - scores.mean())  / (scores.std()  + 1e-8)
    c_n = (cos_sim - cos_sim.mean()) / (cos_sim.std() + 1e-8)
    if (s_n * c_n).mean() < 0:
        scores = -scores
    return scores


# ---------------------------------------------------------------------------
# Shared PCA base — the "pca3 best component" soft map
# ---------------------------------------------------------------------------

def _pca_soft_base(
    fwd: dict,
    n_components: int,
    border: int,
    center_frac: float,
) -> np.ndarray:
    """Compute the soft PCA base map shared by all PCA-based methods.

    Runs PCA with ``n_components`` on the patch features, then selects the
    single best component by center-vs-border contrast scoring.  The winning
    component is polarity-corrected via CLS cosine similarity, then normalized
    to [0, 1].  No binarization, no Gaussian blur.

    Component selection heuristic
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For each component × polarity pair, the score is:

        center_mean(candidate) − border_mean(candidate)

    where "center" is the central ``center_frac`` fraction of the patch grid
    and "border" is the outer ``border``-patch ring.  The pair with the highest
    score wins.  Polarity is handled *after* selection by CLS cosine similarity
    (``_fix_polarity_cls``) — the selection score is used only to pick *which*
    component to use, not to determine its final orientation.

    Args:
        fwd:          Output dict from ``_forward_once``.
        n_components: How many PCA components to evaluate (e.g. 3).
        border:       Border ring width in patches for scoring.
        center_frac:  Central fraction of the grid for scoring.

    Returns:
        (N,) float32 soft map in [0, 1].
    """
    patch_tokens = fwd["patch_tokens"]
    cls_token    = fwd["cls_token"]
    grid_h       = fwd["grid_h"]
    grid_w       = fwd["grid_w"]
    N            = grid_h * grid_w

    feat = patch_tokens.cpu().numpy().astype(np.float32)
    feat -= feat.mean(axis=0, keepdims=True)

    n_comp = min(n_components, feat.shape[0], feat.shape[1])
    pca    = PCA(n_components=n_comp, whiten=False)
    pcs    = pca.fit_transform(feat)               # (N, n_comp)

    # --- Build center and border masks ------------------------------------
    bm = np.zeros((grid_h, grid_w), dtype=bool)
    bm[:border, :]  = True
    bm[-border:, :] = True
    bm[:, :border]  = True
    bm[:, -border:] = True
    border_flat = bm.flatten()

    cm = np.zeros((grid_h, grid_w), dtype=bool)
    h0 = max(int(grid_h * (0.5 - center_frac / 2)), 0)
    h1 = min(int(grid_h * (0.5 + center_frac / 2)), grid_h)
    w0 = max(int(grid_w * (0.5 - center_frac / 2)), 0)
    w1 = min(int(grid_w * (0.5 + center_frac / 2)), grid_w)
    cm[h0:h1, w0:w1] = True
    center_flat = cm.flatten()

    # --- Select best component by center-border contrast ------------------
    best_score = -np.inf
    best_comp  = 0
    for i in range(n_comp):
        for sign in (+1.0, -1.0):
            cand   = sign * pcs[:, i]
            lo, hi = cand.min(), cand.max()
            cand_n = (cand - lo) / (hi - lo + 1e-8)
            score  = cand_n[center_flat].mean() - cand_n[border_flat].mean()
            if score > best_score:
                best_score = score
                best_comp  = i

    # --- Polarity correction via CLS cosine similarity -------------------
    raw = _fix_polarity_cls(pcs[:, best_comp].copy(), patch_tokens, cls_token)

    # --- Normalize to [0, 1] — no threshold, no blur ---------------------
    lo, hi = raw.min(), raw.max()
    if hi - lo > 1e-8:
        return ((raw - lo) / (hi - lo)).astype(np.float32)
    return np.zeros(N, dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-method score extractors (all operate on pre-computed fwd dict)
# ---------------------------------------------------------------------------

def _scores_attn(fwd: dict) -> np.ndarray:
    """Return (N,) float32 head-averaged CLS attention scores, already [0,1]."""
    return fwd["attn_cls"].cpu().numpy().astype(np.float32)


def _scores_pc1(fwd: dict, pc_base: np.ndarray) -> np.ndarray:
    """Return (N,) float32 soft PC1 base map — directly the shared base."""
    return pc_base


def _scores_pc_eigenweighted(fwd: dict, n_components: int) -> np.ndarray:
    """Return (N,) float32 explained-variance-weighted sum of PC1+PC2+PC3.

    Each component is independently polarity-corrected and normalized to [0,1]
    before being weighted by its explained variance ratio (re-normalized to
    sum to 1 across the selected components).  This ensures the blend weights
    reflect genuine variance contribution rather than raw projection scale.

    Args:
        fwd:          Output dict from ``_forward_once``.
        n_components: Number of PCA components to use (capped at 3).

    Returns:
        (N,) float32 combined map in [0, 1].
    """
    patch_tokens = fwd["patch_tokens"]
    cls_token    = fwd["cls_token"]

    feat = patch_tokens.cpu().numpy().astype(np.float32)
    feat -= feat.mean(axis=0, keepdims=True)

    n_comp = min(n_components, feat.shape[0], feat.shape[1])
    pca    = PCA(n_components=n_comp, whiten=False)
    pcs    = pca.fit_transform(feat)

    evr     = pca.explained_variance_ratio_
    evr_sum = evr.sum()
    weights = (evr / evr_sum).astype(np.float32) if evr_sum > 1e-12 \
              else np.full(n_comp, 1.0 / n_comp, dtype=np.float32)

    combined = np.zeros(pcs.shape[0], dtype=np.float32)
    for k in range(n_comp):
        comp = _fix_polarity_cls(pcs[:, k].copy(), patch_tokens, cls_token)
        lo, hi = comp.min(), comp.max()
        comp = ((comp - lo) / (hi - lo)).astype(np.float32) if hi - lo > 1e-8 \
               else np.zeros_like(comp)
        combined += weights[k] * comp

    lo, hi = combined.min(), combined.max()
    return ((combined - lo) / (hi - lo)).astype(np.float32) if hi - lo > 1e-8 \
           else np.zeros_like(combined)


def _scores_pc_l2(fwd: dict, n_components: int) -> np.ndarray:
    """Return (N,) float32 L2 norm of each patch's projection onto PC1+PC2+PC3.

    Squaring each projection before summing eliminates sign ambiguity without
    needing polarity correction.  The L2 norm measures total distance from the
    patch-feature mean in the 3-PC subspace, capturing any patch that is
    semantically distinctive from the average — typically the subject.

    Args:
        fwd:          Output dict from ``_forward_once``.
        n_components: Number of PCA components to use (capped at 3).

    Returns:
        (N,) float32 map in [0, 1].
    """
    patch_tokens = fwd["patch_tokens"]

    feat = patch_tokens.cpu().numpy().astype(np.float32)
    feat -= feat.mean(axis=0, keepdims=True)

    n_comp = min(n_components, feat.shape[0], feat.shape[1])
    pca    = PCA(n_components=n_comp, whiten=False)
    pcs    = pca.fit_transform(feat)

    l2 = np.linalg.norm(pcs, ord=2, axis=1).astype(np.float32)
    lo, hi = l2.min(), l2.max()
    return ((l2 - lo) / (hi - lo)).astype(np.float32) if hi - lo > 1e-8 \
           else np.zeros_like(l2)


def _map_entropy(scores_01: np.ndarray) -> float:
    """Compute normalized Shannon entropy of a [0,1] score array.

    Converts scores to a valid probability distribution (add eps, renormalize),
    computes entropy in nats, then divides by log(N) to normalize to [0, 1].

    Returns:
        float in [0, 1]: 1 = completely flat (uninformative), 0 = perfect spike.
    """
    N = len(scores_01)
    p = scores_01.astype(np.float64) + 1e-10
    p = p / p.sum()
    return float(np.clip(-np.sum(p * np.log(p)) / np.log(N), 0.0, 1.0))


def _scores_combo_fixed(attn: np.ndarray, pc1: np.ndarray) -> np.ndarray:
    """Return (N,) float32 fixed equal-weight blend: 0.5 × attn + 0.5 × PC1."""
    return (0.5 * attn + 0.5 * pc1).astype(np.float32)


def _scores_combo_entropy(attn: np.ndarray, pc1: np.ndarray) -> np.ndarray:
    """Return (N,) float32 entropy-weighted blend of attention and PC1.

    Each map's informativeness weight is ``1 − normalized_entropy``.  The two
    weights are normalized to sum to 1.  A flat map is automatically suppressed
    in favor of the sharper signal.  Falls back to 0.5/0.5 when both maps are
    equally degenerate (both weights near zero).

    Args:
        attn: (N,) float32 attention scores in [0, 1].
        pc1:  (N,) float32 soft PCA scores in [0, 1].

    Returns:
        (N,) float32 blended map in [0, 1].
    """
    w_attn = 1.0 - _map_entropy(attn)
    w_pc1  = 1.0 - _map_entropy(pc1)
    denom  = w_attn + w_pc1
    if denom < 1e-8:
        w_attn, w_pc1 = 0.5, 0.5
    else:
        w_attn, w_pc1 = w_attn / denom, w_pc1 / denom
    return (w_attn * attn + w_pc1 * pc1).astype(np.float32)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _to_output_tensor(
    scores: np.ndarray,
    grid_h: int,
    grid_w: int,
    H_out: int,
    W_out: int,
) -> np.ndarray:
    """Reshape (N,) patch scores to a (H_out, W_out) heatmap.

    Bilinearly upsamples from the patch grid to the original image resolution,
    then re-normalizes to [0, 1] to correct any interpolation-induced drift.

    Args:
        scores:       (N,) float32 values in [0, 1].
        grid_h, grid_w: patch grid dimensions.
        H_out, W_out: target spatial dimensions.

    Returns:
        (H_out, W_out) float32 numpy array in [0, 1].
    """
    t  = torch.from_numpy(scores).reshape(1, 1, grid_h, grid_w).float()
    t  = F.interpolate(t, size=(H_out, W_out), mode="bilinear", align_corners=False)
    t  = t.squeeze()
    lo, hi = t.min(), t.max()
    if (hi - lo).abs() > 1e-8:
        t = (t - lo) / (hi - lo)
    return t.numpy()


# ===========================================================================
# Public AttributionMethod classes
# ===========================================================================

class Dinov2AllMethodsBase(AttributionMethod):
    """Shared base for all six DINOv2 attribution methods.

    Handles model loading, configuration knobs, and the single shared forward
    pass.  Subclasses implement only ``_scores_from_fwd``.
    """

    def __init__(self, method_name: str) -> None:
        super().__init__(method_name)
        self.use_registers: bool  = getattr(config, "DINO_PCA_USE_REGISTERS", True)
        self.n_components:  int   = getattr(config, "DINO_PCA_N_COMPONENTS",  1)
        self.border:        int   = getattr(config, "DINO_PCA_BORDER",        1)
        self.center_frac:   float = getattr(config, "DINO_PCA_CENTER_FRAC",   0.4)
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = _ensure_model(self.use_registers)
        return self._model

    def _shared_forward(self, img_pil: Image.Image) -> tuple[dict, np.ndarray]:
        """Run the shared forward pass and compute the soft PCA base map.

        Returns:
            fwd:     Full forward-pass dict (tokens, attention, grid info).
            pc_base: (N,) float32 soft PCA map in [0, 1] — the shared base.
        """
        fwd     = _forward_once(self._get_model(), img_pil)
        pc_base = _pca_soft_base(fwd, self.n_components, self.border, self.center_frac)
        return fwd, pc_base

    def _scores_from_fwd(self, fwd: dict, pc_base: np.ndarray) -> np.ndarray:
        """Override in subclasses: return (N,) float32 patch-level scores."""
        raise NotImplementedError

    def _heatmap_single(self, img_pil: Image.Image) -> np.ndarray:
        fwd, pc_base = self._shared_forward(img_pil)
        scores = self._scores_from_fwd(fwd, pc_base)
        return _to_output_tensor(
            scores, fwd["grid_h"], fwd["grid_w"], fwd["H_orig"], fwd["W_orig"]
        )

    def compute(
        self,
        model,                   # unused — DINOv2 is self-contained
        images: torch.Tensor,    # (B, C, H, W) ImageNet-normalized float32
        targets: torch.Tensor,   # (B,) — unsupervised, not used
    ) -> torch.Tensor:
        """Compute heatmaps for a batch of images.

        Args:
            model:   Ignored. DINOv2 is loaded internally.
            images:  Float32 tensor (B, C, H, W), ImageNet-normalized.
            targets: Unused. Accepted for interface compatibility.

        Returns:
            Float32 tensor (B, H, W) in [0, 1].
        """
        B, C, H, W = images.shape
        pil_images = _tensor_to_pil(images)
        heatmaps   = np.stack([self._heatmap_single(p) for p in pil_images], axis=0)

        if heatmaps.shape[1] != H or heatmaps.shape[2] != W:
            ht = torch.from_numpy(heatmaps).unsqueeze(1).float()
            ht = F.interpolate(ht, size=(H, W), mode="bilinear", align_corners=False)
            heatmaps = ht.squeeze(1).numpy()

        return torch.from_numpy(heatmaps).float()


# ---------------------------------------------------------------------------
# Method 1 — DINO_ATTN
# ---------------------------------------------------------------------------

class Dinov2AttnMethod(Dinov2AllMethodsBase):
    """Attribution via head-averaged CLS-to-patch self-attention.

    Uses the last transformer block's attention weights.  The CLS token's
    attention distribution over patch tokens directly encodes which spatial
    regions contributed most to the global representation.  No PCA is involved;
    the soft PCA base is computed but not used here (it comes for free from the
    shared forward pass used by other methods in the same batch).

    Signal: continuous probability distribution (softmax), no thresholding.
    """

    def __init__(self) -> None:
        super().__init__("dinov2_attn")

    def _scores_from_fwd(self, fwd: dict, pc_base: np.ndarray) -> np.ndarray:
        return _scores_attn(fwd)


# ---------------------------------------------------------------------------
# Method 2 — DINO_PC1
# ---------------------------------------------------------------------------

class Dinov2Pc1Method(Dinov2AllMethodsBase):
    """Attribution via the single best soft PCA component.

    Directly returns the shared soft PCA base map: the best component (among
    the first ``DINO_PCA_N_COMPONENTS``) selected by center-border contrast
    scoring, polarity-corrected by CLS cosine similarity, normalized to [0,1].

    No binarization, no Gaussian blur, no distance transform.
    """

    def __init__(self) -> None:
        super().__init__("dinov2_pc1")

    def _scores_from_fwd(self, fwd: dict, pc_base: np.ndarray) -> np.ndarray:
        return _scores_pc1(fwd, pc_base)


# ---------------------------------------------------------------------------
# Method 3 — DINO_PC_EV
# ---------------------------------------------------------------------------

class Dinov2PcEigenweightedMethod(Dinov2AllMethodsBase):
    """Attribution via explained-variance-weighted PC1 + PC2 + PC3.

    Runs a fresh 3-component PCA (independent of the base map's single-component
    selection), polarity-corrects each component via CLS cosine similarity,
    normalizes each independently to [0, 1], then blends by explained variance
    ratio.  Captures textures and complex multi-part objects better than PC1 alone.
    """

    def __init__(self) -> None:
        super().__init__("dinov2_pc_ev")

    def _scores_from_fwd(self, fwd: dict, pc_base: np.ndarray) -> np.ndarray:
        return _scores_pc_eigenweighted(fwd, self.n_components)


# ---------------------------------------------------------------------------
# Method 4 — DINO_PC_L2
# ---------------------------------------------------------------------------

class Dinov2PcL2Method(Dinov2AllMethodsBase):
    """Attribution via L2 norm of patch projections onto PC1 + PC2 + PC3.

    Squaring each projection before summing eliminates sign ambiguity without
    needing polarity correction.  Measures total distance from the feature mean
    in the 3-PC subspace — any semantically distinctive patch scores high,
    producing broad spatial activation complementary to PC1.
    """

    def __init__(self) -> None:
        super().__init__("dinov2_pc_l2")

    def _scores_from_fwd(self, fwd: dict, pc_base: np.ndarray) -> np.ndarray:
        return _scores_pc_l2(fwd, self.n_components)


# ---------------------------------------------------------------------------
# Method 5 — COMBO_FIXED
# ---------------------------------------------------------------------------

class Dinov2ComboFixedMethod(Dinov2AllMethodsBase):
    """Attribution via fixed equal-weight blend of attention and PC1.

    Both maps are independently in [0, 1] before blending (attention is
    normalized in ``_forward_once``; PC1 is the soft base map).  The 0.5/0.5
    split is therefore a genuine equal contribution.  Blending is performed at
    patch resolution before a single upsample call to avoid double interpolation.

    Scientific use: diagnostic baseline — if this outperforms both individual
    methods, the two signals are genuinely complementary.
    """

    def __init__(self) -> None:
        super().__init__("dinov2_combo_fixed")

    def _scores_from_fwd(self, fwd: dict, pc_base: np.ndarray) -> np.ndarray:
        return _scores_combo_fixed(_scores_attn(fwd), pc_base)


# ---------------------------------------------------------------------------
# Method 6 — COMBO_ENT
# ---------------------------------------------------------------------------

class Dinov2ComboEntropyMethod(Dinov2AllMethodsBase):
    """Attribution via entropy-adaptive blend of attention and PC1.

    Each map's informativeness weight is ``1 − normalized_Shannon_entropy``.
    A flat, uninformative map is automatically suppressed in favor of the
    sharper signal.  Falls back to 0.5/0.5 when both maps are equally
    degenerate.  This is the most robust combo method for diverse datasets.

    Advantage over COMBO_FIXED: if one signal is "mushy" for a given image
    (e.g. attention spreads uniformly over a cluttered background), the entropy
    weight automatically reduces its contribution without any manual tuning.
    """

    def __init__(self) -> None:
        super().__init__("dinov2_combo_ent")

    def _scores_from_fwd(self, fwd: dict, pc_base: np.ndarray) -> np.ndarray:
        return _scores_combo_entropy(_scores_attn(fwd), pc_base)

# ---------------------------------------------------------------------------
# Method 7 — COMBO_ENT_SMOOTH
# ---------------------------------------------------------------------------

class Dinov2ComboEntSmoothMethod(Dinov2AllMethodsBase):
    """Attribution via entropy-weighted attn + PC1, with Gaussian smoothing
    and a PC1 gate to reduce spatial fragmentation.

    Extends COMBO_ENT (Method 6) with two post-processing steps:

    1. Gaussian smoothing (sigma in patch units) — reduces speckle from
       spatially non-continuous attention activations before gating, so the
       gate boundary is not itself noisy.  The map is re-normalized to [0,1]
       after smoothing so that gamma's effect is image-independent.

    2. PC1 gate — multiplies the smoothed map by (pc1_map ** gamma), where
       pc1_map is the shared soft PCA base in [0,1].  This is a soft mask
       that suppresses patches with low object-likeness according to PCA.
       gamma > 1 sharpens the object boundary more aggressively.

    3. Alpha blend — blends the gated and ungated smoothed maps:
           hm = alpha * gated + (1 - alpha) * smoothed
       Prevents over-suppression of fine details (thin limbs, textured parts)
       where PC1 underestimates the object extent.

    Hyperparameters (configurable via constructor):
        sigma  (float, default 1.0) — Gaussian std-dev in patch units.
                                      Set to 0 to disable smoothing.
        gamma  (float, default 1.0) — PC1 gate exponent (1 = linear).
        alpha  (float, default 0.7) — blend weight for the gated map
                                      (1.0 = fully gated, 0.0 = no gate).
    """

    # def __init__(self, sigma: float = 1.0, gamma: float = 1.0, alpha: float = 0.7) -> None: OLD-> so minor on the previous (5)
    # def __init__(self, sigma: float = 1.0, gamma: float = 2.5, alpha: float = 0.85) -> None: OLD -> worked well (4)
    # def __init__(self, sigma: float = 1.3, gamma: float = .9, alpha: float = 0.6) -> None: OLD -> worked best so far (3)
    # def __init__(self, sigma: float = 1.6, gamma: float = .8, alpha: float = 0.55) -> None: OLD -> worked even better (2)
    # def __init__(self, sigma: float = 2, gamma: float = .7, alpha: float = 0.45) -> None: OLD -> slight better results (1)
    def __init__(self, sigma: float = 2, gamma: float = .7, alpha: float = 0.45) -> None:
        super().__init__("dinov2_combo_ent_smooth")
        self.sigma = sigma
        self.gamma = gamma
        self.alpha = alpha

    def _scores_from_fwd(self, fwd: dict, pc_base: np.ndarray) -> np.ndarray:
        import math

        attn = _scores_attn(fwd)   # (N,) float32, [0,1]
        pc1  = pc_base             # (N,) float32, [0,1]

        grid_h = fwd["grid_h"]
        grid_w = fwd["grid_w"]

        # --- Step 1: entropy-weighted blend (same as COMBO_ENT) ----------
        base = _scores_combo_entropy(attn, pc1)   # (N,) float32, [0,1]

        # --- Step 2: Gaussian smoothing on patch grid --------------------
        hm = torch.from_numpy(base).float().reshape(1, 1, grid_h, grid_w).to(DEVICE)

        if self.sigma > 0:
            half   = math.ceil(3.0 * self.sigma)
            ksize  = 2 * half + 1
            coords = torch.arange(ksize, dtype=torch.float32, device=DEVICE) - half
            g      = torch.exp(-(coords ** 2) / (2.0 * self.sigma ** 2))
            g      = g / g.sum()
            kernel = torch.outer(g, g)
            kernel = (kernel / kernel.sum()).view(1, 1, ksize, ksize)
            hm = F.conv2d(hm, kernel, padding=half)

        # Re-normalize after smoothing so gamma's effect is image-independent
        lo, hi = hm.min(), hm.max()
        hm = (hm - lo) / (hi - lo + 1e-8) if (hi - lo).abs() > 1e-8 \
             else torch.zeros_like(hm)

        # --- Step 3: PC1 gate --------------------------------------------
        pc1_map = torch.from_numpy(pc1).float().reshape(1, 1, grid_h, grid_w).to(DEVICE)
        gated   = hm * (pc1_map ** self.gamma)

        # --- Step 4: alpha blend gated vs smoothed -----------------------
        hm = self.alpha * gated + (1.0 - self.alpha) * hm

        # Flatten back to (N,) for the standard _to_output_tensor pipeline
        scores = hm.reshape(-1).cpu().numpy().astype(np.float32)

        lo, hi = scores.min(), scores.max()
        if hi - lo > 1e-8:
            scores = (scores - lo) / (hi - lo)
        else:
            scores = np.zeros_like(scores)

        return scores