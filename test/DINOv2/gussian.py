"""
Unified DINOv2 XAI Attribution Method.

Single forward pass through the HuggingFace backend extracts both patch
features (for PCA) and self-attention weights simultaneously — no second
model load, no second forward pass.

Key improvements over the original unified method
--------------------------------------------------
1. Attention-adjudicated PCA selection (replaces broken center heuristic)
   → Uses soft-IoU between each PCA candidate's foreground and the attention
     map to pick the component that describes the same object the model
     attended to — works correctly for off-center objects.

2. Per-head attention + head agreement weighting
   → Instead of naively averaging all heads, we weight each head by how
     "peaked" it is (entropy-based confidence). Noisy, diffuse heads
     contribute less; sharp, discriminative heads contribute more.

3. Multi-scale PCA evaluation
   → Evaluates components at two thresholds (fine + coarse) and picks the
     one with better attention-overlap, giving robustness to objects of
     very different sizes.

4. Attention fallback is explicit and logged
   → If attention is flat (all heads diffuse), falls back to center heuristic
     and logs clearly so the caller knows which path was taken.

5. Gamma correction on final heatmap
   → Applies fused_map ** gamma (default 0.7) to lift mid-range values,
     giving the characteristic XAI "warm glow" rather than binary blobs.

Available modes (config.DINO_METHOD_MODE):
  "pca"       – PCA + Gaussian only (attention used only for component selection)
  "attention" – Weighted-head attention heatmap only
  "hybrid"    – PCA foreground × attention detail (default, recommended)

Configuration keys (all optional, read from ``config``)
--------------------------------------------------------
DINO_METHOD_MODE            str    "hybrid"
DINO_USE_REGISTERS          bool   True
DINO_PCA_N_COMPONENTS       int    5
DINO_PCA_THRESHOLD_Q        float  0.45
DINO_PCA_SIGMA              float  10.0
DINO_PCA_BORDER             int    1
DINO_PCA_CENTER_FRAC        float  0.4
DINO_ATTENTION_SMOOTH_SIGMA float  0.0
DINO_ATTN_MIN_PEAK          float  0.25   – min peak to trust attention for selection
DINO_HYBRID_GAMMA           float  0.7    – output gamma (< 1 lifts midrange)
"""

from __future__ import annotations

from typing import Literal, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel

import config
from attribution.base import AttributionMethod


# ---------------------------------------------------------------------------
# Device & constants
# ---------------------------------------------------------------------------
DEVICE = torch.device(
    config.DEVICE if hasattr(config, "DEVICE")
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

_PATCH_SIZE    = 14
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

_MODEL_CACHE: dict[str, tuple] = {}   # "std" | "reg" → (processor, model)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)


def _soft_iou(fg: np.ndarray, attn: np.ndarray) -> float:
    """
    Soft IoU between a [0,1] foreground mask and a [0,1] attention map.
    Treats pixel values as membership probabilities.
    """
    inter = (fg * attn).sum()
    union = (fg + attn - fg * attn).sum()
    return float(inter / (union + 1e-8))


def _attention_entropy(attn_head: np.ndarray) -> float:
    """
    Normalised Shannon entropy of a single attention head's distribution.
    Low entropy = peaked / discriminative head.
    High entropy = diffuse / uninformative head.
    Returns value in [0, 1].
    """
    p = attn_head.flatten()
    p = p / (p.sum() + 1e-8)
    H = -np.sum(p * np.log(p + 1e-12))
    H_max = np.log(len(p) + 1e-12)
    return float(H / (H_max + 1e-8))


def _create_soft_heatmap(binary_mask: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """Distance-transform + Gaussian blur → smooth XAI blob."""
    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    dist    = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    heatmap = cv2.GaussianBlur(dist, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    mx = heatmap.max()
    if mx > 0:
        heatmap = (heatmap - heatmap.min()) / (mx - heatmap.min() + 1e-8)
    return heatmap


def _tensor_batch_to_pil(images: torch.Tensor) -> list[Image.Image]:
    mean = torch.tensor(_IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std  = torch.tensor(_IMAGENET_STD,  device=images.device).view(1, 3, 1, 1)
    imgs = (images * std + mean).clamp(0, 1)
    imgs = (imgs * 255).byte().cpu()
    return [Image.fromarray(imgs[i].permute(1, 2, 0).numpy())
            for i in range(imgs.shape[0])]


def _border_mask(gh: int, gw: int, border: int) -> np.ndarray:
    m = np.zeros((gh, gw), dtype=bool)
    m[:border, :] = m[-border:, :] = m[:, :border] = m[:, -border:] = True
    return m


def _center_mask(gh: int, gw: int, cf: float) -> np.ndarray:
    m = np.zeros((gh, gw), dtype=bool)
    h0 = max(int(gh * (0.5 - cf / 2)), 0);  h1 = min(int(gh * (0.5 + cf / 2)), gh)
    w0 = max(int(gw * (0.5 - cf / 2)), 0);  w1 = min(int(gw * (0.5 + cf / 2)), gw)
    m[h0:h1, w0:w1] = True
    return m


# ===========================================================================
# Dinov2UnifiedMethod
# ===========================================================================

class Dinov2UnifiedMethod(AttributionMethod):
    """
    Unified DINOv2 XAI attribution — single forward pass, attention-adjudicated PCA.

    See module docstring for full design rationale.
    """

    def __init__(self) -> None:
        super().__init__("dinov2_unified")

        self.mode: Literal["pca", "attention", "hybrid"] = getattr(
            config, "DINO_METHOD_MODE", "hybrid"
        )
        self.use_registers  = getattr(config, "DINO_USE_REGISTERS",           True)
        self.n_components   = getattr(config, "DINO_PCA_N_COMPONENTS",        5)
        self.threshold_q    = getattr(config, "DINO_PCA_THRESHOLD_Q",         0.45)
        self.sigma          = getattr(config, "DINO_PCA_SIGMA",               10.0)
        self.border         = getattr(config, "DINO_PCA_BORDER",              1)
        self.center_frac    = getattr(config, "DINO_PCA_CENTER_FRAC",         0.4)
        self.smooth_sigma   = getattr(config, "DINO_ATTENTION_SMOOTH_SIGMA",  0.0)
        self.attn_min_peak  = getattr(config, "DINO_ATTN_MIN_PEAK",           0.25)
        self.gamma          = getattr(config, "DINO_HYBRID_GAMMA",            0.7)

        self._processor:   Optional[AutoImageProcessor] = None
        self._dino_model:  Optional[AutoModel]          = None

    # ------------------------------------------------------------------
    # Model management — single model, single load
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        key = "reg" if self.use_registers else "std"
        if key not in _MODEL_CACHE:
            name = (
                "facebook/dinov2-with-registers-small"
                if self.use_registers else "facebook/dinov2-base"
            )
            print(f"[Dinov2UnifiedMethod] Loading {name} on {DEVICE} …")
            processor = AutoImageProcessor.from_pretrained(name)
            model = AutoModel.from_pretrained(
                name, output_attentions=True
            ).to(DEVICE)
            model.eval()
            _MODEL_CACHE[key] = (processor, model)

        self._processor, self._dino_model = _MODEL_CACHE[key]

    # ------------------------------------------------------------------
    # Attention extraction — entropy-weighted head aggregation
    # ------------------------------------------------------------------

    def _extract_attention(
        self,
        last_attn: torch.Tensor,  # (1, num_heads, N, N)
        num_patches: int,
    ) -> np.ndarray:
        """
        Compute a (num_patches,) CLS-attention vector weighted by head confidence.

        Each head is weighted by (1 - normalised_entropy):
          - A peaked head (attends to a specific region) gets high weight.
          - A diffuse head (uniform attention) gets near-zero weight.

        Falls back to uniform average if all heads are equally diffuse.

        Returns a (num_patches,) array in [0, 1].
        """
        attn_heads = last_attn[0, :, 0, -num_patches:].cpu().numpy()  # (nh, num_patches)

        # Per-head confidence = 1 - normalised entropy
        weights = np.array([
            1.0 - _attention_entropy(h) for h in attn_heads
        ])  # (nh,)

        w_sum = weights.sum()
        if w_sum < 1e-6:
            # All heads equally diffuse — fall back to simple mean
            weights = np.ones(len(weights))
            w_sum   = float(len(weights))

        weighted_attn = (weights[:, None] * attn_heads).sum(0) / w_sum  # (num_patches,)
        return _normalize(weighted_attn)

    # ------------------------------------------------------------------
    # PCA candidate generation — multi-scale thresholds
    # ------------------------------------------------------------------

    def _get_pca_candidates(
        self,
        feat_np: np.ndarray,  # (num_patches, dim)
        gh: int,
        gw: int,
    ) -> list[tuple[np.ndarray, float]]:
        """
        Run PCA and return all (component × polarity × threshold) candidates.

        For each component and polarity we generate two foreground masks:
          - Fine   (threshold_q)        → tight foreground
          - Coarse (threshold_q - 0.15) → looser foreground

        Returns list of (normalised_map, threshold_used) tuples.
        """
        pca_feat = PCA(n_components=self.n_components).fit_transform(feat_np)

        candidates: list[tuple[np.ndarray, float]] = []
        n = min(self.n_components, pca_feat.shape[1])

        for i in range(n):
            for sign in (+1.0, -1.0):
                cand = _normalize(sign * pca_feat[:, i]).reshape(gh, gw)
                # Two scales per candidate
                for q in (self.threshold_q, max(self.threshold_q - 0.15, 0.1)):
                    candidates.append((cand, q))

        return candidates

    # ------------------------------------------------------------------
    # PCA component selection — soft-IoU against attention
    # ------------------------------------------------------------------

    def _select_best_pca(
        self,
        candidates:  list[tuple[np.ndarray, float]],
        attn_map:    np.ndarray,   # (gh, gw) in [0,1]
        gh: int,
        gw: int,
    ) -> tuple[np.ndarray, float, str]:
        """
        Pick the (candidate, threshold) pair with the best soft-IoU against
        the attention map.

        Falls back to the center-vs-border heuristic if attention is
        uninformative (peak < attn_min_peak).

        Returns (best_map, best_threshold, description_str).
        """
        # Resize attention to patch grid if dimensions differ
        if attn_map.shape != (gh, gw):
            at = torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0).float()
            at = F.interpolate(at, size=(gh, gw), mode="bilinear", align_corners=False)
            attn_map = at.squeeze().numpy()

        attn_norm = _normalize(attn_map)
        use_attn  = attn_norm.max() >= self.attn_min_peak

        best_map, best_q, best_score, best_label = None, self.threshold_q, -np.inf, ""

        if use_attn:
            for idx, (cand, q) in enumerate(candidates):
                fg    = (cand > np.quantile(cand, q)).astype(float)
                score = _soft_iou(fg, attn_norm)
                if score > best_score:
                    best_score = score
                    best_map   = cand
                    best_q     = q
                    best_label = (
                        f"attn-IoU comp={idx // 2 // 2 + 1} "
                        f"{'+'if (idx//2)%2==0 else'-'} "
                        f"q={q:.2f} score={score:.4f}"
                    )
        else:
            # Fallback: center-vs-border on the per-component (not per-candidate) maps
            bf = _border_mask(gh, gw, self.border).flatten()
            cf = _center_mask(gh, gw, self.center_frac).flatten()
            seen: set[int] = set()
            for idx, (cand, q) in enumerate(candidates):
                comp_id = idx // 2   # each component appears twice (two thresholds)
                if comp_id in seen:
                    continue
                seen.add(comp_id)
                flat  = cand.flatten()
                score = flat[cf].mean() - flat[bf].mean()
                if score > best_score:
                    best_score = score
                    best_map   = cand
                    best_q     = q
                    best_label = (
                        f"center-fallback comp={comp_id//2+1} "
                        f"{'+'if comp_id%2==0 else'-'} score={score:.4f}"
                    )

        print(f"[Dinov2UnifiedMethod] PCA: {best_label}")
        return best_map, best_q, best_label

    # ------------------------------------------------------------------
    # Core: single-image heatmap (single forward pass)
    # ------------------------------------------------------------------

    def _heatmap_single(self, img_pil: Image.Image) -> np.ndarray:
        inputs = self._processor(images=img_pil, return_tensors="pt").to(DEVICE)

        h, w   = inputs["pixel_values"].shape[2], inputs["pixel_values"].shape[3]
        gh, gw = h // _PATCH_SIZE, w // _PATCH_SIZE
        np_    = gh * gw
        W_orig, H_orig = img_pil.size

        # ── Single forward pass ──────────────────────────────────────────
        with torch.no_grad():
            outputs = self._dino_model(**inputs)

        # Strip CLS + register tokens → patch features
        seq_len  = outputs.last_hidden_state.shape[1]
        num_skip = seq_len - np_
        feat_np  = outputs.last_hidden_state[:, num_skip:, :].squeeze(0).cpu().numpy()

        # Last block attention
        last_attn = outputs.attentions[-1]   # (1, num_heads, N, N)

        # ── Attention map (entropy-weighted heads) ───────────────────────
        attn_vec = self._extract_attention(last_attn, np_)          # (np_,)
        attn_map = attn_vec.reshape(gh, gw)                         # (gh, gw)

        if self.smooth_sigma > 0:
            attn_map = _normalize(gaussian_filter(attn_map, sigma=self.smooth_sigma))

        attn_full = cv2.resize(attn_map, (W_orig, H_orig))

        # ── Mode: attention only ─────────────────────────────────────────
        if self.mode == "attention":
            result = _normalize(attn_full)
            if self.gamma != 1.0:
                result = result ** self.gamma
            return result

        # ── PCA candidates + attention-adjudicated selection ─────────────
        candidates          = self._get_pca_candidates(feat_np, gh, gw)
        best_map, best_q, _ = self._select_best_pca(candidates, attn_map, gh, gw)

        pca_binary = (best_map > np.quantile(best_map, best_q)).astype(np.float32)
        pca_binary_full = cv2.resize(pca_binary, (W_orig, H_orig))

        # Soft blob — slightly wider sigma for PCA to cover full object extent
        pca_soft = _create_soft_heatmap(pca_binary_full, sigma=self.sigma)

        # ── Mode: pca only ───────────────────────────────────────────────
        if self.mode == "pca":
            result = pca_soft
            if self.gamma != 1.0:
                result = result ** self.gamma
            return result

        # ── Mode: hybrid ────────────────────────────────────────────────
        # PCA supplies the object boundary / shape envelope.
        # Attention supplies the within-object saliency detail.
        # Multiply: only locations inside the PCA foreground AND
        #           attended to by the model receive high attribution.
        #
        # Post-multiply smoothing fuses their edges naturally.
        fused = pca_soft * attn_full
        fused = _normalize(fused)

        if self.sigma > 0:
            fused = _normalize(gaussian_filter(fused, sigma=self.sigma / 3.0))

        # Gamma lift — raises mid-range values for warmer XAI appearance
        if self.gamma != 1.0:
            fused = fused ** self.gamma

        return fused

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(
        self,
        model,                   # unused — DINOv2 is self-contained
        images:  torch.Tensor,   # (B, C, H, W) ImageNet-normalised
        targets: torch.Tensor,   # unused
    ) -> torch.Tensor:
        """
        Args:
            model:   Unused.
            images:  (B, C, H, W) float32, ImageNet-normalised.
            targets: Unused.
        Returns:
            (B, H, W) float32 in [0, 1].
        """
        self._ensure_model()
        B, C, H, W = images.shape
        pil_images  = _tensor_batch_to_pil(images)

        heatmaps = np.stack(
            [self._heatmap_single(p) for p in pil_images], axis=0
        )

        if heatmaps.shape[1] != H or heatmaps.shape[2] != W:
            ht = torch.from_numpy(heatmaps).unsqueeze(1).float()
            ht = F.interpolate(ht, size=(H, W), mode="bilinear", align_corners=False)
            heatmaps = ht.squeeze(1).numpy()

        return torch.from_numpy(heatmaps).float()