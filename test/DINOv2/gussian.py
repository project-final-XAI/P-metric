import torch
import numpy as np
import cv2
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from transformers import AutoImageProcessor, AutoModel
import config
from attribution.base import AttributionMethod

DEVICE = torch.device(config.DEVICE if hasattr(config, "DEVICE") else ("cuda" if torch.cuda.is_available() else "cpu"))
_PATCH_SIZE    = 14
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
_N_REGISTERS   = 4


def _normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)


def _tensor_batch_to_pil(images: torch.Tensor) -> list[Image.Image]:
    mean = torch.tensor(_IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std  = torch.tensor(_IMAGENET_STD,  device=images.device).view(1, 3, 1, 1)
    imgs_rgb   = (images * std + mean).clamp(0, 1)
    imgs_uint8 = (imgs_rgb * 255).byte().cpu()
    return [Image.fromarray(imgs_uint8[i].permute(1, 2, 0).numpy()) for i in range(imgs_uint8.shape[0])]


class Dinov2BinarySegmentMethod(AttributionMethod):
    def __init__(self) -> None:
        super().__init__("dinov2_binary_segment")
        self.use_registers  = getattr(config, "DINO_SEG_USE_REGISTERS",  True)
        self.n_components   = getattr(config, "DINO_SEG_N_COMPONENTS",   5)
        self.center_frac    = getattr(config, "DINO_SEG_CENTER_FRAC",    0.4)
        self.grabcut_iters  = getattr(config, "DINO_SEG_GRABCUT_ITERS",  5)
        self._processor, self._dino_model = None, None

    # ------------------------------------------------------------------
    def _ensure_model(self):
        if self._dino_model is None:
            model_name = (
                "facebook/dinov2-with-registers-small"
                if self.use_registers
                else "facebook/dinov2-base"
            )
            self._processor  = AutoImageProcessor.from_pretrained(model_name)
            self._dino_model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()

    # ------------------------------------------------------------------
    def _extract_patch_features(self, inputs: dict, grid_h: int, grid_w: int) -> np.ndarray:
        """Average last 4 hidden layers — balances semantics with spatial detail."""
        with torch.no_grad():
            out = self._dino_model(**inputs, output_hidden_states=True)
        last_4 = torch.stack(out.hidden_states[-4:], dim=0).mean(0)  # (1, seq, D)
        return last_4[0, -grid_h * grid_w:, :].cpu().numpy()         # (N, D)

    # ------------------------------------------------------------------
    def _build_center_mask(self, grid_h: int, grid_w: int) -> np.ndarray:
        h0 = int(grid_h * (0.5 - self.center_frac / 2))
        h1 = int(grid_h * (0.5 + self.center_frac / 2))
        w0 = int(grid_w * (0.5 - self.center_frac / 2))
        w1 = int(grid_w * (0.5 + self.center_frac / 2))
        mask = np.zeros((grid_h, grid_w), dtype=bool)
        mask[h0:h1, w0:w1] = True
        return mask

    # ------------------------------------------------------------------
    def _coarse_patch_mask(self, patch_feats: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
        """
        K-means on first 3 PCA dims → binary (grid_h, grid_w) patch mask.
        Returns 1=foreground, 0=background.
        """
        n            = min(self.n_components, patch_feats.shape[0] - 1)
        pca_features = PCA(n_components=n).fit_transform(patch_feats)

        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        labels = kmeans.fit_predict(pca_features[:, :3]).reshape(grid_h, grid_w)

        center_mask = self._build_center_mask(grid_h, grid_w)
        fg_label    = 0 if labels[center_mask].mean() > 0.5 else 1
        return (labels == fg_label).astype(np.uint8)

    # ------------------------------------------------------------------
    def _patch_mask_to_grabcut_hint(
        self,
        patch_mask: np.ndarray,   # (grid_h, grid_w) binary
        img_h: int,
        img_w: int,
        grid_h: int,
        grid_w: int,
    ) -> np.ndarray:
        """
        Convert coarse patch mask → GrabCut initialisation mask (same size as image).

        GrabCut mask values:
          cv2.GC_BGD (0)    = definite background
          cv2.GC_FGD (1)    = definite foreground
          cv2.GC_PR_BGD (2) = probably background
          cv2.GC_PR_FGD (3) = probably foreground

        Strategy:
          - Erode FG patches  → definite FG core
          - Dilate FG patches → probable FG ring
          - Everything else   → definite BG
          - One-patch-wide border of image → forced BG (GrabCut needs some BG signal)
        """
        kernel = np.ones((3, 3), np.uint8)

        # Shrink FG to get a confident core, expand to get a probable ring
        fg_eroded  = cv2.erode(patch_mask,  kernel, iterations=1)
        fg_dilated = cv2.dilate(patch_mask, kernel, iterations=1)

        # Upsample both to full image size (NEAREST keeps binary values)
        core_full = cv2.resize(fg_eroded,  (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        ring_full = cv2.resize(fg_dilated, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        # Build GrabCut init mask
        gc_mask = np.where(ring_full,  cv2.GC_PR_FGD, cv2.GC_BGD ).astype(np.uint8)
        gc_mask = np.where(core_full,  cv2.GC_FGD,    gc_mask    )

        # Force image border pixels to definite background
        border = max(2, img_h // 20)
        gc_mask[:border,  :]  = cv2.GC_BGD
        gc_mask[-border:, :]  = cv2.GC_BGD
        gc_mask[:,  :border]  = cv2.GC_BGD
        gc_mask[:, -border:]  = cv2.GC_BGD

        return gc_mask

    # ------------------------------------------------------------------
    def _run_grabcut(
        self,
        img_bgr: np.ndarray,
        gc_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Run GrabCut with mask initialisation.
        Returns a binary (H, W) uint8 mask: 1=foreground, 0=background.
        """
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(
            img_bgr, gc_mask, None,
            bgd_model, fgd_model,
            self.grabcut_iters,
            cv2.GC_INIT_WITH_MASK,
        )

        # Both GC_FGD and GC_PR_FGD count as foreground
        binary = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        return binary

    # ------------------------------------------------------------------
    def _postprocess(self, binary: np.ndarray) -> np.ndarray:
        """
        Keep only the largest connected foreground component,
        then morphologically close small holes.
        """
        kernel = np.ones((5, 5), np.uint8)

        # Find largest connected component
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n_labels <= 1:
            return binary

        # Label 0 is background — find largest non-background component
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cleaned = (labels == largest).astype(np.uint8)

        # Fill small interior holes
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        return cleaned

    # ------------------------------------------------------------------
    def _compute_single(self, img_pil: Image.Image) -> np.ndarray:
        """Returns a pixel-precise binary (H, W) float32 mask."""
        img_rgb = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_h, img_w = img_bgr.shape[:2]

        # ── DINO features → coarse patch mask ────────────────────────
        inputs = self._processor(images=img_pil, return_tensors="pt").to(DEVICE)
        grid_h = inputs["pixel_values"].shape[2] // _PATCH_SIZE
        grid_w = inputs["pixel_values"].shape[3] // _PATCH_SIZE

        patch_feats = self._extract_patch_features(inputs, grid_h, grid_w)
        patch_mask  = self._coarse_patch_mask(patch_feats, grid_h, grid_w)

        # ── Patch mask → GrabCut hint → pixel mask ───────────────────
        gc_mask = self._patch_mask_to_grabcut_hint(patch_mask, img_h, img_w, grid_h, grid_w)
        binary  = self._run_grabcut(img_bgr, gc_mask)

        # ── Keep largest component, fill holes ────────────────────────
        binary = self._postprocess(binary)

        return binary.astype(np.float32)

    # ------------------------------------------------------------------
    def compute(self, model, images, targets):
        """Returns (B, H, W) float tensor with values 0.0 or 1.0."""
        self._ensure_model()
        pil_images = _tensor_batch_to_pil(images)
        masks = [self._compute_single(img) for img in pil_images]

        res = torch.from_numpy(np.stack(masks)).float()
        return torch.nn.functional.interpolate(
            res.unsqueeze(1), size=images.shape[2:], mode="nearest"
        ).squeeze(1)