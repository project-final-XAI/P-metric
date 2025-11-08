"""
C3F (Connected Component-based Feature attribution) method.

Combines Grad-CAM seeds with evidence from Integrated Gradients,
SmoothGrad, and Occlusion, using region growing for connectivity.
"""

import math
import heapq
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from attribution.base import AttributionMethod
from captum.attr import IntegratedGradients, Saliency, NoiseTunnel, Occlusion
from models.architectures import get_target_layer


class C3FMethod(AttributionMethod):
    """C3F attribution with IG + SmoothGrad + Occlusion evidence."""
    
    # Hyperparameters
    SEED_PCT = 0.02      # Fraction of pixels for initial seeds
    BUDGET_PCT = 0.5     # Final pixel budget for connected region
    K_COMPONENTS = 1     # Number of seed components to grow
    ALPHA_BASE = 0.60    # Blend weight for Grad-CAM vs evidence
    EVIDENCE_THRESH = 0.1  # Fallback threshold (if evidence too weak)
    MASK_WEIGHT = 0.85   # Weight for grown mask in final heatmap
    MASK_BASE = 0.15     # Base weight in final heatmap
    
    # Evidence toggles
    USE_IG = True
    USE_SG = True
    USE_OCC = True
    
    # Occlusion settings
    OCCL_WINDOW = 32
    OCCL_STRIDE = 16
    OCCL_BASELINE_MODE = "zeros"  # "zeros" or "imagenet_mean"
    
    def __init__(self):
        super().__init__("c3f", "single", 1)
        self._attributors = {}  # Cache attributors per model
        
    def _init_attributors(self, model):
        """Initialize Captum attributors if needed (cached per model)."""
        model_id = id(model)
        if model_id in self._attributors:
            return self._attributors[model_id]
        
        attributors = {}
        if self.USE_IG:
            attributors['ig'] = IntegratedGradients(model)
        if self.USE_SG:
            attributors['sal'] = Saliency(model)
            attributors['nt'] = NoiseTunnel(attributors['sal'])
        if self.USE_OCC:
            attributors['occ'] = Occlusion(model)
        
        self._attributors[model_id] = attributors
        return attributors
    
    @staticmethod
    def _to01(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Normalize tensor to [0, 1] range."""
        t = t - t.min()
        return t / (t.max() + eps)
    
    @staticmethod
    def _topk_mask01(map01: torch.Tensor, pct: float) -> torch.Tensor:
        """Get top-k pixels mask based on percentage."""
        flat = map01.view(1, -1)
        k = max(1, int(math.ceil(pct * flat.size(1))))
        thr = torch.topk(flat, k, dim=1).values.min()
        return (map01 >= thr).float()
    
    @staticmethod
    def _smooth_depthwise(x: torch.Tensor, k: int = 5, sigma: float = 1.2) -> torch.Tensor:
        """Light separable Gaussian-like smoothing."""
        c = x.shape[1]
        device = x.device
        half = k // 2
        t = torch.arange(k, device=device, dtype=x.dtype) - half
        g1d = torch.exp(-0.5 * (t / sigma) ** 2)
        g1d = g1d / g1d.sum()
        g_row = g1d.view(1, 1, 1, k).expand(c, 1, 1, k).contiguous()
        g_col = g1d.view(1, 1, k, 1).expand(c, 1, k, 1).contiguous()
        x = F.pad(x, (half, half, half, half), mode="reflect")
        x = F.conv2d(x, g_row, groups=c)
        x = F.conv2d(x, g_col, groups=c)
        return x
    
    def _gradcam_single(self, model: nn.Module, x: torch.Tensor, 
                       class_idx: int, size: Tuple[int, int]) -> torch.Tensor:
        """Classic single-layer Grad-CAM -> (1,1,H,W) in [0,1]."""
        H, W = size
        model.eval()
        
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            target_mod = get_target_layer(model)
            acts_holder = {"acts": None}
            
            def fwd_hook(mod, inp, out):
                acts_holder["acts"] = out
                return out
            
            h = target_mod.register_forward_hook(fwd_hook)
            try:
                logits = model(x)
                score = logits[:, class_idx].sum()
                
                A = acts_holder["acts"]
                if A is None:
                    raise RuntimeError("Forward hook failed")
                
                G = torch.autograd.grad(score, A, retain_graph=False, create_graph=False)[0]
                w = G.mean(dim=(2, 3), keepdim=True)
                cam = (w * A).sum(dim=1, keepdim=True)
                cam = F.relu(cam)
                
                cam = self._to01(cam)
                cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
                cam = self._smooth_depthwise(cam, k=5, sigma=1.2)
                return self._to01(cam)
            finally:
                h.remove()
    
    def _process_attribution(self, attr: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Process single attribution: abs, sum channels, smooth, normalize."""
        attr = attr.abs().sum(1, keepdim=True)
        attr = self._smooth_depthwise(attr, k=5, sigma=1.5)
        return self._to01(attr)
    
    def _get_occlusion_baseline(self, x: torch.Tensor) -> torch.Tensor:
        """Get baseline for occlusion (zeros or ImageNet mean in normalized space)."""
        if self.OCCL_BASELINE_MODE == "zeros":
            return torch.zeros_like(x)
        # ImageNet mean in normalized space (zeros)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        baseline = (mean - mean) / std
        return baseline.expand_as(x)
    
    def _evidence_map(self, model: nn.Module, x: torch.Tensor, 
                     target: int, attributors: dict) -> torch.Tensor:
        """Evidence = mean(|IG|, |SmoothGrad|, Occlusion+)."""
        comps = []
        H, W = x.shape[-2:]
        
        if self.USE_IG and 'ig' in attributors:
            ig = attributors['ig'].attribute(x, target=target, baselines=torch.zeros_like(x), n_steps=64)
            comps.append(self._process_attribution(ig, H, W))
        
        if self.USE_SG and 'nt' in attributors:
            sg = attributors['nt'].attribute(x, nt_type="smoothgrad_sq", nt_samples=8, 
                                   stdevs=0.10, target=target)
            comps.append(self._process_attribution(sg, H, W))
        
        if self.USE_OCC and 'occ' in attributors:
            baseline = self._get_occlusion_baseline(x)
            occ = attributors['occ'].attribute(
                x.detach(),
                strides=(1, self.OCCL_STRIDE, self.OCCL_STRIDE),
                sliding_window_shapes=(3, self.OCCL_WINDOW, self.OCCL_WINDOW),
                target=target,
                baselines=baseline
            )
            occ = (-occ).relu()  # Negative effect (drop when occluded)
            occ = F.interpolate(occ, size=(H, W), mode="bilinear", align_corners=False)
            comps.append(self._process_attribution(occ, H, W))
        
        if not comps:
            return torch.ones((1, 1, H, W), device=x.device)
        
        evid = sum(comps) / float(len(comps))
        return self._to01(evid)
    
    def _grow_connected(self, seeds01: torch.Tensor, evidence01: torch.Tensor, 
                       budget_px: int, k_components: int = 1) -> torch.Tensor:
        """Region growing with 8-connectivity guided by evidence scores."""
        seeds_np = seeds01[0, 0].detach().cpu().numpy().astype(np.uint8)
        evid_np = evidence01[0, 0].detach().cpu().numpy().astype(np.float32)
        H, W = evid_np.shape
        
        comp_id, labels = cv2.connectedComponents(seeds_np, connectivity=8)
        work = np.zeros_like(seeds_np, dtype=np.uint8)
        
        if seeds_np.sum() == 0:
            y, x = np.unravel_index(np.argmax(evid_np), evid_np.shape)
            seeds_np[y, x] = 1
            comp_id, labels = cv2.connectedComponents(seeds_np, connectivity=8)
        
        scores = []
        for cid in range(1, comp_id):
            mask = (labels == cid)
            scores.append((evid_np[mask].mean(), cid))
        scores.sort(reverse=True)
        chosen = [cid for _, cid in scores[:k_components]]
        for cid in chosen:
            work |= (labels == cid).astype(np.uint8)
        
        selected = set(zip(*np.where(work > 0)))
        frontier = []
        
        def push_nbrs(y, x):
            for ny, nx in (
                (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1),
                (y - 1, x - 1), (y - 1, x + 1), (y + 1, x - 1), (y + 1, x + 1)
            ):
                if 0 <= ny < H and 0 <= nx < W and (ny, nx) not in selected:
                    heapq.heappush(frontier, (-evid_np[ny, nx], ny, nx))
        
        for (y, x) in list(selected):
            push_nbrs(y, x)
        
        target_sz = min(budget_px, H * W)
        while len(selected) < target_sz and frontier:
            _, y, x = heapq.heappop(frontier)
            if (y, x) in selected:
                continue
            work[y, x] = 1
            selected.add((y, x))
            push_nbrs(y, x)
        
        mask = np.zeros((H, W), np.uint8)
        for (y, x) in selected:
            mask[y, x] = 1
        return torch.from_numpy(mask[None, None]).to(seeds01.device).float()
    
    def compute(self, model: nn.Module, images: torch.Tensor, 
               targets: torch.Tensor) -> torch.Tensor:
        """
        Compute C3F attribution.
        
        Args:
            model: Neural network model
            images: Input images tensor (B, C, H, W)
            targets: Target class indices (B,)
            
        Returns:
            Normalized heatmap tensor (B, H, W) in [0, 1]
        """
        attributors = self._init_attributors(model)
        
        results = []
        for i in range(images.shape[0]):
            x = images[i:i+1]
            target = int(targets[i].item())
            H, W = x.shape[-2:]
            
            # 1) Classic Grad-CAM
            gcam = self._gradcam_single(model, x, target, (H, W))
            
            # 2) Evidence map
            evid = self._evidence_map(model, x, target, attributors)
            
            # 3) Fuse
            alpha = self.ALPHA_BASE
            fused = self._to01(alpha * gcam + (1.0 - alpha) * evid)
            
            # 4) Connectivity growth
            seeds = self._topk_mask01(gcam, self.SEED_PCT)
            budget = int(self.BUDGET_PCT * H * W)
            grown = self._grow_connected(seeds, evid, budget, self.K_COMPONENTS)
            
            # 5) Final heatmap
            heat = self._to01(fused * (grown * self.MASK_WEIGHT + self.MASK_BASE))
            
            # 6) Fallback: if evidence too weak, use pure Grad-CAM
            if evid.max() < self.EVIDENCE_THRESH:
                pure_grown = self._grow_connected(seeds, gcam, budget, self.K_COMPONENTS)
                heat = self._to01(gcam * (pure_grown * self.MASK_WEIGHT + self.MASK_BASE))
            
            # Extract single channel heatmap (1, 1, H, W) -> (H, W)
            heat_2d = heat[0, 0]
            results.append(heat_2d)
        
        # Stack results: (B, H, W) and normalize using base class method
        heatmaps = torch.stack(results)
        return self._normalize_attribution(heatmaps.unsqueeze(1)).squeeze(1)

