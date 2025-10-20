# attribution_methods.py

import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import Saliency, InputXGradient, GuidedBackprop, IntegratedGradients, GradientShap, Occlusion
from skimage.segmentation import slic

from config import (
    device, GRADCAM_TARGET_LAYER, OCCL_WINDOW, OCCL_STRIDE,
    EG_NUM_BASELINES, EG_ALPHA_RANGE, EG_NOISE_IMG_STD, EG_SMOOTH, EG_KERNEL_SIZE, EG_SIGMA
)
from utils import denormalize, MEAN, STD, _get_target_module, get_module_by_name


# ============================================================
# CNN CAM METHODS
# ============================================================
def gradcam_mask_once(model: torch.nn.Module, x: torch.Tensor, class_idx: int) -> torch.Tensor:
    """Standard Grad-CAM calculation (CNN only)."""
    model.eval()
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        target_mod = _get_target_module(model, GRADCAM_TARGET_LAYER)
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
                raise RuntimeError("Forward hook did not capture activations.")

            G = torch.autograd.grad(score, A, retain_graph=False, create_graph=False, allow_unused=False)[0]
            w = G.mean(dim=(2, 3), keepdim=True)
            cam = (w * A).sum(dim=1, keepdim=True).clamp_min(0)

            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            return F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        finally:
            h.remove()
            torch.cuda.empty_cache()


def guided_gradcam_mask(model, x, class_idx):
    """Guided Grad-CAM = Guided Backprop (pixel-level) ⨂ Grad-CAM (region-level) — CNN only."""
    x = x.clone().detach().requires_grad_()
    gbp = GuidedBackprop(model)
    cam = gradcam_mask_once(model, x, class_idx)
    gb_attr = gbp.attribute(x, target=class_idx).detach()
    # Masking with Grad-CAM requires upsampling CAM to match GBP size
    return (gb_attr * F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)).abs().mean(dim=1,
                                                                                                            keepdim=True)


def expected_gradcam_mask(model, x_norm, class_idx,
                          num_baselines=8, alpha_steps=32,
                          alpha_min=0.0, alpha_max=1.0,
                          noise_std=0.03, smooth=True,
                          kernel_size=7, sigma=2.0):
    """Expected Grad-CAM across noisy baselines (CNN)."""
    model.eval()
    x_img = denormalize(x_norm).clamp(0, 1)  # back to image space for baseline mixing
    mean_img = MEAN.expand_as(x_img)
    cams_sum = None

    for _ in range(num_baselines):
        try:
            alpha0 = torch.empty(1, 1, 1, 1, device=x_img.device).uniform_(alpha_min, alpha_max)
            baseline_img = (alpha0 * x_img + (1 - alpha0) * mean_img).clamp(0, 1)
            baseline_img = (baseline_img + noise_std * torch.randn_like(baseline_img)).clamp(0, 1)

            grads_accum = None
            activations_accum = None

            for alpha in torch.linspace(0, 1, alpha_steps, device=x_img.device):
                x_interp_img = baseline_img + alpha * (x_img - baseline_img)
                x_interp_norm = (x_interp_img - MEAN) / STD
                x_interp_norm.requires_grad_()

                activation_holder = {}

                def forward_hook(module, input, output):
                    activation_holder["acts"] = output
                    output.requires_grad_(True)
                    return output

                # Find and hook the target layer
                hook = None
                for name, m in model.named_modules():
                    if name == GRADCAM_TARGET_LAYER:
                        hook = m.register_forward_hook(forward_hook)
                        break
                if hook is None:
                    raise ValueError(f"Target layer '{GRADCAM_TARGET_LAYER}' not found")

                try:
                    out = model(x_interp_norm)
                    score = out[0, class_idx]
                    acts = activation_holder["acts"]
                    acts.requires_grad_(True)
                    model.zero_grad()
                    grads = torch.autograd.grad(score, acts, retain_graph=True, create_graph=False)[0]

                    if grads_accum is None:
                        grads_accum = grads.clone().detach()
                        activations_accum = acts.detach().clone()
                    else:
                        grads_accum += grads.detach()
                        activations_accum += acts.detach()
                finally:
                    hook.remove()

            grads_avg = grads_accum / alpha_steps
            activations_avg = activations_accum / alpha_steps
            weights = grads_avg.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations_avg).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam_up = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)

            cams_sum = cam_up if cams_sum is None else (cams_sum + cam_up)

        except Exception as e:
            print(f"Warning: Failed baseline CAM: {e}")
            continue

    if cams_sum is None:
        return gradcam_mask_once(model, x_norm, class_idx)

    cam_avg = cams_sum / float(num_baselines)

    if smooth:
        k = torch.exp(
            -(torch.arange(kernel_size, device=x_img.device) - kernel_size // 2)[:, None] ** 2 / (2 * sigma ** 2))
        k2d = k @ k.t()
        k2d = (k2d / k2d.sum()).view(1, 1, kernel_size, kernel_size)
        cam_avg = F.conv2d(cam_avg, k2d, padding=kernel_size // 2)

    cam_norm = cam_avg - cam_avg.min()
    return cam_norm / (cam_norm.max() + 1e-8)


# ============================================================
# ViT CAM METHOD (Token-based)
# ============================================================
def vit_gradcam_token(model: torch.nn.Module, x: torch.Tensor, class_idx: int) -> torch.Tensor:
    # ViT Grad-CAM implementation based on https://arxiv.org/abs/2012.09838
    target_mod = get_module_by_name(model, GRADCAM_TARGET_LAYER)
    acts = {"imagenet": None}

    def hook_fn(mod, inp, out):
        acts["imagenet"] = out
        return out

    handle = target_mod.register_forward_hook(hook_fn)
    try:
        x = x.detach().requires_grad_(True)
        logits = model(x)
        score = logits[:, class_idx].sum()
        A = acts["imagenet"]
        if A is None:
            raise RuntimeError("Hook did not capture activations.")

        G = torch.autograd.grad(score, A, retain_graph=False, create_graph=False)[0]
        w = G.mean(dim=1, keepdim=True)
        token_importance = (A * w).sum(dim=-1)
        token_importance = F.relu(token_importance)

        # Drop CLS token, reshape to patch grid
        B, T = token_importance.shape
        patch_tokens = token_importance[:, 1:]
        ps = model.conv_proj.kernel_size[0]
        H = W = 224 // ps
        heat = patch_tokens.reshape(B, 1, H, W)
        heat = (heat - heat.min()) / (heat.max() + 1e-8)
        return F.interpolate(heat, size=(224, 224), mode="bilinear", align_corners=False)
    finally:
        handle.remove()


# ============================================================
# CAPTUM & COMMON GRAD-BASED METHODS (CNN + ViT compatible)
# ============================================================
def saliency_mask(model, x, class_idx):
    """Raw input-gradient saliency |∂y_class/∂x|, channel-averaged to (1,1,H,W)."""
    x = x.clone().detach().requires_grad_()
    return Saliency(model).attribute(x, target=class_idx).abs().mean(dim=1, keepdim=True)


def inputxgradient_mask(model, x, class_idx):
    """Input × Gradient (elementwise), |x · ∂y/∂x|, channel-averaged."""
    x = x.clone().detach().requires_grad_()
    return InputXGradient(model).attribute(x, target=class_idx).abs().mean(dim=1, keepdim=True)


def guided_backprop_mask(model, x, class_idx):
    """Guided Backpropagation |∂y/∂x| with ReLU-only positive backprop rules."""
    x = x.clone().detach().requires_grad_()
    return GuidedBackprop(model).attribute(x, target=class_idx).abs().mean(dim=1, keepdim=True)


def smoothgrad_mask(model, x, class_idx, n_samples=8, noise_level=0.1):
    """SmoothGrad: average saliency over noisy inputs to reduce visual noise."""
    base = x.clone().detach()
    sal = Saliency(model)
    acc = torch.zeros_like(base)
    for _ in range(n_samples):
        noisy = (base + noise_level * torch.randn_like(base)).detach().requires_grad_(True)
        acc += sal.attribute(noisy, target=class_idx).abs().detach()
    return (acc / n_samples).mean(dim=1, keepdim=True)


def integrated_gradients_mask(model, x, class_idx, steps=50):
    """Integrated Gradients from zero baseline; absolute value + channel-average."""
    x = x.clone().detach().requires_grad_()
    baseline = torch.zeros_like(x)
    return IntegratedGradients(model).attribute(x, baselines=baseline, target=class_idx, n_steps=steps).abs().mean(
        dim=1, keepdim=True)


def gradientshap_mask(model, x, class_idx, n_samples=64):
    """GradientShap: stochastic integration between zero and noisy baselines."""
    x = x.clone().detach().requires_grad_()
    baseline_dist = torch.cat([torch.zeros_like(x), torch.randn_like(x) * 0.1], dim=0)
    return GradientShap(model).attribute(x, baselines=baseline_dist, target=class_idx, n_samples=n_samples).abs().mean(
        dim=1, keepdim=True)


def xrai_mask(model, x, class_idx):
    """XRAI (approx): segment-wise aggregation of IG to create region attributions."""
    ig = IntegratedGradients(model)
    xg = x.clone().detach().requires_grad_()
    # Amp disabled to keep IG numerics stable/consistent
    with torch.cuda.amp.autocast(enabled=False):
        attr = ig.attribute(xg, baselines=torch.zeros_like(xg), target=class_idx, n_steps=32)
    attr = attr.detach().abs().mean(dim=1).squeeze(0).cpu().numpy()  # (H,W)

    # Segment the *denormalized* image for region stats
    img_denorm = denormalize(x.detach()).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    segments = slic((img_denorm * 255).astype(np.uint8), n_segments=50, compactness=10)

    seg_heat = np.zeros_like(attr)
    for seg_id in np.unique(segments):
        seg_heat[segments == seg_id] = attr[segments == seg_id].mean()

    seg_heat = (seg_heat - seg_heat.min()) / (seg_heat.max() - seg_heat.min() + 1e-12)
    return torch.tensor(seg_heat, device=device).unsqueeze(0).unsqueeze(0)


# ============================================================
# OCCLUSION & BASELINES
# ============================================================
def occlusion_mask(model, img_tensor, class_idx):
    """Captum Occlusion: slide a window and measure output sensitivity."""
    input_ = img_tensor.clone().detach()
    occlusion = Occlusion(model)
    attr = occlusion.attribute(
        input_,
        strides=(1, OCCL_STRIDE, OCCL_STRIDE),
        sliding_window_shapes=(3, OCCL_WINDOW, OCCL_WINDOW),
        target=class_idx,
        baselines=torch.zeros_like(input_),
    )
    return attr.abs().mean(dim=1, keepdim=True)


def naive_occ_mask(img_tensor):
    """Simple analytic radial map (not class-specific) used as a toy baseline."""
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    _, _, h, w = img_tensor.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=img_tensor.device),
        torch.linspace(-1, 1, w, device=img_tensor.device),
        indexing='ij'
    )
    sigma = 0.5
    heat = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-12)
    return heat.unsqueeze(0).unsqueeze(0)


def random_baseline_mask(img_tensor):
    """Random importance map ∈ [0,1]."""
    _, _, h, w = img_tensor.shape
    return torch.rand(1, 1, h, w, device=img_tensor.device)
