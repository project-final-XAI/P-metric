# https://www.kaggle.com/code/niladrishekharray/sipakmed-resnet
# import os
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from captum.attr import Saliency, InputXGradient, GuidedBackprop, IntegratedGradients, GradientShap
# from torchcam.methods import GradCAM
from skimage.segmentation import slic
from torchvision import models, transforms

torch.backends.cudnn.benchmark = True  # small perf win on fixed input size

# ============================================================
# CONFIGURATION
# ============================================================
# Masking mode for Stage 2 occlusion implementation:
#  - "fixed":         replace masked pixels with DEFAULT_MASK_COLOR (after normalizing)
#  - "imagenet_mean": replace masked pixels with the normalized ImageNet mean (neutral baseline)
#  - "random":        replace masked pixels with random noise (strong baseline)
DEFAULT_MASK_COLOR = (1, 1., 1)  # grey
MASK_MODE = "random"  # good default for faithfulness comparisons

# -------------------- MODEL SELECTION --------------------
from torchvision.models import (
    ResNet18_Weights, ResNet50_Weights, MobileNet_V2_Weights,
    ViT_B_16_Weights, vit_b_16
)

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
        # gradcam_layer key is kept for API parity but unused for ViT here
        "gradcam_layer": "encoder.layers.encoder_layer_11.ln_1",
        "model_type": "vit",
    },
}

# Choose which model to run: "resnet18", "resnet50", "mobilenet_v2", or "vit_b_16"
MODEL_NAME = "mobilenet_v2"

CNN_MODEL_FUNC = MODEL_CONFIG[MODEL_NAME]["model_func"]
CNN_WEIGHTS = MODEL_CONFIG[MODEL_NAME]["weights"]
GRADCAM_TARGET_LAYER = MODEL_CONFIG[MODEL_NAME]["gradcam_layer"]
MODEL_TYPE = MODEL_CONFIG[MODEL_NAME]["model_type"]  # "cnn" or "vit"

# Occlusion hyperparameters (Captum Occlusion for Stage 1)
OCCL_WINDOW = 24
OCCL_STRIDE = 12
OCCL_BATCH_SIZE = 16  # not directly used by Captum here; kept for reference

# Expected Grad-CAM (averaging Grad-CAM across noisy baselines) — CNN only
EG_NUM_BASELINES = 32
EG_ALPHA_RANGE = (0.30, 1.00)
EG_NOISE_IMG_STD = 0.03
EG_SMOOTH = True
EG_KERNEL_SIZE = 7
EG_SIGMA = 2.0

# ============================================================
# DEVICE & NORMALIZATION
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization constants
MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


def get_mask_color_tensor(mask_color):
    """Convert an RGB mask color into normalized tensor space so compositing is consistent."""
    color = torch.tensor(mask_color, device=device).view(1, 3, 1, 1)
    return (color - MEAN) / STD


MASK_COLOR_TENSOR = get_mask_color_tensor(DEFAULT_MASK_COLOR)
IMAGENET_MEAN_TENSOR = (MEAN - MEAN) / STD  # zeros in normalized space

# Standard ImageNet preprocessing for 224×224 models
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def denormalize(t: torch.Tensor) -> torch.Tensor:
    """Undo normalization back to [0,1] space (clipping happens at save)."""
    return t * STD + MEAN


# ============================================================
# UTILS
# ============================================================
def get_module_by_name(model: torch.nn.Module, dotted: str):
    """Resolve a nested module by its dotted path (e.g., 'encoder.layers.encoder_layer_11.ln_1')."""
    cur = model
    for p in dotted.split('.'):
        if not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur


# ============================================================
# VISUALIZATION HELPERS
# ============================================================
def normalize_attribution_for_display(attr_tensor: torch.Tensor, percentile_clip: float = 1.0) -> torch.Tensor:
    """
    Normalize an attribution map for *display only* (does not affect saved .pt values).
    - Takes abs() so both positive/negative contributions show as intensity.
    - Clips extreme percentiles to improve contrast for flatter methods (e.g., IG).
    - Returns (H, W) tensor in [0,1].
    """
    a = attr_tensor.detach().float().cpu()
    if a.ndim == 4:
        a = a.squeeze(0).squeeze(0)
    elif a.ndim == 3:
        a = a.squeeze(0)

    a = a.abs()
    if 0 < percentile_clip < 50:
        low = torch.quantile(a, percentile_clip / 100.0)
        high = torch.quantile(a, 1.0 - percentile_clip / 100.0)
        a = a.clamp(low, high)

    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    return a  # (H, W) in [0,1]


def save_tensor_img(tensor: torch.Tensor, path: str, pct_removed=None):
    """Save a denormalized tensor as a JPG. Skips if file exists to save time."""
    if os.path.exists(path):
        return
    img = denormalize(tensor.detach()).squeeze(0).clamp(0, 1)
    img_np = img.permute(1, 2, 0).cpu().numpy()
    Image.fromarray((img_np * 255).astype(np.uint8)).save(path, quality=95)


def save_heatmap_jpg(mask_tensor: torch.Tensor, path: str, original_image: Image.Image = None):
    """
    Save a colored heatmap (.jpg). If original_image provided, also save an overlay image.
    Uses percentile-based normalization for nicer contrast across methods.
    """
    heatmap_norm = normalize_attribution_for_display(mask_tensor, percentile_clip=1.0).numpy()
    colored = cm.viridis(heatmap_norm)[:, :, :3]
    Image.fromarray((colored * 255).astype(np.uint8)).save(path, quality=95)

    if original_image is not None:
        overlay = create_overlay(original_image, colored, alpha=0.6)
        overlay_path = path.replace(".jpg", "_overlay.jpg")
        overlay.save(overlay_path, quality=95)


def create_overlay(original_image: Image.Image, colored_heatmap: np.ndarray, alpha: float = 0.6) -> Image.Image:
    """Blend heatmap with original image for visual inspection."""
    orig_resized = original_image.resize((colored_heatmap.shape[1], colored_heatmap.shape[0]))
    orig_np = np.array(orig_resized).astype(np.float32) / 255.0
    overlay_np = (1 - alpha) * orig_np + alpha * colored_heatmap
    overlay_np = np.clip(overlay_np, 0, 1)
    return Image.fromarray((overlay_np * 255).astype(np.uint8))


# ============================================================
# ATTRIBUTION METHODS (CNN + ViT-compatible non-CAM set)
# ============================================================
# All return a (1, 1, H, W) tensor.

def _get_target_module(model: torch.nn.Module, dotted: str):
    cur = model
    for p in dotted.split('.'):
        if not hasattr(cur, p):
            raise AttributeError(f"Target module '{dotted}' not found (missing '{p}')")
        cur = getattr(cur, p)
    return cur


# --- wrap grad-dependent work to be explicit about autograd being ON ---
def gradcam_mask_once(model: torch.nn.Module, x: torch.Tensor, class_idx: int) -> torch.Tensor:
    model.eval()
    with torch.enable_grad():  # <— ensure grads are ON
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
                raise RuntimeError(f"Forward hook did not capture activations at {GRADCAM_TARGET_LAYER}")

            G = torch.autograd.grad(score, A, retain_graph=False, create_graph=False, allow_unused=False)[0]
            w = G.mean(dim=(2, 3), keepdim=True)
            cam = (w * A).sum(dim=1, keepdim=True).clamp_min(0)

            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
            return cam
        finally:
            h.remove()
            torch.cuda.empty_cache()


def guided_gradcam_mask(model, x, class_idx):
    """Guided Grad-CAM = Guided Backprop (pixel-level) ⨂ Grad-CAM (region-level) — CNN only."""
    x = x.clone().detach().requires_grad_()
    gbp = GuidedBackprop(model)
    cam = gradcam_mask_once(model, x, class_idx)
    gb_attr = gbp.attribute(x, target=class_idx).detach()
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


def occlusion_mask(model, img_tensor, class_idx):
    """Captum Occlusion: slide a window and measure output sensitivity to occluding patches."""
    from captum.attr import Occlusion
    input_ = img_tensor.clone().detach()  # occlusion doesn't need grads
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


# ============================================================
# STAGE 1: HEATMAP GENERATION
# ============================================================
def generate_heatmaps(image_path: str,
                      heatmap_folder: str,
                      model: torch.nn.Module,
                      index: int = None,
                      max_visuals: int = 10) -> str:
    """
    Compute and cache all heatmaps (.pt) for one image.
    - For CNNs: includes Grad-CAM, Guided Grad-CAM, Expected Grad-CAM
    - For ViT:  CAM-like methods are intentionally disabled; only non-CAM methods run
    - Saves pretty JPG overlays only for the first `max_visuals` images (to save time).
    """
    os.makedirs(heatmap_folder, exist_ok=True)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # ---- Choose which methods to run (depends on backbone type) ----
    cam_methods_cnn = ["gradcam", "guided_gradcam", "expected_gradcam"]
    cam_methods_vit = []  # CAM-like methods for ViT removed by request
    common_methods = [
        "occlusion", "naive_occ",
        "saliency", "inputxgradient", "guided_backprop", "smoothgrad",
        "integrated_gradients", "gradientshap", "xrai", "random_baseline",
    ]
    if MODEL_TYPE == "cnn":
        methods_list = cam_methods_cnn + common_methods
    else:  # MODEL_TYPE == "vit"
        methods_list = cam_methods_vit + common_methods

    # print(f"[Debug] methods_list for {image_name}: {methods_list}")

    # ---- Figure out which ones still need to be computed for this image ----
    methods_to_compute = []
    for method_name in methods_list:
        method_dir = os.path.join(heatmap_folder, method_name)
        heatmap_file = os.path.join(method_dir, f"{image_name}.pt")
        if not os.path.exists(heatmap_file):
            methods_to_compute.append(method_name)

    if not methods_to_compute:
        return f"SKIPPED: {image_name} (all heatmaps exist)"

    print(f"Processing {image_name}: computing {len(methods_to_compute)} methods {methods_to_compute}")

    # ---- Load & preprocess image ----
    image = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    # ---- Classify once (no-grad) to get target class ----
    with torch.no_grad():
        pred = model(img_tensor)
    class_idx = pred.argmax(dim=1).item()

    # IMPORTANT: re-enable grads for attribution methods & hooks
    img_tensor = img_tensor.detach().requires_grad_(True)

    # ---- Build method table (name → callable that returns (1,1,224,224) tensor) ----
    table = {
        # Common (CNN & ViT)
        "occlusion": lambda: occlusion_mask(model, img_tensor, class_idx),
        "naive_occ": lambda: naive_occ_mask(img_tensor),
        "saliency": lambda: saliency_mask(model, img_tensor, class_idx),
        "inputxgradient": lambda: inputxgradient_mask(model, img_tensor, class_idx),
        "guided_backprop": lambda: guided_backprop_mask(model, img_tensor, class_idx),
        "smoothgrad": lambda: smoothgrad_mask(model, img_tensor, class_idx),
        "integrated_gradients": lambda: integrated_gradients_mask(model, img_tensor, class_idx),
        "gradientshap": lambda: gradientshap_mask(model, img_tensor, class_idx),
        "xrai": lambda: xrai_mask(model, img_tensor, class_idx),
        "random_baseline": lambda: random_baseline_mask(img_tensor),
    }

    if MODEL_TYPE == "cnn":
        table.update({
            "gradcam": lambda: gradcam_mask_once(model, img_tensor, class_idx),
            "guided_gradcam": lambda: guided_gradcam_mask(model, img_tensor, class_idx),
            "expected_gradcam": lambda: expected_gradcam_mask(model, img_tensor, class_idx),
        })
    # else: no ViT CAM-like entries added

    # ---- Compute & save ----
    completed_methods = []
    for method_name in methods_to_compute:
        try:
            mask = table[method_name]()
            if mask is None:
                continue

            method_dir = os.path.join(heatmap_folder, method_name)
            os.makedirs(method_dir, exist_ok=True)

            # Save raw tensor (float32). If I/O is a bottleneck, switch to mask.half().cpu()
            torch.save(mask.cpu(), os.path.join(method_dir, f"{image_name}.pt"))

            # Save preview JPGs for the first N images only
            if index is not None and index < max_visuals:
                save_heatmap_jpg(mask, os.path.join(method_dir, f"{image_name}.jpg"), original_image=image)

            completed_methods.append(method_name)

        except Exception as e:
            print(f"Error generating heatmap for {method_name}: {e}")
        finally:
            torch.cuda.empty_cache()

    return f"COMPLETED: {image_name} - {len(completed_methods)}/{len(methods_to_compute)} methods"


# ============================================================
# STAGE 2: OCCLUSION FROM HEATMAPS
# ============================================================
def mask_multiple_thresholds(img_tensor_norm, mask, thresholds):
    """
    Given one (1,1,H,W) heatmap, produce masked images for multiple quantile thresholds.
    Pixels with attribution <= cutoff are replaced according to MASK_MODE.
    """
    if img_tensor_norm.dim() == 3:
        img_tensor_norm = img_tensor_norm.unsqueeze(0)

    m = mask.squeeze(0).squeeze(0)
    m_norm = (m - m.min()) / (m.max() - m.min() + 1e-8)

    # Cutoffs per requested quantile thresholds (e.g., [0.05, 0.10, ...])
    cutoffs = torch.quantile(m_norm.flatten(), torch.tensor(thresholds, device=m.device))

    results = {}
    for thr, cutoff in zip(thresholds, cutoffs):
        keep_mask = (m_norm > cutoff).to(dtype=img_tensor_norm.dtype).unsqueeze(0).unsqueeze(0)
        keep_mask_3c = keep_mask.expand(img_tensor_norm.shape)

        if MASK_MODE == "random":
            rand_pixels_img = torch.rand_like(img_tensor_norm)
            rand_pixels_norm = (rand_pixels_img - MEAN) / STD
            masked_img_norm = img_tensor_norm * keep_mask_3c + rand_pixels_norm * (1 - keep_mask_3c)
        elif MASK_MODE == "imagenet_mean":
            masked_img_norm = img_tensor_norm * keep_mask_3c + IMAGENET_MEAN_TENSOR.expand_as(img_tensor_norm) * (
                        1 - keep_mask_3c)
        else:  # "fixed" or any other → use DEFAULT_MASK_COLOR in normalized space
            masked_img_norm = img_tensor_norm * keep_mask_3c + MASK_COLOR_TENSOR.expand_as(img_tensor_norm) * (
                        1 - keep_mask_3c)

        pct_removed = 100 * (1 - keep_mask.sum().item() / keep_mask.numel())
        results[thr] = (masked_img_norm, pct_removed)

    return results


def generate_occlusions_from_heatmaps(image_path, heatmap_folder, output_folder, thresholds):
    """
    For a single image: load each saved heatmap .pt, apply multiple thresholds,
    and save the masked images (JPG) in method-specific subfolders.
    """
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    for method_name in os.listdir(heatmap_folder):
        method_dir = os.path.join(heatmap_folder, method_name)
        heatmap_path = os.path.join(method_dir, f"{image_name}.pt")
        if not os.path.exists(heatmap_path):
            continue

        mask = torch.load(heatmap_path, map_location=device).float()
        thr_results = mask_multiple_thresholds(img_tensor, mask, thresholds)

        for thr, (masked_tensor, pct_removed) in thr_results.items():
            out_dir = os.path.join(output_folder, method_name, f"{method_name}{int(thr * 100):02d}")
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, f"{image_name}_{method_name}{int(thr * 100):02d}.jpg")
            save_tensor_img(masked_tensor, save_path, pct_removed)


# ============== ViT Grad-CAM Implementation ==============
def vit_gradcam_token(model: torch.nn.Module, x: torch.Tensor, class_idx: int) -> torch.Tensor:
    # ViT Grad-CAM implementation based on https://arxiv.org/abs/2012.09838
    target_mod = get_module_by_name(model, GRADCAM_TARGET_LAYER)
    acts = {"val": None}

    def hook_fn(mod, inp, out):
        acts["val"] = out
        return out

    handle = target_mod.register_forward_hook(hook_fn)
    try:
        x = x.detach().requires_grad_(True)
        logits = model(x)
        score = logits[:, class_idx].sum()
        A = acts["val"]
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
# MAIN
# ============================================================
if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    torch.manual_seed(42)

    # Paths & thresholds (edit as needed)
    input_folder = r"C:\imagenet"
    heatmap_folder = r"C:\heatmaps-mobilenet_v2"
    output_folder = r"C:\Users\avi\Desktop\tmp\mobilenet_v2\random"
    thresholds = [round(i * 0.05, 2) for i in range(1, 20)]  # 5%..95%

    os.makedirs(output_folder, exist_ok=True)

    image_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print(f"[Model] NAME={MODEL_NAME} TYPE={MODEL_TYPE}  CAM_LAYER={GRADCAM_TARGET_LAYER}")
    print(f"Processing {len(image_files)} images...")

    # Load model ONCE and warmup
    model = CNN_MODEL_FUNC(weights=CNN_WEIGHTS).to(device).eval()
    with torch.no_grad():
        _ = model(torch.zeros(1, 3, 224, 224, device=device))

    # --------------------
    # Stage 1: Heatmaps
    # --------------------
    # Tip: Increase max_workers gradually (2→4→6). Too high can cause CUDA OOM or hook contention.
    max_visuals = 10  # only save visual JPGs for first N images
    skipped_count = 0
    processed_count = 0

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {
            executor.submit(generate_heatmaps, img_path, heatmap_folder, model, index=i,
                            max_visuals=max_visuals): img_path
            for i, img_path in enumerate(image_files)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing heatmaps"):
            try:
                result = future.result()
                if result and result.startswith("SKIPPED"):
                    skipped_count += 1
                    if skipped_count <= 10:
                        print(result)
                    elif skipped_count == 11:
                        print("... (showing only first 10 skipped files)")
                elif result and result.startswith("COMPLETED"):
                    processed_count += 1
                    print(result)
            except Exception as e:
                print(f"Error processing image: {e}")

    print(f"Stage 1 Summary: {processed_count} processed, {skipped_count} skipped")

    # --------------------
    # Stage 2: Occlusions
    # --------------------
    # Mostly CPU-bound; you can use more workers here if disk I/O keeps up.
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(generate_occlusions_from_heatmaps, img_path, heatmap_folder, output_folder,
                            thresholds): img_path
            for img_path in image_files
        }
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Occlusions created"):
            pass

    print("Stage 2: All heatmaps and occlusion images have been generated.")
