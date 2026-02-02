"""
Central registry for all attribution methods.

Provides unified access to all XAI methods with their configurations.
"""

# Import all methods
from attribution.gradient_based import SaliencyMethod, InputXGradientMethod, SmoothGradMethod
from attribution.integration_based import IntegratedGradientsMethod, GradientSHAPMethod
from attribution.cam_based import GradCAMMethod, GuidedGradCAMMethod
from attribution.perturbation_based import OcclusionMethod, XRAIMethod
from attribution.other import GuidedBackpropMethod, RandomBaselineMethod
from attribution.c3f import C3FMethod
from attribution.dinov2_methods import Dinov2PcaGaussianMethod, Dinov2AttentionMethod


# Registry of all available methods
METHOD_REGISTRY = {
    "saliency": SaliencyMethod(),
    "inputxgradient": InputXGradientMethod(),
    "smoothgrad": SmoothGradMethod(),
    "guided_backprop": GuidedBackpropMethod(),
    "integrated_gradients": IntegratedGradientsMethod(),
    "gradientshap": GradientSHAPMethod(),
    "occlusion": OcclusionMethod(),
    "xrai": XRAIMethod(),
    "grad_cam": GradCAMMethod(),
    "guided_gradcam": GuidedGradCAMMethod(),
    "random_baseline": RandomBaselineMethod(),
    "c3f": C3FMethod(),
    # DINOv2-based custom methods
    "dinov2_pca_gaussian": Dinov2PcaGaussianMethod(),
    "dinov2_attention": Dinov2AttentionMethod(),
}


def get_attribution_method(name: str):
    """Get attribution method by name."""
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown attribution method: {name}")
    return METHOD_REGISTRY[name]


def get_all_methods():
    """Get all available method names."""
    return list(METHOD_REGISTRY.keys())
