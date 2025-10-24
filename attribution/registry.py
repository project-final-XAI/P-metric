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
}


def get_attribution_method(name: str):
    """Get attribution method by name."""
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown attribution method: {name}")
    return METHOD_REGISTRY[name]


def get_all_methods():
    """Get all available method names."""
    return list(METHOD_REGISTRY.keys())


def get_method_info(name: str):
    """Get method configuration info."""
    method = get_attribution_method(name)
    return {
        "name": method.name,
        "strategy": method.strategy,
        "max_batch_size": method.max_batch_size
    }
