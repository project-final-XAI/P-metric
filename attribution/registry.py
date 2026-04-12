"""
Central registry for all attribution methods.

Provides unified access to all XAI methods with their configurations.
Methods are organized into three categories:
  1. Model-dependent   — need the classifier model (gradients / activations)
  2. Model-independent — ignore the classifier model (DINO, U2Net, fusion)
  3. Continuous         — wrappers that smooth any base method
"""

# --- Category 1: Model-dependent -------------------------------------------
from attribution.model_dependent.gradient_based import (
    SaliencyMethod,
    InputXGradientMethod,
    SmoothGradMethod,
)
from attribution.model_dependent.integration_based import (
    IntegratedGradientsMethod,
    GradientSHAPMethod,
)
from attribution.model_dependent.cam_based import (
    GradCAMMethod,
    GuidedGradCAMMethod,
)
from attribution.model_dependent.perturbation_based import (
    OcclusionMethod,
    XRAIMethod,
)
from attribution.model_dependent.other import (
    GuidedBackpropMethod,
    RandomBaselineMethod,
)
from attribution.model_dependent.c3f import C3FMethod

# --- Category 2: Model-independent -----------------------------------------
from attribution.model_independent.dinov2_methods import (
    Dinov2AttnMethod,
    Dinov2Pc1Method,
    Dinov2PcEigenweightedMethod,
    Dinov2PcL2Method,
    Dinov2ComboFixedMethod,
    Dinov2ComboEntropyMethod,
    Dinov2ComboEntSmoothMethod,
)
from attribution.model_independent.unet_based import U2NetSaliencyMethod
from attribution.model_independent.unet_dino import U2NetDinoFusionMethod

# --- Category 3: Continuous wrappers ---------------------------------------
from attribution.continuous import ContinuousWrapper


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    # --- Model-dependent ---------------------------------------------------
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

    # --- Model-independent -------------------------------------------------
    "dinov2_attention": Dinov2AttnMethod(),
    "dinov2_PC1": Dinov2Pc1Method(),
    "dinov2_PC_EV": Dinov2PcEigenweightedMethod(),
    "dinov2_PC_L2": Dinov2PcL2Method(),
    "dinov2_COMBO_FIXED": Dinov2ComboFixedMethod(),
    "dinov2_ENT": Dinov2ComboEntropyMethod(),
    "dinov2_COMBO_ENT_SMOOTH": Dinov2ComboEntSmoothMethod(),
    "U2Net-Saliency": U2NetSaliencyMethod(),
    "u2net_dino_fusion": U2NetDinoFusionMethod(),

    # --- Continuous wrappers -----------------------------------------------
    "integrated_gradients_continuous": ContinuousWrapper(IntegratedGradientsMethod(), sigma=2.0),
    "gradientshap_continuous": ContinuousWrapper(GradientSHAPMethod(), sigma=2.0),
}


def get_attribution_method(name: str):
    """Get attribution method by name."""
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown attribution method: {name}")
    return METHOD_REGISTRY[name]


def get_all_methods():
    """Get all available method names."""
    return list(METHOD_REGISTRY.keys())
