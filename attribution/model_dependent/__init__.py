"""
Model-dependent attribution methods.

These methods require the classifier model to compute gradients or activations.
"""

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
