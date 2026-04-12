"""
Model-independent attribution methods.

These methods compute heatmaps without needing the classifier model.
"""

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
