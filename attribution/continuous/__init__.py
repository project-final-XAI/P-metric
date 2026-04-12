"""
Continuous attribution wrappers.

Wraps any existing attribution method and applies spatial smoothing
to produce spatially continuous heatmaps.
"""

from attribution.continuous.continuous_wrapper import ContinuousWrapper
from attribution.continuous.unet_underlay_fill_wrapper import U2NetUnderlayFillWrapper
