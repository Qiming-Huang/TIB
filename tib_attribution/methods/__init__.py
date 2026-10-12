from .information_bottleneck import BranchAttribution, bottleneck_attribution

METHODS = ("tib",)

__all__ = ["METHODS", "BranchAttribution", "bottleneck_attribution"]
