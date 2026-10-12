from dataclasses import dataclass
import math
import json
from pathlib import Path


@dataclass(frozen=True)
class AttributionConfig:
    """TIB defaults from Appendix C.1; target_layer is a zero-based input boundary."""

    target_layer: int = 9
    layer_position: str = "input"
    steps: int = 10
    num_samples: int = 5
    lr: float = 0.01
    variance_lr: float | None = None
    mask_lr_schedule: str = "constant"
    mask_lr_final_ratio: float = 0.1
    mask_adam_beta1: float = 0.9
    mask_adam_beta2: float = 0.999
    mask_adam_eps: float = 1e-8
    beta: float = 0.1
    alpha: float = 0.5
    mask_init: float = 0.0
    variance_hidden_dim: int = 128
    min_log_variance: float = -8.0
    max_log_variance: float = 8.0
    fitting: str = "cosine"
    seed: int = 42

    def __post_init__(self):
        for name in ("target_layer", "steps", "num_samples", "variance_hidden_dim", "seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if self.target_layer < 0 or self.seed < 0:
            raise ValueError("target_layer and seed must be non-negative")
        if min(self.steps, self.num_samples, self.variance_hidden_dim) < 1:
            raise ValueError("steps, num_samples and variance_hidden_dim must be positive")
        for name in ("lr", "beta", "alpha", "mask_init", "min_log_variance", "max_log_variance"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.lr <= 0 or self.beta < 0 or not 0 <= self.alpha <= 1:
            raise ValueError("Require lr > 0, beta >= 0 and alpha in [0, 1]")
        if self.variance_lr is not None and (not math.isfinite(self.variance_lr) or self.variance_lr <= 0):
            raise ValueError("variance_lr must be positive and finite, or None to use lr")
        if self.mask_lr_schedule not in {"constant", "linear", "cosine"}:
            raise ValueError("mask_lr_schedule must be 'constant', 'linear' or 'cosine'")
        if not math.isfinite(self.mask_lr_final_ratio) or not 0 <= self.mask_lr_final_ratio <= 1:
            raise ValueError("mask_lr_final_ratio must be finite and in [0, 1]")
        for name in ("mask_adam_beta1", "mask_adam_beta2"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be finite and in [0, 1)")
        if not math.isfinite(self.mask_adam_eps) or self.mask_adam_eps <= 0:
            raise ValueError("mask_adam_eps must be positive and finite")
        if not -30 <= self.min_log_variance < self.max_log_variance <= 30:
            raise ValueError("Require -30 <= min_log_variance < max_log_variance <= 30")
        if self.fitting not in {"cosine", "gaussian"}:
            raise ValueError("fitting must be 'cosine' or 'gaussian'")
        if self.layer_position not in {"input", "output"}:
            raise ValueError("layer_position must be 'input' or 'output'")


def load_branch_configs(path, *, model=None):
    """Load a model-specific TIB profile with independent modality settings."""
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(profile, dict) or profile.get("schema_version") != 1:
        raise ValueError("TIB profile requires schema_version=1")
    if model is not None and profile.get("model") != model:
        raise ValueError(f"TIB profile model {profile.get('model')!r} differs from {model!r}")
    branches = profile.get("branch_configs")
    if not isinstance(branches, dict) or set(branches) != {"vision", "text"}:
        raise ValueError("TIB profile requires exactly vision and text branch_configs")
    try:
        configs = {name: AttributionConfig(**values) for name, values in branches.items()}
    except TypeError as error:
        raise ValueError(f"Invalid TIB profile configuration: {error}") from error
    return configs
