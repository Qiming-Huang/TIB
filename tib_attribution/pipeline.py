from dataclasses import asdict, dataclass, field
from pathlib import Path
import json

import numpy as np
import torch
from torch.nn import functional as F

from .config import AttributionConfig
from .methods import METHODS, bottleneck_attribution


@dataclass
class AttributionResult:
    method: str
    image_scores: np.ndarray
    text_scores: np.ndarray
    similarity: float
    image_mask: np.ndarray | None = None
    text_mask: np.ndarray | None = None
    image_variance: float | None = None
    text_variance: float | None = None
    history: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def save(self, path):
        """Save numerical attributions and optimisation trace, without evaluation."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"image_scores": self.image_scores, "text_scores": self.text_scores}
        for name in ("image_mask", "text_mask"):
            if getattr(self, name) is not None:
                arrays[name] = getattr(self, name)
        arrays["metadata"] = np.array(json.dumps({
            "method": self.method, "similarity": self.similarity,
            "image_variance": self.image_variance, "text_variance": self.text_variance,
            "history": self.history, "config": self.config,
        }, ensure_ascii=False))
        np.savez_compressed(path, **arrays)


def image_score_map(scores, backbone, inputs):
    if backbone.has_cls_token:
        scores = scores[1:]
    rows, columns = backbone.patch_grid(inputs)
    if scores.numel() != rows * columns:
        raise ValueError(f"Expected {rows * columns} image patch scores, got {scores.numel()}")
    grid = scores.reshape(1, 1, rows, columns)
    height, width = inputs["pixel_values"].shape[-2:]
    return F.interpolate(grid, (height, width), mode="bilinear", align_corners=False)[0, 0].numpy()


def explain(backbone, inputs, method="tib", config=None, *, branch_configs=None):
    """Explain one image-caption pair. Use repeated calls for multiple captions."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    if config is None:
        config = AttributionConfig()
    if branch_configs is not None:
        if method != "tib" or set(branch_configs) != {"vision", "text"}:
            raise ValueError("branch_configs requires TIB and exactly vision/text configurations")
        if not all(isinstance(value, AttributionConfig) for value in branch_configs.values()):
            raise ValueError("branch_configs values must be AttributionConfig instances")
    required = {"pixel_values", "input_ids"}
    if not required.issubset(inputs):
        raise ValueError("inputs must contain pixel_values and input_ids")
    inputs = {key: value.to(backbone.device) for key, value in inputs.items() if key in required | {"attention_mask"}}
    inputs["pixel_values"] = inputs["pixel_values"].to(backbone.dtype)
    if inputs["pixel_values"].ndim != 4 or inputs["input_ids"].ndim != 2:
        raise ValueError("Expected pixel_values [1, C, H, W] and input_ids [1, T]")
    if inputs["pixel_values"].shape[0] != 1 or inputs["input_ids"].shape[0] != 1:
        raise ValueError("explain accepts exactly one image-caption pair per call")
    if inputs["input_ids"].shape[1] == 0:
        raise ValueError("input_ids cannot be empty")
    if "attention_mask" in inputs:
        if inputs["attention_mask"].shape != inputs["input_ids"].shape or not inputs["attention_mask"].bool().any():
            raise ValueError("attention_mask must match input_ids and contain a valid token")
    if not backbone.valid_text_tokens(inputs).any():
        raise ValueError("input_ids must contain at least one non-padding token")
    for modality in ("vision", "text"):
        effective = config if branch_configs is None else branch_configs[modality]
        backbone.layer(modality, effective.target_layer)

    cuda_devices = [backbone.device.index or 0] if backbone.device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices), backbone.frozen(), torch.enable_grad():
        torch.random.default_generator.manual_seed(config.seed)
        for device_index in cuda_devices:
            torch.cuda.default_generators[device_index].manual_seed(config.seed)
        with torch.no_grad():
            image_features = backbone.features("vision", inputs)
            text_features = backbone.features("text", inputs)
            similarity = F.cosine_similarity(image_features, text_features, dim=-1).item()
        branches = []
        for modality, target in (("vision", text_features), ("text", image_features)):
            effective = config if branch_configs is None else branch_configs[modality]
            if branch_configs is not None:
                # Independent RNG streams make text tuning independent of vision settings.
                # The default interface retains its original shared RNG sequence.
                torch.random.default_generator.manual_seed(effective.seed)
                for device_index in cuda_devices:
                    torch.cuda.default_generators[device_index].manual_seed(effective.seed)
            branches.append(bottleneck_attribution(backbone, inputs, modality, target, effective))
        vision, text = branches

    return AttributionResult(
        method=method, image_scores=image_score_map(vision.scores, backbone, inputs),
        text_scores=text.scores.numpy(), similarity=similarity,
        image_mask=None if vision.mask is None else vision.mask.numpy(),
        text_mask=None if text.mask is None else text.mask.numpy(),
        image_variance=vision.variance, text_variance=text.variance,
        history={"vision": vision.history, "text": text.history}, config={
            **asdict(config), **({"branch_configs": {name: asdict(value) for name, value in branch_configs.items()}}
                                if branch_configs is not None else {}),
        },
    )
