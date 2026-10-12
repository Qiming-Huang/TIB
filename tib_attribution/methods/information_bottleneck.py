from dataclasses import dataclass, field
import math

import torch
from torch.nn import functional as F

from ..models.bottleneck import PredictiveVarianceBottleneck


@dataclass
class BranchAttribution:
    scores: torch.Tensor
    mask: torch.Tensor | None = None
    variance: float | None = None
    history: list[dict[str, float]] = field(default_factory=list)


def repeat_inputs(inputs, count):
    return {name: value.repeat(count, *([1] * (value.ndim - 1))) for name, value in inputs.items()}


def fitting_score(prediction, target, mode):
    if mode == "cosine":
        # On unit embeddings, cosine - 1 == -0.5 * squared distance (Appendix E.4).
        # Positive raw cosine would make the heteroscedastic loss unbounded as variance -> 0.
        return F.cosine_similarity(prediction, target, dim=-1, eps=1e-6) - 1
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target, dim=-1)
    return -0.5 * (prediction - target).square().sum(-1)


def tempered_loss(fitting, log_variance, compression, *, target_dim, beta, alpha):
    """Eq. (6): stop-gradient variance weight applies only to the fitting bracket."""
    weight = (alpha * log_variance).exp().detach()
    likelihood = 0.5 * target_dim * log_variance - fitting * (-log_variance).exp()
    return (weight * likelihood).mean() + beta * compression


def mask_learning_rate(config, step):
    """Decay the mask's Adam learning rate; keep variance optimization separate."""
    if config.mask_lr_schedule == "constant" or config.steps == 1:
        return config.lr
    progress = step / (config.steps - 1)
    remaining = (1 + math.cos(math.pi * progress)) / 2 if config.mask_lr_schedule == "cosine" else 1 - progress
    return config.lr * (config.mask_lr_final_ratio + (1 - config.mask_lr_final_ratio) * remaining)


def bottleneck_attribution(backbone, inputs, modality, target, config):
    features, _ = backbone.capture(modality, inputs, config.target_layer, position=config.layer_position)
    valid = backbone.valid_text_tokens(inputs) if modality == "text" else None
    bottleneck = PredictiveVarianceBottleneck(features, config)
    custom_mask_adam = (config.mask_adam_beta1, config.mask_adam_beta2, config.mask_adam_eps) != (0.9, 0.999, 1e-8)
    if config.variance_lr is not None or config.mask_lr_schedule != "constant" or custom_mask_adam:
        optimizer = torch.optim.Adam([
            {"params": [bottleneck.mask_logits], "lr": config.lr,
             "betas": (config.mask_adam_beta1, config.mask_adam_beta2), "eps": config.mask_adam_eps},
            {"params": bottleneck.variance_predictor.parameters(), "lr": config.variance_lr if config.variance_lr is not None else config.lr},
        ])
    else:
        optimizer = torch.optim.Adam(bottleneck.parameters(), lr=config.lr)
    repeated = repeat_inputs(inputs, config.num_samples)
    history = []

    for step in range(config.steps):
        optimizer.param_groups[0]["lr"] = mask_learning_rate(config, step)
        optimizer.zero_grad(set_to_none=True)
        samples, _, log_variance, capacity = bottleneck(features, config.num_samples, valid)
        prediction = backbone.with_hidden(modality, repeated, config.target_layer, samples, position=config.layer_position)
        fitting = fitting_score(prediction, target.detach(), config.fitting).mean()
        compression = capacity.mean() if valid is None else capacity.sum() / (valid.sum() * features.shape[-1])
        loss = tempered_loss(fitting, log_variance, compression, target_dim=target.shape[-1], beta=config.beta, alpha=config.alpha)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {modality} bottleneck loss at step {step + 1}")
        loss.backward()
        gradients = [parameter.grad for parameter in bottleneck.parameters() if parameter.grad is not None]
        if any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError(f"Non-finite {modality} bottleneck gradient at step {step + 1}")
        optimizer.step()
        history.append({
            "step": step + 1, "loss": loss.detach().item(), "compression": compression.detach().item(),
            "fitting": fitting.detach().item(), "variance": log_variance.detach().exp().item(),
            "mask_lr": optimizer.param_groups[0]["lr"],
        })

    with torch.no_grad():
        _, _, mask, log_variance, capacity = bottleneck.distribution(features, valid)
    # Preserve the original capacity readout; no min-max scaling here.
    return BranchAttribution(
        scores=capacity.sum(-1)[0].cpu(), mask=mask[0].cpu(),
        variance=log_variance.exp().item(), history=history,
    )
