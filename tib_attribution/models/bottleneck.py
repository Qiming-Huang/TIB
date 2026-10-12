"""Predictive-variance bottleneck: paper Eq. (7), Eq. (8) and Figure 9."""

import torch
from torch import nn


def diagonal_gaussian_kl(mean, variance):
    """Elementwise KL[N(mean, variance) || N(0, I)]."""
    variance = variance.clamp_min(1e-8)
    return 0.5 * (mean.square() + variance - variance.log() - 1)


class SampleVariancePredictor(nn.Module):
    """Pool valid tokens, then predict one scalar log variance per input sample."""

    def __init__(self, feature_dim, hidden_dim=128, log_bounds=(-8.0, 8.0)):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        # Start at variance=1 while allowing gradients through both layers.
        nn.init.normal_(self.network[-1].weight, std=1e-3)
        nn.init.zeros_(self.network[-1].bias)
        self.log_bounds = log_bounds

    def forward(self, features, valid_tokens=None):
        if valid_tokens is None:
            pooled = features.mean(dim=1)
        else:
            weights = valid_tokens.to(features.dtype).unsqueeze(-1)
            pooled = (features * weights).sum(1) / weights.sum(1).clamp_min(1)
        raw = self.network(pooled).squeeze(-1)
        lower, upper = self.log_bounds
        midpoint, half_range = (lower + upper) / 2, (upper - lower) / 2
        # A hard clamp can permanently stop variance learning after an Adam overshoot.
        return midpoint + half_range * torch.tanh((raw - midpoint) / half_range)


class PredictiveVarianceBottleneck(nn.Module):
    def __init__(self, features, config):
        super().__init__()
        if features.ndim != 3 or features.shape[0] != 1:
            raise ValueError("Per-pair bottleneck expects features [1, tokens, channels]")
        self.mask_logits = nn.Parameter(torch.full_like(features, config.mask_init))
        self.variance_predictor = (
            SampleVariancePredictor(
                features.shape[-1], config.variance_hidden_dim,
                (config.min_log_variance, config.max_log_variance),
            ).to(device=features.device, dtype=features.dtype)
        )

    def distribution(self, features, valid_tokens=None):
        mask = self.mask_logits.sigmoid()
        if valid_tokens is not None:
            # Padding does not receive attribution or participate in compression.
            mask = torch.where(valid_tokens.unsqueeze(-1).bool(), mask, torch.ones_like(mask))
        log_variance = self.variance_predictor(features, valid_tokens)
        sigma2 = log_variance.exp()
        mean = mask * features
        posterior_variance = sigma2[:, None, None] * (1 - mask).square()
        capacity = diagonal_gaussian_kl(mean, posterior_variance)
        if valid_tokens is not None:
            capacity = capacity * valid_tokens.unsqueeze(-1)
        return mean, posterior_variance, mask, log_variance, capacity

    def forward(self, features, num_samples=1, valid_tokens=None):
        mean, variance, mask, log_variance, capacity = self.distribution(features, valid_tokens)
        noise = torch.randn((num_samples, *features.shape), device=features.device, dtype=features.dtype)
        # Avoid sqrt(0) gradients at padded positions where mask is exactly one.
        std = (0.5 * log_variance).exp()[:, None, None] * (1 - mask)
        samples = mean.unsqueeze(0) + std.unsqueeze(0) * noise
        return samples[:, 0], mask, log_variance, capacity
