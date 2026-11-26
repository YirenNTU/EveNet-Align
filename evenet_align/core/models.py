"""Networks: the small reference MLP policy (tests) and the General classifier head.

The default policy is EveNet (``core/evenet_policy.py``). ``MLPPolicy``
implements the same ``Policy`` interface on a flat condition vector so the
algorithm can be tested without EveNet.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .types import MLPConfig


class FiLMResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.context = nn.Linear(width, 2 * width)
        self.hidden = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        nn.init.zeros_(self.context.weight)
        nn.init.zeros_(self.context.bias)

    def forward(self, h: Tensor, c: Tensor) -> Tensor:
        gamma, beta = self.context(c).chunk(2, -1)
        return h + self.output(F.silu(self.hidden((1 + gamma) * self.norm(h) + beta)))


class VelocityMLP(nn.Module):
    """Conditional velocity network v(x_t, condition, t) on flattened normalized targets.

    x_t: (N, D), condition: (N, C) normalized, t: (N,) in [0, 1]. Returns (N, D).
    """

    def __init__(self, condition_dim: int, target_dim: int, cfg: MLPConfig):
        super().__init__()
        width = cfg.hidden
        self.encoder = nn.Sequential(nn.Linear(condition_dim, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU())
        self.register_buffer("frequencies", math.pi * 2.0 ** torch.arange(cfg.time_frequencies) / 8)
        self.time = nn.Sequential(nn.Linear(2 * cfg.time_frequencies, width), nn.SiLU())
        self.input = nn.Linear(target_dim, width)
        self.blocks = nn.ModuleList([FiLMResidualBlock(width) for _ in range(cfg.depth)])
        self.out = nn.Linear(width, target_dim)

    def encode(self, condition: Tensor) -> Tensor:
        return self.encoder(condition)

    def forward(self, x_t: Tensor, condition: Tensor, t: Tensor) -> Tensor:
        angle = t[:, None] * self.frequencies
        context = self.encode(condition) + self.time(torch.cat((angle.sin(), angle.cos()), -1))
        h = self.input(x_t)
        for block in self.blocks:
            h = block(h, context)
        return self.out(h)


class MLPPolicy(nn.Module):
    """``Policy`` on ``observed["condition"]`` (N, C), with standardization pinned from the training set."""

    def __init__(self, condition_dim: int, target_shape: tuple[int, int], cfg: MLPConfig,
                 stats: dict[str, Tensor]):
        super().__init__()
        self.target_shape = tuple(target_shape)
        self.net = VelocityMLP(condition_dim, target_shape[0] * target_shape[1], cfg)
        for name in ("condition_mean", "condition_std", "target_mean", "target_std"):
            self.register_buffer(name, stats[name].clone())

    def _condition(self, observed: dict[str, Tensor]) -> Tensor:
        return (observed["condition"] - self.condition_mean) / self.condition_std

    def normalize(self, target: Tensor) -> Tensor:
        return ((target - self.target_mean) / self.target_std).flatten(1)

    def denormalize(self, x: Tensor) -> Tensor:
        return x.reshape(-1, *self.target_shape) * self.target_std + self.target_mean

    def velocity(self, x_t: Tensor, observed: dict[str, Tensor], t: Tensor) -> Tensor:
        return self.net(x_t, self._condition(observed), t)

    def condition_embedding(self, observed: dict[str, Tensor]) -> Tensor:
        return self.net.encode(self._condition(observed))

    def candidate_features(self, candidates: Tensor, observed: dict[str, Tensor]) -> Tensor:
        return candidates.new_zeros(len(candidates), 0)  # no backbone features of the candidate


def mlp_statistics(events) -> dict[str, Tensor]:
    """Condition and target standardization measured on the training set."""
    condition, target = events.observed["condition"].double(), events.target.double()
    return {"condition_mean": condition.mean(0).float(), "condition_std": condition.std(0).clamp_min(1e-6).float(),
            "target_mean": target.mean(0).float(), "target_std": target.std(0).clamp_min(1e-6).float()}


def build_mlp_policy(condition_dim: int, target_shape: tuple[int, int], cfg: MLPConfig,
                     stats: dict[str, Tensor], seed: int) -> MLPPolicy:
    """MLP policy with seeded initialization (the global RNG is restored afterwards)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return MLPPolicy(condition_dim, target_shape, cfg, stats)


class ConditioningBlock(nn.Module):
    """FiLM block of the General classifier; the context projection starts at zero (identity FiLM)."""

    def __init__(self, dropout: float, context_width: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(64)
        self.context = nn.Linear(context_width, 128, bias=False)
        self.hidden = nn.Linear(64, 128)
        self.output = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.context.weight)

    def forward(self, h: Tensor, c: Tensor) -> Tensor:
        gamma, beta = self.context(c).chunk(2, -1)
        return self.output(self.dropout(F.silu(self.hidden((1 + gamma) * self.norm(h) + beta))))


class GeneralClassifier(nn.Module):
    """Head that scores (condition embedding, candidate features) with a bounded logit.

    The logit approximates log p_MC / p_policy for one candidate given its event.
    ``z = log R - softplus(log(R - 1) - z_raw)`` bounds it above by log R and
    maps z_raw = 0 to exactly 0. A fresh head has a zero readout, so all
    candidates of an event start with equal weight.
    """

    def __init__(self, condition_dim: int, candidate_dim: int, dropout: float, ratio_bound: float):
        super().__init__()
        self.condition_encoder = nn.Sequential(nn.Linear(condition_dim, 256), nn.SiLU(),
                                               nn.Linear(256, 256), nn.SiLU())
        self.process_label_dense = nn.Linear(1, 256, bias=False)
        self.spin_encoder = nn.Sequential(nn.Linear(candidate_dim, 64), nn.SiLU(),
                                          nn.Linear(64, 64), nn.SiLU())
        self.blocks = nn.ModuleList([ConditioningBlock(dropout) for _ in range(3)])
        self.readout = nn.Linear(64, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.log_bound = math.log(ratio_bound)
        self.log_bound_minus_one = math.log(ratio_bound - 1)

    def forward(self, condition: Tensor, candidate: Tensor) -> Tensor:
        """condition (N, condition_dim), candidate (N, candidate_dim) -> logits (N,)."""
        c = self.condition_encoder(condition) + self.process_label_dense.weight[:, 0]
        h = self.spin_encoder(candidate)
        for block in self.blocks:
            h = h + block(h, c) / math.sqrt(3)
        z = self.readout(h).flatten()
        return self.log_bound - F.softplus(self.log_bound_minus_one - z)


def build_classifier_head(condition_dim: int, candidate_dim: int, dropout: float,
                          ratio_bound: float, seed: int) -> GeneralClassifier:
    """Fresh seeded head with identity FiLM and a zero readout."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return GeneralClassifier(condition_dim, candidate_dim, dropout, ratio_bound)
