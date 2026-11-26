"""Supervised velocity-MSE training of the reference MLP policy (tests and examples only).

EveNet policies are trained with ``evenet-train`` instead; RL always starts
from a supervised policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Union

import torch
from torch import Tensor

from .diffusion import noise_and_target
from .losses import row_velocity_mse
from .models import MLPPolicy
from .types import EventSet, MLPConfig, derive_seed, make_generator


@dataclass(frozen=True)
class SupervisedMetrics:
    step: int
    values: dict[str, float]


@dataclass(frozen=True)
class SupervisedPolicy:
    state: dict[str, Tensor]


def train_supervised(policy: MLPPolicy, train: EventSet, cfg: MLPConfig,
                     seed: int) -> Iterator[Union[SupervisedMetrics, SupervisedPolicy]]:
    generator = make_generator(derive_seed(seed, "pretrain"), train.target.device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    target = policy.normalize(train.target)
    device = generator.device
    for step in range(1, cfg.supervised_steps + 1):
        index = torch.randint(len(train), (cfg.batch_size,), generator=generator, device=device)
        x0 = target[index]
        t = torch.rand(len(index), generator=generator, device=device)
        eps = torch.randn(x0.shape, generator=generator, device=device)
        x_t, v_target = noise_and_target(x0, t, eps)
        observed = {key: value[index] for key, value in train.observed.items()}
        loss = row_velocity_mse(policy.velocity(x_t, observed, t), v_target).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        yield SupervisedMetrics(step, {"loss": float(loss.detach())})
    yield SupervisedPolicy({k: v.detach().clone() for k, v in policy.state_dict().items()})
