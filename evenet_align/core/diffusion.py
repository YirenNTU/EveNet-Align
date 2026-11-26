"""Cosine variance-preserving schedule, velocity parameterization and DDIM sampling.

Matches the EveNet diffusion utilities used in the paper:
``logSNR(t) = -2 log tan(a t + b)`` with logSNR in [-20, 20],
``alpha = sqrt(sigmoid(logSNR))``, ``sigma = sqrt(sigmoid(-logSNR))``,
``x_t = alpha x_0 + sigma eps`` and velocity target ``v = alpha eps - sigma x_0``.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor

LOGSNR_MIN = -20.0
LOGSNR_MAX = 20.0

VelocityFn = Callable[[Tensor, Tensor], Tensor]
"""Maps (x_t of shape (N, D), t of shape (N,)) to a velocity of shape (N, D)."""


def logsnr_cosine(t: Tensor) -> Tensor:
    b = math.atan(math.exp(-0.5 * LOGSNR_MAX))
    a = math.atan(math.exp(-0.5 * LOGSNR_MIN)) - b
    return -2.0 * torch.log(torch.tan(a * t + b))


def alpha_sigma(t: Tensor) -> tuple[Tensor, Tensor]:
    """(alpha, sigma) for diffusion time ``t`` in [0, 1], same shape as ``t``."""
    logsnr = logsnr_cosine(t)
    return torch.sigmoid(logsnr).sqrt(), torch.sigmoid(-logsnr).sqrt()


def noise_and_target(x0: Tensor, t: Tensor, eps: Tensor) -> tuple[Tensor, Tensor]:
    """Noised input ``x_t`` and velocity target for clean ``x0`` of shape (N, D), ``t`` (N,)."""
    alpha, sigma = alpha_sigma(t)
    alpha, sigma = alpha[:, None], sigma[:, None]
    return alpha * x0 + sigma * eps, alpha * eps - sigma * x0


@torch.no_grad()
def ddim_sample(velocity: VelocityFn, n: int, dim: int, steps: int,
                generator: torch.Generator) -> Tensor:
    """Deterministic DDIM from the Gaussian prior, ``steps`` uniform steps from t=1 to 0.

    Uses the paper's update: ``eps = alpha v + sigma x``,
    ``x0 = (x - sigma eps) / alpha``, ``x <- alpha' x0 + sigma' eps``.
    Returns normalized samples of shape (n, dim).
    """
    device = generator.device
    x = torch.randn(n, dim, generator=generator, device=device)
    for step in range(steps, 0, -1):
        t = torch.full((n,), step / steps, device=device)
        t_prev = torch.full((n,), (step - 1) / steps, device=device)
        alpha, sigma = (s[:, None] for s in alpha_sigma(t))
        alpha_prev, sigma_prev = (s[:, None] for s in alpha_sigma(t_prev))
        v = velocity(x, t)
        eps = v * alpha + x * sigma
        x0 = (x - sigma * eps) / alpha
        x = alpha_prev * x0 + sigma_prev * eps
    return x
