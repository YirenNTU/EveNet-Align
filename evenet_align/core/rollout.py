"""Drawing candidates from a policy: K DDIM samples per event, in normalized or physical units."""
from __future__ import annotations

import torch
from torch import Tensor

from .diffusion import ddim_sample
from .types import EventSet, Policy

SAMPLING_CHUNK = 8192  # rows per DDIM call; bounds memory only, does not change results


def repeat_rows(x: Tensor, k: int) -> Tensor:
    """(B, ...) -> (K * B, ...) ordered candidate-major: [cand0 events..., cand1 events..., ...]."""
    return x[None].expand(k, *x.shape).reshape(k * x.shape[0], *x.shape[1:])


def repeat_observed(observed: dict[str, Tensor], k: int) -> dict[str, Tensor]:
    return {key: repeat_rows(value, k) for key, value in observed.items()}


def sample_normalized(policy: Policy, events: EventSet, k: int, steps: int, generator: torch.Generator) -> Tensor:
    """K independent DDIM draws per event, normalized, shape (K, B, D)."""
    observed = repeat_observed(events.observed, k)
    rows, dim = k * len(events), events.target[0].numel()
    chunks = []
    for start in range(0, rows, SAMPLING_CHUNK):
        obs = {key: value[start:start + SAMPLING_CHUNK] for key, value in observed.items()}
        n = min(SAMPLING_CHUNK, rows - start)
        chunks.append(ddim_sample(lambda x, t, obs=obs: policy.velocity(x, obs, t), n, dim, steps, generator))
    return torch.cat(chunks).reshape(k, len(events), dim)


def to_physical(normalized: Tensor, policy: Policy) -> Tensor:
    """(K, B, D) normalized -> (K, B, n, F) physical."""
    k, b, d = normalized.shape
    return policy.denormalize(normalized.reshape(k * b, d)).reshape(k, b, *policy.target_shape)
