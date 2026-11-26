"""Hybrid within-event reward and leave-one-out advantage.

For one event with K candidates, classifier logits s and physics-vote distances d:

    winner = first argmin_i d_i
    r_i    = log( (1 - alpha) softmax(s)_i + alpha * 1[i = winner] )
    A_i    = r_i - mean_{j != i} r_j

The mixture is formed in probability space before the log. All tensors are (K, B).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def vote_winner(distance: Tensor) -> Tensor:
    """Index (B,) of the smallest distance per event; the first one on a tie."""
    return distance.argmin(0)


@torch.no_grad()
def hybrid_reward(logits: Tensor, distance: Tensor, oracle_fraction: float) -> Tensor:
    """Log of the classifier-softmax / physics-vote probability mixture, shape (K, B).

    ``oracle_fraction`` (alpha) must lie in [0, 1). Computed in float64.
    """
    log_classifier = logits.double().log_softmax(0)
    if oracle_fraction == 0.0:
        return log_classifier
    vote = torch.full_like(log_classifier, -math.inf)
    vote.scatter_(0, vote_winner(distance)[None], math.log(oracle_fraction))
    return torch.logaddexp(log_classifier + math.log1p(-oracle_fraction), vote)


def leave_one_out_advantage(reward: Tensor) -> Tensor:
    """Unscaled ``A_i = r_i - mean_{j != i} r_j`` over candidates (dim 0), shape (K, B)."""
    k = reward.shape[0]
    return reward - (reward.sum(0, keepdim=True) - reward) / (k - 1)


def effective_sample_size(log_weights: Tensor) -> Tensor:
    """Per-event ESS 1 / sum_i w_i^2 of normalized weights, shape (B,)."""
    w = log_weights.softmax(0)
    return 1.0 / w.square().sum(0)
