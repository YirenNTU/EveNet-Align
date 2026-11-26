"""Velocity-space losses: diffusion pretraining, the gated DGPO term and the reference anchor."""
from __future__ import annotations

import torch
from torch import Tensor


def row_velocity_mse(v_pred: Tensor, v_target: Tensor) -> Tensor:
    """Mean squared velocity error per row: (N, D) -> (N,)."""
    return (v_pred - v_target).square().mean(-1)


def dgpo_loss(error_current: Tensor, error_reference: Tensor, advantage: Tensor,
              beta: float) -> tuple[Tensor, dict[str, Tensor]]:
    """Advantage-weighted denoising loss with a detached event-level sigmoid gate.

    Args:
        error_current: (K, B) per-candidate velocity MSE of the policy (carries gradients).
        error_reference: (K, B) the same for the frozen round reference, on the same (t, eps).
        advantage: (K, B) leave-one-out advantages.
        beta: gate scale.

    The gate is ``w_e = sg(sigmoid((beta / K) sum_i A_ie (L_cur - L_ref)_ie))``
    and the loss is ``mean_{e, i}(w_e A_ie L_cur_ie)``. Gradients flow only
    through ``error_current``.
    """
    k = advantage.shape[0]
    delta = error_current.detach() - error_reference.detach()
    gate = torch.sigmoid((beta / k) * (advantage * delta).sum(0)).detach()
    loss = (gate[None] * advantage * error_current).mean()
    return loss, {"gate_mean": gate.mean(), "gate_min": gate.min(), "gate_max": gate.max(),
                  "error_current": error_current.detach().mean(),
                  "error_reference": error_reference.detach().mean()}


def reference_anchor_loss(v_current: Tensor, v_reference: Tensor) -> Tensor:
    """``0.5 * mean((v_cur - v_ref)^2)`` on a shared (t, eps) draw; the reference is detached."""
    return 0.5 * (v_current - v_reference.detach()).square().mean()
