"""Deterministic pseudo events with the same structure as the paper's two processes.

These are not physics simulations. They reproduce what the algorithm relies
on: two invisible legs per event that the observed inputs only partly
constrain, a missing transverse momentum built from the invisibles, and MC
truth that the physics vote can compare against.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .types import EventSet, make_generator


def _wrap(phi: Tensor) -> Tensor:
    return torch.remainder(phi + math.pi, 2 * math.pi) - math.pi


def _angles(direction: Tensor) -> tuple[Tensor, Tensor]:
    theta = torch.atan2(torch.linalg.vector_norm(direction[..., :2], dim=-1), direction[..., 2])
    return theta, torch.atan2(direction[..., 1], direction[..., 0])


def _unit(theta: Tensor, phi: Tensor) -> Tensor:
    return torch.stack((theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()), -1)


TAU_ENERGY_GEV, TAU_MASS_GEV = 45.6, 1.777


def pseudo_tau(n: int, seed: int, first_id: int = 0) -> EventSet:
    """Pseudo Z->tautau: nearly back-to-back taus, each with a visible and an invisible part.

    Target: offsets (d_theta, d_phi) from each visible direction to its MC tau direction.
    observed["condition"]: both visible momenta and the missing transverse momentum (8 features).
    extra: ``visible`` (N, 2, 3) and ``truth_p4`` (N, 2, 4), as ``examples/tau.py`` expects.
    """
    g = make_generator(seed)
    rand = lambda *s: torch.rand(*s, generator=g, dtype=torch.float64)
    randn = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
    # MC taus: leg a isotropic within |cos theta| < 0.9; leg b opposite up to a small boost.
    theta_a, phi_a = torch.arccos(1.8 * rand(n) - 0.9), 2 * math.pi * rand(n) - math.pi
    tau_a = _unit(theta_a, phi_a)
    tau_b = -tau_a + 0.08 * randn(n, 3)
    tau_b = tau_b / torch.linalg.vector_norm(tau_b, dim=-1, keepdim=True)
    tau = torch.stack((tau_a, tau_b), 1)  # (n, 2, 3)
    energy = TAU_ENERGY_GEV * (1 + 0.04 * randn(n, 2))
    momentum = (energy.square() - TAU_MASS_GEV ** 2).sqrt()
    mc_tau_p4 = torch.cat((energy[..., None], momentum[..., None] * tau), -1)
    # Visible decay products: a momentum fraction z along a direction smeared away from the tau.
    z = 0.2 + 0.7 * rand(n, 2)
    spread = 0.6 * TAU_MASS_GEV / (z * momentum)
    theta_tau, phi_tau = _angles(tau)
    theta_vis = (theta_tau + spread * randn(n, 2)).clamp(0.05, math.pi - 0.05)
    phi_vis = _wrap(phi_tau + spread * randn(n, 2) / theta_tau.sin().clamp_min(0.2))
    visible = (z * momentum)[..., None] * _unit(theta_vis, phi_vis)
    offsets = torch.stack((theta_tau - theta_vis, _wrap(phi_tau - phi_vis)), -1)
    invisible = mc_tau_p4[..., 1:] - visible
    met = invisible[..., :2].sum(1) + 2.0 * randn(n, 2)
    condition = torch.cat((visible.flatten(1), met), 1)
    return EventSet(observed={"condition": condition.float()}, target=offsets.float(),
                    event_id=torch.arange(n, dtype=torch.int64) + first_id, event_weight=torch.ones(n),
                    extra={"visible": visible.float(), "truth_p4": mc_tau_p4.float()})


def pseudo_ttbar(n: int, seed: int, first_id: int = 0) -> EventSet:
    """Pseudo dileptonic ttbar: two (b + lepton) visible systems, each with one neutrino.

    Target: native neutrino coordinates [log1p(pT/GeV), eta, phi].
    observed["condition"]: per visible system [log pT, eta, sin phi, cos phi, log E] and MET (12 features).
    extra: ``visible`` (N, 2, 4), as ``examples/ttbar.py`` expects.
    """
    g = make_generator(seed)
    rand = lambda *s: torch.rand(*s, generator=g, dtype=torch.float64)
    randn = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
    pt_vis = 40 + 60 * -torch.log(rand(n, 2))
    eta_vis = 1.2 * randn(n, 2)
    phi_vis = 2 * math.pi * rand(n, 2) - math.pi
    mass_vis = 30 + 60 * rand(n, 2)
    p_vis = torch.stack((pt_vis * phi_vis.cos(), pt_vis * phi_vis.sin(), pt_vis * eta_vis.sinh()), -1)
    e_vis = (p_vis.square().sum(-1) + mass_vis.square()).sqrt()
    visible = torch.cat((e_vis[..., None], p_vis), -1)
    pt_nu = pt_vis * torch.exp(-0.3 + 0.5 * randn(n, 2))
    eta_nu = eta_vis + 0.6 * randn(n, 2)
    phi_nu = _wrap(phi_vis + 0.5 * randn(n, 2))
    target = torch.stack((torch.log1p(pt_nu), eta_nu, phi_nu), -1)
    met = torch.stack(((pt_nu * phi_nu.cos()).sum(1), (pt_nu * phi_nu.sin()).sum(1)), -1) + 10.0 * randn(n, 2)
    per_leg = torch.stack((pt_vis.log(), eta_vis, phi_vis.sin(), phi_vis.cos(), e_vis.log()), -1)
    condition = torch.cat((per_leg.flatten(1), met), 1)
    return EventSet(observed={"condition": condition.float()}, target=target.float(),
                    event_id=torch.arange(n, dtype=torch.int64) + first_id, event_weight=torch.ones(n),
                    extra={"visible": visible.float()})
