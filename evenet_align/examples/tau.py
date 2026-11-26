"""Z -> tau tau oracle and classifier features used in the paper.

Targets are angular offsets [d_theta, d_phi] (rad), per tau leg, that rotate
the reconstructed visible decay direction onto the MC tau direction.

Expected ``EventSet.extra`` arrays:
    visible  (N, 2, 3)  visible tau-decay momenta [px, py, pz], GeV
    truth_p4 (N, 2, 4)  MC tau four-momenta [E, px, py, pz], GeV

Config::

    oracle:
      function: evenet_align.examples.tau:momentum_oracle
      params: {fixed_energy_GeV: 45.6, tau_mass_GeV: 1.777}
    classifier:
      features: {function: evenet_align.examples.tau:direction_features}
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from evenet_align.core.types import EventSet


def tau_directions(offsets: Tensor, visible: Tensor) -> Tensor:
    """Unit tau directions from offsets [..., 2, 2] (d_theta, d_phi) and visible momenta [..., 2, 3]."""
    theta = torch.atan2(torch.linalg.vector_norm(visible[..., :2], dim=-1), visible[..., 2])
    phi = torch.atan2(visible[..., 1], visible[..., 0])
    theta, phi = theta + offsets[..., 0], phi + offsets[..., 1]
    return torch.stack((theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()), -1)


def check_truth(events: EventSet) -> None:
    """MC taus must be physical and point where the target offsets say."""
    p4 = events.extra["truth_p4"].double()
    energy, p2 = p4[..., 0], p4[..., 1:].square().sum(-1)
    if (energy <= 0).any() or (p2 <= 0).any() or (energy.square() - p2 < -1e-5 * energy.square()).any():
        raise ValueError("truth_p4 has non-positive energy/momentum or is spacelike")
    mc_direction = p4[..., 1:] / p2.sqrt()[..., None]
    truth_direction = tau_directions(events.target.double(), events.extra["visible"].double())
    mismatch = torch.linalg.vector_norm(mc_direction - truth_direction, dim=-1).max()
    if mismatch > 1e-4:
        raise ValueError(f"MC tau directions disagree with target offsets (max {float(mismatch):.2e})")


class MomentumOracle:
    """Normalized two-leg Euclidean four-momentum error of the reconstructed candidate taus.

    Each candidate tau gets the fixed energy E and momentum p0 = sqrt(E^2 - m^2)
    along its predicted direction u. It is compared with the actual MC tau
    four-momentum component by component (a Euclidean distance in
    (E, px, py, pz), not a Minkowski interval), divided by p0^2 and averaged
    over the two legs.
    """

    def __init__(self, energy: float, p0: float):
        self.energy, self.p0 = energy, p0

    def distance(self, candidates: Tensor, events: EventSet) -> Tensor:
        u = tau_directions(candidates.double(), events.extra["visible"].double()[None])
        mc = events.extra["truth_p4"].double()[None]
        spatial = (self.p0 * u - mc[..., 1:]).square().sum(-1)
        energy = (self.energy - mc[..., 0]).square()
        return ((spatial + energy) / self.p0 ** 2).mean(-1)

    def constants(self) -> dict:
        return {"fixed_energy_GeV": self.energy, "p0_GeV": self.p0}


def momentum_oracle(train: EventSet, fixed_energy_GeV: float = 45.6, tau_mass_GeV: float = 1.777) -> MomentumOracle:
    if not fixed_energy_GeV > tau_mass_GeV >= 0:
        raise ValueError("need fixed_energy_GeV > tau_mass_GeV >= 0")
    check_truth(train)
    return MomentumOracle(fixed_energy_GeV, math.sqrt(fixed_energy_GeV ** 2 - tau_mass_GeV ** 2))


def direction_features(train: EventSet):
    """Features (K, B, 19): offsets (4), tau directions (6), direction outer product (9)."""
    del train

    def features(candidates: Tensor, events: EventSet) -> Tensor:
        u = tau_directions(candidates, events.extra["visible"][None])
        outer = (u[..., 0, :, None] * u[..., 1, None, :]).flatten(-2)
        return torch.cat((candidates.flatten(-2), u.flatten(-2), outer), -1).float()

    return features


def derive(arrays: dict) -> dict:
    """``visible`` (N, 2, 3) and ``truth_p4`` (N, 2, 4) from the Z->tautau parquet columns.

    Uses ``lead_{a,b}_visible_{px,py,pz}`` and ``truth_tau_{a,b}_{E,px,py,pz}``.
    """
    stack = lambda name, qs: np.stack([np.stack([np.asarray(arrays[name.format(leg=leg, q=q)], dtype=np.float32)
                                                 for q in qs], -1) for leg in "ab"], 1)
    return {"visible": stack("lead_{leg}_visible_{q}", ("px", "py", "pz")),
            "truth_p4": stack("truth_tau_{leg}_{q}", ("E", "px", "py", "pz"))}
