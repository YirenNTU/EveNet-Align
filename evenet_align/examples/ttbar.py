"""Dileptonic ttbar oracle and classifier features used in the paper.

Targets are the two neutrinos in native coordinates [log1p(pT/GeV), eta, phi].

Expected ``EventSet.extra`` arrays:
    visible (N, 2, 4)  reconstructed b + lepton four-momentum of each top [E, px, py, pz], GeV

Config::

    oracle: {function: evenet_align.examples.ttbar:top_momentum_oracle}
    classifier:
      features: {function: evenet_align.examples.ttbar:top_features}
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from evenet_align.core.types import EventSet


def neutrino_p4(native: Tensor) -> Tensor:
    """Massless neutrino [E, px, py, pz] (GeV) from native [log1p(pT), eta, phi]."""
    x = native.double()
    pt, eta, phi = x[..., 0].expm1(), x[..., 1], x[..., 2]
    return torch.stack((pt * eta.cosh(), pt * phi.cos(), pt * phi.sin(), pt * eta.sinh()), -1)


class TopMomentumOracle:
    """Mean normalized squared top four-momentum error over both tops and E, px, py, pz.

    Each top is the same reconstructed b + lepton plus either the candidate or
    the MC neutrino, so the visible part cancels in the difference.
    """

    def __init__(self, scale: Tensor):
        self.scale = scale

    def distance(self, candidates: Tensor, events: EventSet) -> Tensor:
        delta = neutrino_p4(candidates) - neutrino_p4(events.target)[None]
        return (delta / self.scale.to(delta)).square().mean((-1, -2))

    def constants(self) -> dict:
        return {"top_p4_scale_GeV": self.scale}


def top_momentum_oracle(train: EventSet) -> TopMomentumOracle:
    """Scales [s_E, s, s, s] pinned from the training set (reconstructed visibles + MC neutrinos).

    s_E is the RMS top energy; one shared spatial scale s (RMS over px, py, pz)
    keeps the distance rotation invariant.
    """
    top = train.extra["visible"].double() + neutrino_p4(train.target)
    mean_square = top.square().reshape(-1, 4).mean(0)
    spatial = float(mean_square[1:].mean().sqrt())
    return TopMomentumOracle(torch.tensor([float(mean_square[0].sqrt()), spatial, spatial, spatial],
                                          dtype=torch.float64, device=train.target.device))


def top_features(train: EventSet):
    """Features (K, B, 25): native nu8, top four-vectors (8), top-direction outer product (9).

    nu8 = [log1p pT, eta, sin phi, cos phi] for each neutrino.
    """
    del train

    def features(candidates: Tensor, events: EventSet) -> Tensor:
        x = candidates
        nu8 = torch.stack((x[..., 0], x[..., 1], x[..., 2].sin(), x[..., 2].cos()), -1).flatten(-2)
        top = (events.extra["visible"].double()[None] + neutrino_p4(x)).float()
        direction = top[..., 1:] / torch.linalg.vector_norm(top[..., 1:], dim=-1, keepdim=True)
        outer = (direction[..., 0, :, None] * direction[..., 1, None, :]).flatten(-2)
        return torch.cat((nu8.float(), top.flatten(-2), outer), -1)

    return features


def derive(arrays: dict) -> dict:
    """``visible`` (N, 2, 4) from EveNet's EXTRA columns: b + lepton four-momentum of each top.

    Uses ``EXTRA/t{1,2}/{b,l}/{pt,eta,phi,energy}`` (the truth-matched reconstructed objects).
    """
    def p4(prefix: str) -> np.ndarray:
        pt, eta, phi, energy = (np.asarray(arrays[f"{prefix}/{v}"], dtype=np.float64)
                                for v in ("pt", "eta", "phi", "energy"))
        return np.stack((energy, pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)), -1)

    tops = [p4(f"EXTRA/t{i}/b") + p4(f"EXTRA/t{i}/l") for i in (1, 2)]
    return {"visible": np.stack(tops, 1).astype(np.float32)}
