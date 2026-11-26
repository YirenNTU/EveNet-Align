"""The example oracles and feature maps (Z->tautau and ttbar) on pseudo events."""
import math

import pytest
import torch

from evenet_align.core.pseudo_data import pseudo_tau, pseudo_ttbar
from evenet_align.examples import tau, ttbar
from evenet_align.experiments.datasets import validate


def test_pseudo_events_are_valid():
    validate(pseudo_tau(300, 1), "tau")
    validate(pseudo_ttbar(300, 1), "ttbar")


def test_tau_oracle_at_truth_is_only_the_energy_and_magnitude_error():
    events = pseudo_tau(500, 3)
    oracle = tau.momentum_oracle(events, fixed_energy_GeV=45.6, tau_mass_GeV=1.777)
    assert math.isclose(oracle.p0, math.sqrt(45.6 ** 2 - 1.777 ** 2))
    distance = oracle.distance(events.target[None], events)[0]
    mc = events.extra["truth_p4"].double()
    p = torch.linalg.vector_norm(mc[..., 1:], dim=-1)
    expected = ((p - oracle.p0).square() + (45.6 - mc[..., 0]).square()).mean(-1) / oracle.p0 ** 2
    assert torch.allclose(distance, expected, rtol=1e-4, atol=1e-8)


def test_tau_oracle_prefers_the_true_direction_and_rejects_bad_truth():
    events = pseudo_tau(200, 4)
    oracle = tau.momentum_oracle(events)
    d = oracle.distance(torch.stack((events.target + torch.tensor([0.05, 0.0]), events.target)), events)
    assert (d[1] < d[0]).all()
    broken = pseudo_tau(50, 5)
    broken.extra["truth_p4"][:, 0, 1:] *= -1
    with pytest.raises(ValueError):
        tau.momentum_oracle(broken)


def test_ttbar_oracle_zero_at_truth_with_shared_spatial_scale():
    events = pseudo_ttbar(300, 5)
    oracle = ttbar.top_momentum_oracle(events)
    s = oracle.constants()["top_p4_scale_GeV"]
    assert s[1] == s[2] == s[3]
    d = oracle.distance(events.target[None], events)
    assert torch.allclose(d, torch.zeros_like(d))


def test_neutrino_p4_is_massless():
    p4 = ttbar.neutrino_p4(torch.tensor([[math.log1p(30.0), 0.7, -2.0]], dtype=torch.float64))[0]
    assert math.isclose(float(p4[0] ** 2 - (p4[1:] ** 2).sum()), 0.0, abs_tol=1e-9)
    assert math.isclose(float(torch.hypot(p4[1], p4[2])), 30.0, rel_tol=1e-12)


def test_feature_shapes():
    for module, factory, events in ((tau, "direction_features", pseudo_tau(10, 1)),
                                    (ttbar, "top_features", pseudo_ttbar(10, 1))):
        f = getattr(module, factory)(events)(events.target[None].expand(3, -1, -1, -1), events)
        assert f.shape[:2] == (3, 10) and torch.isfinite(f).all()
