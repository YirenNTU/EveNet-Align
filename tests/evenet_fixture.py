"""Test fixture: a tiny random EveNet 'fine-tuned' checkpoint and EveNet-format pseudo ttbar events.

    python tests/evenet_fixture.py <EveNet share/ directory> <output directory>

Uses EveNet's own config loader and model (package evenet-core). The weights
are random; this checks the plumbing, not physics.
"""
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from evenet.control.global_config import Config
from evenet.network.evenet_model import EveNetModel

TINY = """options:
  default: options/options.yaml
  Training:
    Components:
      Classification: {include: false}
      Regression: {include: false}
      Assignment: {include: false}
      Segmentation: {include: false}
      GlobalGeneration: {include: false}
      ReconGeneration: {include: false}
      TruthGeneration: {include: true}
network:
  default: network/network-20M.yaml
  Body:
    GlobalEmbedding: {hidden_dim: 32, num_embedding_layers: 1, dropout: 0.0}
    PET: {hidden_dim: 32, num_layers: 2, num_local_layer: 1, num_heads: 2, feature_drop: 0.0, drop_probability: 0.0, dropout: 0.0}
    ObjectEncoder: {hidden_dim: 32, num_embedding_layers: 1, num_encoder_layers: 1, num_attention_heads: 2, dropout: 0.0}
  TruthGeneration: {hidden_dim: 32, num_heads: 2, dropout: 0.0, drop_probability: 0.0, feature_drop: 0.0}
event_info:
  default: event_info/multi_process.yaml
resonance:
  default: resonance/standard_model.yaml
"""

share, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
shutil.copytree(share, out / "share", dirs_exist_ok=True)
(out / "share" / "tiny-finetune.yaml").write_text(TINY)
cfg = Config(); cfg.load_yaml(out / "share" / "tiny-finetune.yaml")
ei = cfg.event_info
names = list(ei.input_types)
n_seq = len(ei.input_features["Source"]); n_glob = len(ei.input_features["Conditions"])
norm = {"input_mean": {"Source": torch.zeros(n_seq), "Conditions": torch.zeros(n_glob)},
        "input_std": {"Source": torch.ones(n_seq), "Conditions": torch.ones(n_glob)},
        "invisible_mean": {"Source": torch.tensor([3.5, 0.0, 0.0])},
        "invisible_std": {"Source": torch.tensor([0.8, 1.3, 1.0])}}
torch.save(norm, out / "normalization.pt")
torch.manual_seed(0)
model = EveNetModel(config=cfg, device=torch.device("cpu"), neutrino_generation=True, normalization_dict=norm)
torch.save({"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}, "epoch": 1}, out / "finetuned.ckpt")

def events(n, seed, first):
    g = np.random.default_rng(seed)
    x = np.zeros((n, 18, n_seq), np.float32); mask = np.zeros((n, 18), bool)
    k = 4
    pt = 40 + 60 * g.exponential(size=(n, k)); eta = g.normal(0, 1.2, (n, k)); phi = g.uniform(-math.pi, math.pi, (n, k))
    e = pt * np.cosh(eta)
    x[:, :k, 0], x[:, :k, 1], x[:, :k, 2], x[:, :k, 3] = np.log1p(e), np.log1p(pt), eta, phi
    x[:, :2, 4] = 1; x[:, 2:4, 5] = 1; x[:, 2, 6] = 1; x[:, 3, 6] = -1; mask[:, :k] = True
    # b + lepton per top: (b0, l2) and (b1, l3)
    p4 = np.stack((e, pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)), -1)
    visible = np.stack((p4[:, 0] + p4[:, 2], p4[:, 1] + p4[:, 3]), 1).astype(np.float32)
    pt_nu = pt[:, 2:4] * np.exp(g.normal(-0.3, 0.5, (n, 2))); eta_nu = eta[:, 2:4] + g.normal(0, 0.6, (n, 2))
    phi_nu = np.mod(phi[:, 2:4] + g.normal(0, 0.5, (n, 2)) + math.pi, 2 * math.pi) - math.pi
    nu = np.stack((np.log1p(pt_nu), eta_nu, phi_nu), -1).astype(np.float32)
    met = np.stack(((pt_nu * np.cos(phi_nu)).sum(1), (pt_nu * np.sin(phi_nu)).sum(1)), -1)
    cond = np.zeros((n, n_glob), np.float32)
    cond[:, 0] = np.log1p(np.hypot(*met.T)); cond[:, 1] = np.arctan2(met[:, 1], met[:, 0])
    return dict(x=x, x_mask=mask, conditions=cond, conditions_mask=np.ones((n, 1), bool), x_invisible=nu,
                x_invisible_mask=np.ones((n, 2), bool), visible=visible, event_id=np.arange(n) + first)

(out / "data").mkdir(exist_ok=True)
np.savez(out / "data" / "train.npz", **events(3000, 1, 0))
np.savez(out / "data" / "validation.npz", **events(500, 2, 10**6))
