"""Template for using your own diffusion model instead of EveNet.

    policy:
      name: custom
      custom:
        function: evenet_align.examples.custom_policy:mlp_policy
        params: {checkpoint: my_supervised_weights.pt}
        files: [my_supervised_weights.pt]      # hashed into the run ID

``function(train_events, **params)`` is called once on the training events and
returns an ``nn.Module`` implementing ``evenet_align.core.types.Policy``:

* ``target_shape``: (n, F) of the invisible objects;
* ``normalize`` / ``denormalize``: physical (N, n, F) <-> normalized (N, n * F);
* ``velocity(x_t, observed, t)``: the diffusion model's velocity prediction;
* ``condition_embedding(observed)``: (N, E) event features for the classifier;
* ``candidate_features(candidates, observed)``: (N, F_c) features of clean
  candidates for the classifier (F_c may be 0).

DGPO fine-tunes this policy, so load weights from your own supervised training.
This template wraps the reference MLP, which reads ``observed["condition"]``;
replace it with your model.
"""
from __future__ import annotations

from typing import Optional

import torch

from evenet_align.core.models import build_mlp_policy, mlp_statistics
from evenet_align.core.types import EventSet, MLPConfig, Policy


def mlp_policy(train: EventSet, checkpoint: Optional[str] = None, hidden: int = 128, depth: int = 3,
               time_frequencies: int = 16, seed: int = 0) -> Policy:
    cfg = MLPConfig(hidden=hidden, depth=depth, time_frequencies=time_frequencies)
    policy = build_mlp_policy(train.observed["condition"].shape[1], tuple(train.target.shape[1:]), cfg,
                              mlp_statistics(train), seed)
    if checkpoint is not None:
        policy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return policy
