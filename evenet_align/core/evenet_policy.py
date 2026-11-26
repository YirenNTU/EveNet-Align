"""EveNet as the DGPO policy.

Wraps an ``evenet.network.evenet_model.EveNetModel`` built with neutrino
generation (package ``evenet-core``). EveNet owns the normalization of both
its observed inputs and the invisible targets; this wrapper only reshapes.

The classifier inputs follow the paper:

* condition: EveNet-normalized point cloud and global conditions with their masks;
* candidate: the frozen network's features of the clean candidate, read at the
  input of the final ``TruthGeneration.generator`` layer at t = 0.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn


class EveNetPolicy(nn.Module):
    """``Policy`` on EveNet inputs ``x``, ``x_mask``, ``conditions``, ``conditions_mask`` (+ ``classification``)."""

    def __init__(self, build: Callable[[], nn.Module], target_shape: tuple[int, int]):
        """``build()`` returns a fresh EveNetModel of the right architecture; weights are loaded separately."""
        super().__init__()
        self.build = build
        self.model = build()
        self.target_shape = tuple(target_shape)

    def __deepcopy__(self, memo: dict) -> "EveNetPolicy":
        """EveNetModel cannot be deep-copied; rebuild it and copy the weights (as the research trainer did)."""
        clone = EveNetPolicy(self.build, self.target_shape).to(next(self.parameters()).device)
        clone.load_state_dict(self.state_dict())
        clone.train(self.training)
        for mine, theirs in zip(self.parameters(), clone.parameters()):
            theirs.requires_grad_(mine.requires_grad)
        return clone

    def _mask(self, n: int, like: Tensor) -> Tensor:
        return torch.ones(n, self.target_shape[0], 1, device=like.device, dtype=like.dtype)

    def _padded(self, x: Tensor) -> Tensor:
        """Older EveNet pads the invisible normalizer to the visible feature width; pad to match."""
        width = self.model.invisible_normalizer.mean.shape[-1]
        return torch.nn.functional.pad(x, (0, width - x.shape[-1]))

    def normalize(self, target: Tensor) -> Tensor:
        f = self.target_shape[1]
        out = self.model.invisible_normalizer(x=self._padded(target), mask=self._mask(len(target), target))
        return out[..., :f].flatten(1)

    def denormalize(self, x: Tensor) -> Tensor:
        x = x.reshape(-1, *self.target_shape)
        out = self.model.invisible_normalizer.denormalize(self._padded(x), mask=self._mask(len(x), x))
        return out[..., :self.target_shape[1]]

    def velocity(self, x_t: Tensor, observed: dict[str, Tensor], t: Tensor) -> Tensor:
        noise = x_t.reshape(-1, *self.target_shape)
        v = self.model.predict_diffusion_vector(noise_x=noise, cond_x=observed, time=t, mode="neutrino",
                                                noise_mask=self._mask(len(noise), noise))
        return v.reshape(len(noise), -1)

    def condition_embedding(self, observed: dict[str, Tensor]) -> Tensor:
        x_mask = observed["x_mask"][..., None].to(observed["x"].dtype)
        c_mask = observed["conditions_mask"][..., None].to(observed["conditions"].dtype)
        x = self.model.sequential_normalizer(observed["x"], mask=x_mask)
        g = self.model.global_normalizer(observed["conditions"][:, None], mask=c_mask)
        return torch.cat((x.flatten(1), x_mask.flatten(1), g.flatten(1), c_mask.flatten(1)), 1)

    def candidate_features(self, candidates: Tensor, observed: dict[str, Tensor]) -> Tensor:
        n = self.target_shape[0]
        captured = []
        hook = self.model.TruthGeneration.generator.register_forward_pre_hook(
            lambda module, args: captured.append(args[0][:, -n:]))
        try:
            self.velocity(self.normalize(candidates), observed, torch.zeros(len(candidates), device=candidates.device))
        finally:
            hook.remove()
        return captured[0].flatten(1)
