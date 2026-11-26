"""Domain types: events, the policy / oracle / feature interfaces, and the typed run configuration.

The config dataclasses are the single source of truth for every parameter and
its default. Configs are parsed into them once, in ``experiments/config.py``.

Shapes use these letters: N events, B events in a batch, K candidates per
event, n invisible objects per event, F features per invisible object,
D = n * F.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol

import torch
from torch import Tensor


def _map(d: dict[str, Tensor], fn) -> dict[str, Tensor]:
    return {k: fn(v) for k, v in d.items()}


@dataclass(frozen=True)
class EventSet:
    """Events, already parsed and validated at the boundary.

    Attributes:
        observed: policy inputs, each with leading dimension N (for EveNet:
            ``x``, ``x_mask``, ``conditions``, ``conditions_mask``). Only these
            reach the policy and the classifier.
        target: (N, n, F) physical MC targets of the invisible objects, in the
            representation the policy generates.
        event_id: (N,) int64.
        event_weight: (N,) float32 MC event weights; they weight the classifier BCE.
        extra: any further per-event arrays (leading dimension N) that a
            user-defined oracle or feature map needs, e.g. reconstructed
            visible objects or MC four-momenta. Never given to the policy.
    """

    observed: dict[str, Tensor]
    target: Tensor
    event_id: Tensor
    event_weight: Tensor
    extra: dict[str, Tensor] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def take(self, index: Tensor) -> "EventSet":
        return EventSet(observed=_map(self.observed, lambda v: v[index]), target=self.target[index],
                        event_id=self.event_id[index], event_weight=self.event_weight[index],
                        extra=_map(self.extra, lambda v: v[index]))

    def to(self, device: str | torch.device) -> "EventSet":
        move = lambda v: v.to(device)
        return EventSet(observed=_map(self.observed, move), target=move(self.target),
                        event_id=move(self.event_id), event_weight=move(self.event_weight),
                        extra=_map(self.extra, move))


# ----------------------------------------------------------------------------
# Interfaces. Implementations: policies in core/models.py and core/evenet_policy.py;
# oracles and feature maps are user code (examples in evenet_align/examples/).
# ----------------------------------------------------------------------------


class Policy(Protocol):
    """A conditional diffusion policy over the invisible objects of an event (an ``nn.Module``).

    Diffusion runs on flattened normalized targets of width D = n * F; the
    policy owns that normalization.
    """

    target_shape: tuple[int, int]  # (n, F)

    def normalize(self, target: Tensor) -> Tensor:
        """(N, n, F) physical -> (N, D) normalized."""

    def denormalize(self, x: Tensor) -> Tensor:
        """(N, D) normalized -> (N, n, F) physical."""

    def velocity(self, x_t: Tensor, observed: dict[str, Tensor], t: Tensor) -> Tensor:
        """Predicted velocity (N, D) for noised x_t (N, D) at diffusion time t (N,)."""

    def condition_embedding(self, observed: dict[str, Tensor]) -> Tensor:
        """(N, E) classifier condition input from the observed event."""

    def candidate_features(self, candidates: Tensor, observed: dict[str, Tensor]) -> Tensor:
        """(N, F_c) backbone features of clean physical candidates (N, n, F); F_c may be 0."""


class Oracle(Protocol):
    """MC-supervised physics vote. Built once by ``factory(train_events, **params)``."""

    def distance(self, candidates: Tensor, events: EventSet) -> Tensor:
        """Non-negative distance (K, B) of physical candidates (K, B, n, F) to the MC truth."""

    def constants(self) -> dict[str, Any]:
        """Constants pinned when the oracle was built (saved with the run)."""


class FeatureMap(Protocol):
    """Extra classifier features. Built once by ``factory(train_events, **params)``."""

    def __call__(self, candidates: Tensor, events: EventSet) -> Tensor:
        """(K, B, F_x) features of physical candidates (K, B, n, F)."""


# ----------------------------------------------------------------------------
# Configuration schema. Every default lives here and nowhere else.
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Factory:
    """``function: "package.module:name"`` called as ``name(train_events, **params)``."""

    function: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetConfig:
    """Which events to use.

    ``source: npz`` reads ``<path>/train.npz`` and ``<path>/validation.npz``.
    Arrays named in ``observed`` go to the policy, ``target`` is the MC target,
    ``event_id`` and optional ``event_weight`` are bookkeeping, and every other
    array becomes ``EventSet.extra`` for the oracle and feature maps. The
    defaults match EveNet's input names.

    ``source: parquet`` reads EveNet preprocessed parquet directories
    (``train_parquet``, ``validation_parquet``) directly, keeps events whose
    invisible objects are all valid (and ``event_valid`` > 0), and calls
    ``derive`` (``package.module:function``) to compute the extra arrays the
    oracle needs from the parquet columns; columns listed in ``extra`` pass
    through unchanged.

    ``source: pseudo`` generates deterministic pseudo events (``pseudo``: tau
    or ttbar) for tests; ``n_train``, ``n_validation`` and ``seed`` apply to it only.
    """

    source: Literal["npz", "parquet", "pseudo"] = "npz"
    path: Optional[str] = None
    train_parquet: Optional[str] = None
    validation_parquet: Optional[str] = None
    derive: Optional[str] = None
    extra: tuple[str, ...] = ()
    observed: tuple[str, ...] = ("x", "x_mask", "conditions", "conditions_mask")
    target: str = "x_invisible"
    pseudo: Literal["tau", "ttbar"] = "tau"
    n_train: int = 8192
    n_validation: int = 2048
    seed: int = 1234


@dataclass(frozen=True)
class EveNetConfig:
    """A supervised fine-tuned EveNet checkpoint produced by ``evenet-train``.

    checkpoint: the fine-tuned Lightning ``.ckpt``; the released pretrained
        weights are refused, because RL starts after supervised training.
    config: the EveNet training YAML used for that fine-tune (network, event_info, options).
    normalization: the ``normalization.pt`` of that fine-tune.
    source: optional directory holding the ``evenet`` package that trained the
        checkpoint (e.g. a fork with extra modules); default: the installed ``evenet-core``.

    Memory and precision of the trainable policy (as the paper's DGPO runs):
    activation_checkpointing recomputes PET transformer-block activations in
    the backward pass; activation_memory_mode (e.g. ``recompute``) applies to
    PET local-embedding blocks; both need an EveNet build that supports them
    (``evenet.utilities.partial_checkpointing`` / ``activation_offload``).
    float32_matmul_precision is passed to ``torch.set_float32_matmul_precision``.
    """

    checkpoint: Optional[str] = None
    config: Optional[str] = None
    normalization: Optional[str] = None
    source: Optional[str] = None
    activation_checkpointing: bool = False
    activation_memory_mode: Optional[Literal["offload", "recompute", "adaptive"]] = None
    float32_matmul_precision: Optional[Literal["highest", "high", "medium"]] = None


@dataclass(frozen=True)
class MLPConfig:
    """Small reference policy for tests; trained here with velocity MSE before RL."""

    hidden: int = 128
    depth: int = 3
    time_frequencies: int = 16
    supervised_steps: int = 3000
    batch_size: int = 512
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0


@dataclass(frozen=True)
class CustomPolicyConfig:
    """Your own supervised policy: ``function(train_events, **params)`` returns a ``Policy`` (an ``nn.Module``).

    ``files`` (for example your checkpoint) are hashed with sha256 into the run ID,
    so a changed file gives a new run.
    """

    function: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)
    files: tuple[str, ...] = ()


@dataclass(frozen=True)
class PolicyConfig:
    name: Literal["evenet", "mlp", "custom"] = "evenet"
    evenet: EveNetConfig = field(default_factory=EveNetConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    custom: CustomPolicyConfig = field(default_factory=CustomPolicyConfig)


@dataclass(frozen=True)
class DGPOConfig:
    """One policy update draws ``candidates`` DDIM samples per event for ``batch_size`` events.

    Policy batches walk through the training set in shuffled epochs (incomplete
    last batch dropped). ``checkpoint_every`` sets how often (in completed
    updates) the run saves a resumable state; refit boundaries always save one.
    """

    steps: int = 2000
    candidates: int = 6
    batch_size: int = 2048
    oracle_fraction: float = 0.3
    beta: float = 1.0
    trust_coefficient: float = 1.0
    train_timesteps: int = 6
    t_min: float = 0.0
    t_max: float = 0.7
    learning_rate: float = 5.0e-5
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    ddim_steps: int = 20
    validate_every: int = 50
    refit_every: int = 500
    checkpoint_every: int = 50


@dataclass(frozen=True)
class ClassifierConfig:
    """General classifier fit: MC targets (label 1) against current-policy samples (label 0).

    The classifier population is ``fit_events`` training events (null: all of
    them) chosen once per run and split once into fit and validation events
    (``validation_fraction``). Every refit regenerates one policy sample per
    population event. ``batch_size`` counts events; each batch holds the MC
    target and the policy sample of the same events. Each round starts a fresh
    head from the same seed.

    Candidate features are the policy's frozen backbone features of the
    candidate plus, if ``features`` is set, a user-defined feature map.
    """

    features: Optional[Factory] = None
    fit_events: Optional[int] = None
    validation_fraction: float = 0.2
    batch_size: int = 32768
    learning_rate: float = 1.0e-4
    min_learning_rate: float = 1.0e-5
    lr_decay_epochs: int = 1500
    weight_decay: float = 0.005
    dropout: float = 0.1
    grad_clip_norm: float = 1.0
    max_epochs: int = 3000
    min_updates: int = 1000
    patience: int = 25
    min_delta: float = 1.0e-4
    require_plateau: bool = True
    ratio_bound: float = 30.0


@dataclass(frozen=True)
class ValidationConfig:
    ddim_steps: int = 20


@dataclass(frozen=True)
class RunConfig:
    """``results`` and ``device`` say where and on what a run executes; they do not enter the run hash."""

    oracle: Factory
    experiment: str = "default"
    results: str = "results"
    device: str = "cpu"
    seeds: tuple[int, ...] = (0,)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    dgpo: DGPOConfig = field(default_factory=DGPOConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


def make_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    """The only way randomness enters ``core``: an explicit, seeded generator.

    Random tensors are drawn on the generator's device, so pass the device the
    models run on.
    """
    return torch.Generator(device=device).manual_seed(int(seed))


def derive_seed(seed: int, *labels: int | str) -> int:
    """Deterministic child seed for an independent random stream (e.g. one refit round)."""
    text = ":".join(str(x) for x in (seed, *labels))
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little") % (2**62)
