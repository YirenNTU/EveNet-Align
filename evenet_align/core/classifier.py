"""General classifier: candidate features, a resumable plateau-selected fit, and candidate scoring."""
from __future__ import annotations

import copy
import math
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Iterator, Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .models import GeneralClassifier, build_classifier_head
from .rollout import repeat_observed
from .types import ClassifierConfig, EventSet, FeatureMap, Policy, derive_seed, make_generator


@dataclass(frozen=True)
class FeatureScale:
    """Pinned standardization of classifier candidate features, shape (n_features,)."""

    mean: Tensor
    std: Tensor

    def apply(self, features: Tensor) -> Tensor:
        return (features - self.mean) / self.std


def measure_feature_scale(target_features: Tensor, policy_features: Tensor) -> FeatureScale:
    """Mean and std over the MC-target and policy features of the initial fit events."""
    f = torch.cat((target_features, policy_features)).double()
    return FeatureScale(mean=f.mean(0).float(), std=f.std(0).clamp_min(1e-6).float())


FEATURE_CHUNK = 8192  # candidate rows per backbone forward; bounds memory only


@dataclass(frozen=True)
class CandidateFeatures:
    """Classifier candidate features: frozen-backbone features of the candidate, then the optional extra map."""

    backbone: Policy
    extra: Optional[FeatureMap]

    @torch.no_grad()
    def __call__(self, candidates: Tensor, events: EventSet) -> Tensor:
        """(K, B, F_total) for physical candidates (K, B, n, F)."""
        k, b = candidates.shape[:2]
        rows = candidates.reshape(k * b, *candidates.shape[2:])
        observed = repeat_observed(events.observed, k)
        backbone = torch.cat([self.backbone.candidate_features(
            rows[i:i + FEATURE_CHUNK], {key: v[i:i + FEATURE_CHUNK] for key, v in observed.items()})
            for i in range(0, len(rows), FEATURE_CHUNK)])
        parts = [backbone.reshape(k, b, -1)]
        if self.extra is not None:
            parts.append(self.extra(candidates, events))
        return torch.cat(parts, -1)


@dataclass(frozen=True)
class Scorer:
    """Frozen pretrained backbone + candidate features + installed classifier head."""

    backbone: Policy
    features: CandidateFeatures
    scale: FeatureScale
    head: GeneralClassifier

    @torch.no_grad()
    def logits(self, candidates: Tensor, events: EventSet) -> Tensor:
        """Classifier logits (K, B) for physical candidates (K, B, n, F)."""
        k, b = candidates.shape[:2]
        embedding = self.backbone.condition_embedding(events.observed)
        embedding = embedding[None].expand(k, -1, -1).reshape(k * b, -1)
        features = self.scale.apply(self.features(candidates, events)).reshape(k * b, -1)
        return self.head.eval()(embedding, features).reshape(k, b)


@dataclass(frozen=True)
class FitData:
    """Paired classifier rows: per event, the MC-target and the policy-sample features."""

    embedding: Tensor  # (N, E) frozen-backbone condition embedding
    target_features: Tensor  # (N, F), label 1
    policy_features: Tensor  # (N, F), label 0
    weight: Tensor  # (N,) event weights


@dataclass(frozen=True)
class FitState:
    """Everything needed to continue an interrupted fit exactly where it stopped."""

    epochs: int
    updates: int
    head: dict[str, Tensor]
    optimizer: dict
    best_head: dict[str, Tensor]
    best_bce: float
    best_epoch: int
    since_best: int
    rng: dict[str, Tensor]
    finished: bool
    plateau: bool


@dataclass(frozen=True)
class FitResult:
    head: GeneralClassifier
    validation_bce: float
    validation_auc: float
    best_epoch: int
    epochs: int
    updates: int
    plateau: bool


def binary_auc(scores: Tensor, labels: Tensor) -> float:
    """Area under the ROC curve (Mann-Whitney U with average ranks on ties)."""
    scores, labels = scores.double().cpu(), labels.bool().cpu()
    order = scores.argsort()
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    _, inverse, counts = torch.unique(scores[order], return_inverse=True, return_counts=True)
    rank_sum = torch.zeros(len(counts), dtype=torch.float64).index_add_(0, inverse, ranks[order])
    ranks[order] = (rank_sum / counts)[inverse]
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def paired_bce(head: GeneralClassifier, data: FitData, index: Tensor) -> Tensor:
    """Event-weighted mean of (BCE(target, 1) + BCE(policy, 0)) / 2 over the events in ``index``."""
    embedding = data.embedding[index]
    logits = head(torch.cat((embedding, embedding)),
                  torch.cat((data.target_features[index], data.policy_features[index])))
    p, q = logits.chunk(2)
    w = data.weight[index]
    return (w * (F.softplus(-p) + F.softplus(q)) / 2).sum() / w.sum()


@torch.no_grad()
def _validate(head: GeneralClassifier, data: FitData, batch_size: int) -> tuple[float, Tensor, Tensor]:
    """Event-weighted paired validation BCE, plus scores and labels for the AUC."""
    head.eval()
    target, policy = [], []
    for start in range(0, len(data.weight), batch_size):
        index = slice(start, start + batch_size)
        target.append(head(data.embedding[index], data.target_features[index]))
        policy.append(head(data.embedding[index], data.policy_features[index]))
    p, q = torch.cat(target), torch.cat(policy)
    bce = float((data.weight * (F.softplus(-p) + F.softplus(q)) / 2).sum() / data.weight.sum())
    labels = torch.cat((torch.ones_like(p), torch.zeros_like(q)))
    return bce, torch.cat((p, q)), labels


def cosine_learning_rate(cfg: ClassifierConfig, epoch: int) -> float:
    progress = min(epoch, cfg.lr_decay_epochs) / cfg.lr_decay_epochs
    return cfg.min_learning_rate + 0.5 * (cfg.learning_rate - cfg.min_learning_rate) * (1 + math.cos(math.pi * progress))


@contextmanager
def _isolated_rng(device: torch.device, seed: int, saved: Optional[dict[str, Tensor]]):
    """Private global RNG (dropout) for the fit; the caller's RNG state is restored afterwards."""
    cuda = device.type == "cuda"
    with torch.random.fork_rng(devices=[device] if cuda else []):
        if saved is None:
            torch.manual_seed(seed)
        else:
            torch.set_rng_state(saved["cpu"].cpu())
            if cuda:
                torch.cuda.set_rng_state(saved["cuda"].cpu(), device)
        yield


def _rng_state(device: torch.device) -> dict[str, Tensor]:
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def fit_classifier(train: FitData, validation: FitData, cfg: ClassifierConfig, seed: int,
                   resume: Optional[FitState] = None) -> Iterator[Union[FitState, FitResult]]:
    """Fit a fresh seeded head; select the epoch with the best validation BCE.

    Yields a ``FitState`` after every epoch (enough to resume exactly), then the
    ``FitResult``. Stops at a plateau: at least ``min_updates`` optimizer updates
    and no validation-BCE improvement larger than ``min_delta`` for ``patience``
    epochs. The AUC comes from the same best-BCE checkpoint.
    """
    device = train.embedding.device
    with _isolated_rng(device, seed, None if resume is None else resume.rng):
        head = build_classifier_head(train.embedding.shape[1], train.target_features.shape[1],
                                     cfg.dropout, cfg.ratio_bound, seed).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        if resume is None:
            state = FitState(epochs=0, updates=0, head={}, optimizer={}, best_head=copy.deepcopy(head.state_dict()),
                             best_bce=math.inf, best_epoch=0, since_best=0, rng={}, finished=False, plateau=False)
        else:
            state = resume
            head.load_state_dict(state.head)
            optimizer.load_state_dict(state.optimizer)
        n = len(train.weight)
        while not state.finished:
            epoch = state.epochs
            for group in optimizer.param_groups:
                group["lr"] = cosine_learning_rate(cfg, epoch)
            head.train()
            order = torch.randperm(n, generator=make_generator(derive_seed(seed, "epoch", epoch), device), device=device)
            updates = state.updates
            for start in range(0, n, cfg.batch_size):
                loss = paired_bce(head, train, order[start:start + cfg.batch_size])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), cfg.grad_clip_norm)
                optimizer.step()
                updates += 1
            bce, _, _ = _validate(head, validation, cfg.batch_size)
            improved = bce < state.best_bce - cfg.min_delta
            since_best = 0 if improved else state.since_best + 1
            plateau = updates >= cfg.min_updates and since_best >= cfg.patience
            state = replace(
                state, epochs=epoch + 1, updates=updates, since_best=since_best, plateau=plateau,
                finished=plateau or epoch + 1 >= cfg.max_epochs,
                best_bce=bce if improved else state.best_bce,
                best_epoch=epoch + 1 if improved else state.best_epoch,
                best_head=copy.deepcopy(head.state_dict()) if improved else state.best_head,
                head=copy.deepcopy(head.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
                rng=_rng_state(device))
            yield state
        head.load_state_dict(state.best_head)
        bce, scores, labels = _validate(head, validation, cfg.batch_size)
        yield FitResult(head=head.eval().requires_grad_(False), validation_bce=bce,
                        validation_auc=binary_auc(scores, labels), best_epoch=state.best_epoch,
                        epochs=state.epochs, updates=state.updates, plateau=state.plateau)
