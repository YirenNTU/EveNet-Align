"""DGPO: one policy update, classifier rounds, validation draws and the resumable schedule.

``train_dgpo`` is a generator. It yields typed records (metrics, validation
draws, classifier rounds, resumable states, final weights) and never touches
the filesystem; the experiment shell decides where each record goes.

Random streams are independent and keyed by position, so an interrupted run
resumes to the same result as an uninterrupted one:

* policy update s: ``derive_seed(seed, "update", s)`` (rollouts, t, eps)
* policy batches: one shuffle per data epoch, ``derive_seed(seed, "epoch", e)``
* refit at step s: ``derive_seed(seed, "refit", s)`` (policy samples for the classifier)
* classifier fit: ``derive_seed(seed, "classifier")``, the same for every round
* validation: ``derive_seed(seed, "validation")``, the same prior noise at every checkpoint
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterator, Optional, Union

import torch
from torch import Tensor, nn

from .classifier import (CandidateFeatures, FeatureScale, FitData, FitResult, FitState, Scorer,
                         fit_classifier, measure_feature_scale)
from .diffusion import noise_and_target
from .losses import dgpo_loss, reference_anchor_loss, row_velocity_mse
from .models import GeneralClassifier, build_classifier_head
from .reward import effective_sample_size, hybrid_reward, leave_one_out_advantage, vote_winner
from .rollout import repeat_observed, repeat_rows, sample_normalized, to_physical
from .types import DGPOConfig, EventSet, Oracle, Policy, RunConfig, derive_seed, make_generator

# ----------------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class UpdateMetrics:
    step: int
    values: dict[str, float]


@dataclass(frozen=True)
class ValidationDraw:
    """One raw DDIM draw per validation event after ``step`` completed policy updates."""

    step: int
    prediction: Tensor  # (N_val, 2, F), physical
    values: dict[str, float]


@dataclass(frozen=True)
class DGPOSetup:
    """Oracle constants pinned on the training set; yielded once at the start of every invocation."""

    oracle_constants: dict


@dataclass(frozen=True)
class ClassifierRound:
    step: int
    round: int
    fit: FitResult
    scale: FeatureScale
    values: dict[str, float]


@dataclass(frozen=True)
class FitProgress:
    """A classifier fit in progress for the refit at ``step``; resumable on its own."""

    step: int
    state: FitState


@dataclass(frozen=True)
class DGPOState:
    """Resumable state after ``step`` completed updates, before that step's validation and refit."""

    step: int
    policy: dict[str, Tensor]
    optimizer: dict
    reference: Optional[dict[str, Tensor]]
    head: Optional[dict[str, Tensor]]
    scale: Optional[FeatureScale]
    round: int  # installed classifier round, -1 before the first fit


@dataclass(frozen=True)
class FinalPolicy:
    state: dict[str, Tensor]


Record = Union[DGPOSetup, UpdateMetrics, ValidationDraw, ClassifierRound, FitProgress, DGPOState, FinalPolicy]


# ----------------------------------------------------------------------------
# One policy update
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardContext:
    """Everything that scores candidates during one classifier round."""

    scorer: Scorer
    oracle: Oracle


def policy_update(policy: Policy, reference: Policy, optimizer: torch.optim.Optimizer,
                  context: RewardContext, events: EventSet, cfg: DGPOConfig,
                  generator: torch.Generator) -> dict[str, float]:
    """One DGPO policy update on a batch of B events. Returns diagnostics.

    1. Draw K completed DDIM candidates per event from the current policy.
    2. Hybrid reward from the frozen installed classifier and the physics vote.
    3. Leave-one-out advantages.
    4. ``train_timesteps`` shared (t, eps) draws per event; both the policy and
       the frozen round reference see the same draw. The loss is the gated
       DGPO term plus ``trust_coefficient`` times the velocity anchor. The
       draws are averaged into one AdamW step.
    """
    k, b = cfg.candidates, len(events)
    device = generator.device
    with torch.no_grad():
        x0 = sample_normalized(policy, events, k, cfg.ddim_steps, generator)
        physical = to_physical(x0, policy)
        logits = context.scorer.logits(physical, events)
        distance = context.oracle.distance(physical, events)
        reward = hybrid_reward(logits, distance, cfg.oracle_fraction)
        advantage = leave_one_out_advantage(reward).float()
    x0 = x0.reshape(k * b, -1)
    observed = repeat_observed(events.observed, k)

    optimizer.zero_grad(set_to_none=True)
    totals = {"loss_dgpo": 0.0, "loss_anchor": 0.0, "gate_mean": 0.0}
    for _ in range(cfg.train_timesteps):
        t = cfg.t_min + (cfg.t_max - cfg.t_min) * torch.rand(b, generator=generator, device=device)
        eps = torch.randn(b, x0.shape[1], generator=generator, device=device)
        t_rows = repeat_rows(t, k)
        x_t, v_target = noise_and_target(x0, t_rows, repeat_rows(eps, k))
        v_current = policy.velocity(x_t, observed, t_rows)
        with torch.no_grad():
            v_reference = reference.velocity(x_t, observed, t_rows)
        main, diag = dgpo_loss(row_velocity_mse(v_current, v_target).reshape(k, b),
                               row_velocity_mse(v_reference, v_target).reshape(k, b), advantage, cfg.beta)
        anchor = reference_anchor_loss(v_current, v_reference)
        ((main + cfg.trust_coefficient * anchor) / cfg.train_timesteps).backward()
        totals["loss_dgpo"] += float(main.detach()) / cfg.train_timesteps
        totals["loss_anchor"] += float(anchor.detach()) / cfg.train_timesteps
        totals["gate_mean"] += float(diag["gate_mean"]) / cfg.train_timesteps
    grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip_norm)
    optimizer.step()

    winner = vote_winner(distance)
    return {**totals,
            "grad_norm_pre_clip": float(grad_norm),
            "reward_mean": float(reward.mean()),
            "advantage_abs_mean": float(advantage.abs().mean()),
            "ess_mixture": float(effective_sample_size(reward).mean()),
            "ess_classifier": float(effective_sample_size(logits.double()).mean()),
            "vote_distance_mean": float(distance.mean()),
            "vote_distance_winner": float(distance.gather(0, winner[None]).mean()),
            "classifier_agrees_with_vote": float((logits.argmax(0) == winner).float().mean())}


# ----------------------------------------------------------------------------
# Classifier population and rounds
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassifierPopulation:
    """Fixed training events used for every classifier fit, split once into fit and validation."""

    fit: EventSet
    validation: EventSet


def classifier_population(train: EventSet, run: RunConfig, seed: int) -> ClassifierPopulation:
    cfg = run.classifier
    device = train.target.device
    order = torch.randperm(len(train), generator=make_generator(derive_seed(seed, "population"), device),
                           device=device)[:cfg.fit_events]  # None keeps every training event
    n_val = int(round(cfg.validation_fraction * len(order)))
    return ClassifierPopulation(fit=train.take(order[n_val:]), validation=train.take(order[:n_val]))


@dataclass(frozen=True)
class RefitSamples:
    """Unscaled classifier inputs for one population split: one policy draw per event."""

    embedding: Tensor  # (N, E) frozen-backbone condition embedding
    target_features: Tensor  # (N, F)
    policy_features: Tensor  # (N, F)
    weight: Tensor  # (N,)

    def scaled(self, scale: FeatureScale) -> FitData:
        return FitData(self.embedding, scale.apply(self.target_features), scale.apply(self.policy_features),
                       self.weight)


def _refit_samples(events: EventSet, policy: Policy, backbone: Policy, features: CandidateFeatures,
                   ddim_steps: int, generator: torch.Generator) -> RefitSamples:
    sample = to_physical(sample_normalized(policy, events, 1, ddim_steps, generator), policy)
    with torch.no_grad():
        return RefitSamples(embedding=backbone.condition_embedding(events.observed),
                            target_features=features(events.target[None], events)[0],
                            policy_features=features(sample, events)[0], weight=events.event_weight)


def refit_samples(policy: Policy, backbone: Policy, features: CandidateFeatures, population: ClassifierPopulation,
                  run: RunConfig, seed: int, step: int) -> tuple[RefitSamples, RefitSamples]:
    """Fresh current-policy samples for the fixed population (fit split, validation split)."""
    generator = make_generator(derive_seed(seed, "refit", step), population.fit.target.device)
    return tuple(_refit_samples(events, policy, backbone, features, run.dgpo.ddim_steps, generator)
                 for events in (population.fit, population.validation))


@torch.no_grad()
def validation_draw(policy: Policy, validation: EventSet, oracle: Oracle, run: RunConfig, seed: int,
                    step: int) -> ValidationDraw:
    """One raw DDIM draw per event with the same prior noise at every checkpoint."""
    generator = make_generator(derive_seed(seed, "validation"), validation.target.device)
    prediction = to_physical(sample_normalized(policy, validation, 1, run.validation.ddim_steps, generator), policy)
    distance = oracle.distance(prediction, validation)[0]
    return ValidationDraw(step=step, prediction=prediction[0],
                          values={"vote_distance_mean": float(distance.mean()),
                                  "vote_distance_median": float(distance.median())})


def policy_batch(train: EventSet, batch_size: int, seed: int, step: int) -> EventSet:
    """Batch for update ``step``: shuffled epochs over the training set, last partial batch dropped."""
    per_epoch = len(train) // batch_size
    epoch, position = divmod(step, per_epoch)
    device = train.target.device
    order = torch.randperm(len(train), generator=make_generator(derive_seed(seed, "epoch", epoch), device),
                           device=device)
    return train.take(order[position * batch_size:(position + 1) * batch_size])


# ----------------------------------------------------------------------------
# Schedule
# ----------------------------------------------------------------------------


def _frozen(module: Policy) -> Policy:
    return copy.deepcopy(module).eval().requires_grad_(False)


def _head(state: dict[str, Tensor], run: RunConfig, device: torch.device) -> GeneralClassifier:
    head = build_classifier_head(state["condition_encoder.0.weight"].shape[1], state["spin_encoder.0.weight"].shape[1],
                                 run.classifier.dropout, run.classifier.ratio_bound, 0).to(device)
    head.load_state_dict(state)
    return head.eval().requires_grad_(False)


def train_dgpo(supervised: Policy, train: EventSet, validation: EventSet, oracle: Oracle,
               extra_features: Optional[object], run: RunConfig, seed: int,
               resume: Optional[DGPOState] = None, resume_fit: Optional[FitProgress] = None) -> Iterator[Record]:
    """Full DGPO schedule starting from the supervised policy, or from ``resume``.

    At each count s of completed policy updates, in this order: yield a
    resumable state (every ``checkpoint_every`` and at refit boundaries),
    validate (every ``validate_every`` and at the end), refit the classifier
    (every ``refit_every``, including s = 0) and refresh the round reference to
    the current policy, then update. ``resume_fit`` continues an interrupted
    classifier fit of the refit at ``resume.step``. The frozen classifier
    backbone is the supervised policy.
    """
    cfg = run.dgpo
    device = train.target.device
    backbone = _frozen(supervised)
    features = CandidateFeatures(backbone, extra_features)
    policy = copy.deepcopy(supervised).train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    yield DGPOSetup(oracle.constants())
    population = classifier_population(train, run, seed)
    start, scale, reference, context, round_index = 0, None, None, None, -1
    if resume is not None:
        start, scale, round_index = resume.step, resume.scale, resume.round
        policy.load_state_dict(resume.policy)
        optimizer.load_state_dict(resume.optimizer)
        if resume.head is not None:
            reference = _frozen(policy)
            reference.load_state_dict(resume.reference)
            context = RewardContext(Scorer(backbone, features, scale, _head(resume.head, run, device)), oracle)
    for step in range(start, cfg.steps + 1):
        refit_due = step < cfg.steps and step % cfg.refit_every == 0
        if (step % cfg.checkpoint_every == 0 or refit_due) and not (resume is not None and step == start):
            yield DGPOState(step=step, policy=copy.deepcopy(policy.state_dict()),
                            optimizer=copy.deepcopy(optimizer.state_dict()),
                            reference=None if reference is None else copy.deepcopy(reference.state_dict()),
                            head=None if context is None else copy.deepcopy(context.scorer.head.state_dict()),
                            scale=scale, round=round_index)
        if step % cfg.validate_every == 0 or step == cfg.steps:
            yield validation_draw(policy, validation, oracle, run, seed, step)
        if step == cfg.steps:
            break
        if refit_due:
            fit_samples, val_samples = refit_samples(policy, backbone, features, population, run, seed, step)
            if scale is None:
                scale = measure_feature_scale(fit_samples.target_features, fit_samples.policy_features)
            fit_resume = resume_fit.state if (resume_fit is not None and resume_fit.step == step) else None
            for item in fit_classifier(fit_samples.scaled(scale), val_samples.scaled(scale), run.classifier,
                                       derive_seed(seed, "classifier"), fit_resume):
                if isinstance(item, FitState):
                    yield FitProgress(step, item)
                else:
                    fit = item
            if run.classifier.require_plateau and not fit.plateau:
                raise RuntimeError(f"classifier fit at step {step} reached max_epochs without a plateau; "
                                   "not installing it")
            round_index = step // cfg.refit_every
            context = RewardContext(Scorer(backbone, features, scale, fit.head), oracle)
            reference = _frozen(policy)
            yield ClassifierRound(step, round_index, fit, scale,
                                  {"validation_bce": fit.validation_bce, "validation_auc": fit.validation_auc,
                                   "best_epoch": fit.best_epoch, "epochs": fit.epochs,
                                   "updates": fit.updates, "plateau": float(fit.plateau)})
        batch = policy_batch(train, cfg.batch_size, seed, step)
        generator = make_generator(derive_seed(seed, "update", step), device)
        values = policy_update(policy, reference, optimizer, context, batch, cfg, generator)
        yield UpdateMetrics(step + 1, values)
    yield FinalPolicy({k: v.detach().clone() for k, v in policy.state_dict().items()})
