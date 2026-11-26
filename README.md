# EveNet-Align: Preference Optimization of Diffusion Models for Neutrino Reconstruction

<!-- TODO before release: replace the arXiv ID, author list and BibTeX below. -->
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue.svg)

Official implementation of **DGPO** (preference optimization of conditional
diffusion models) for reconstructing the invisible particles of a collision
event, from the paper:

> **TODO: Paper title**<br>
> TODO: Authors<br>
> *arXiv:XXXX.XXXXX* (2026)

A supervised conditional diffusion model predicts the invisible particles,
such as the two neutrinos of dileptonic tt̄ or the two tau directions of Z→ττ.
You can start from [EveNet](https://github.com/EveNet-HEP/EveNet-Full), as in
the paper, or plug in **your own diffusion model**
([Using your own model](#using-your-own-model)). DGPO then fine-tunes it by comparing
several candidates drawn for the **same** observed event. Each candidate is
scored by a classifier, trained to separate MC truth from policy samples, and
by a physics oracle that you define. The policy moves toward the preferred
candidates.

## Contents

- [Installation](#installation)
- [Quickstart (CPU, no EveNet)](#quickstart-cpu-no-evenet)
- [Reproducing the paper](#reproducing-the-paper)
- [Using your own model](#using-your-own-model)
- [Using your own data and oracle](#using-your-own-data-and-oracle)
- [Method](#method)
- [Outputs and resuming](#outputs-and-resuming)
- [Reproducibility](#reproducibility)
- [Repository structure and tests](#repository-structure-and-tests)
- [Citation](#citation)
- [License](#license)

## Installation

Requires Python ≥ 3.10 and PyTorch ≥ 2.1.

```sh
git clone https://github.com/YirenNTU/EveNet-Align.git && cd EveNet-Align
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip           # editable installs need pip >= 21.3
pip install -e ".[evenet,dev]"      # EveNet (evenet-core), plus pytest, import-linter, matplotlib
```

Without the `evenet` extra, everything except the EveNet policy works. That is
enough for your own model, the quickstart and the tests
(`pip install -e ".[dev]"`).

## Quickstart (CPU, no EveNet)

`configs/smoke.yaml` swaps EveNet for a small reference MLP policy that trains
its own supervised stage, and uses pseudo Z→ττ events. It runs the same DGPO
code in about ten seconds on a CPU:

```sh
python scripts/run.py configs/smoke.yaml
python scripts/summarize.py results/smoke --output figures/smoke
python -m pytest
```

`summarize.py` writes `history.csv`, `summary.csv` and `history.png`. For every
validation step they show how much each metric changed relative to the same
run's supervised starting point. The pseudo events are not a physics
simulation. They only reproduce the structure the algorithm relies on, so
numbers measured on them say nothing about physics results.

## Reproducing the paper

### 1. Supervised EveNet (from the public release)

Everything starts from the public EveNet release: the model code
([EveNet-Full](https://github.com/EveNet-HEP/EveNet-Full), installed as
`evenet-core` by the `evenet` extra) and its pretrained weights. DGPO starts
from a supervised policy, not from the pretrained weights themselves (it
refuses those files), so first fine-tune them on your events:

1. Download the pretrained weights from
   [huggingface.co/Avencast/EveNet](https://huggingface.co/Avencast/EveNet).
2. Fine-tune them on your events with `evenet-train`. Set
   `options.Training.pretrain_model_load_path` to the downloaded file and
   include `TruthGeneration`. See the
   [EveNet documentation](https://evenet-hep.github.io/EveNet-Full/).
3. Keep three outputs: the fine-tuned checkpoint, its YAML and its
   `normalization.pt`.

The supervised checkpoints used in the paper are not released; train your own
as above. If your checkpoint was trained with a modified EveNet, point
`policy.evenet.source` to that EveNet checkout; otherwise leave it `null` to
use the installed `evenet-core`. The loader refuses a checkpoint whose
parameters do not match the model it builds.

### 2. Data

The paper configs read EveNet preprocessed parquet directly
(`dataset.source: parquet`, with `dataset.train_parquet` and
`dataset.validation_parquet`). To use npz files instead, convert them once
(see [Your events](#your-events)):

```sh
python scripts/evenet_parquet_to_npz.py --train <train parquet dir> --validation <validation parquet dir> \
    --output data/ttbar --derive evenet_align.examples.ttbar:derive
```

### 3. DGPO

`configs/ttbar.yaml` and `configs/ztautau.yaml` hold the single-GPU paper
recipe. Their paths are placeholders; set them in the file or with `--set`:

```sh
python scripts/run.py configs/ztautau.yaml --device cuda \
    --set policy.evenet.checkpoint=<ckpt> --set policy.evenet.config=<yaml> \
    --set policy.evenet.normalization=<normalization.pt>
```

| setting | value |
|---|---|
| candidates per event K | 6 |
| events per update | 2,048 |
| oracle fraction, β, anchor coefficient | 0.3, 1, 1 |
| diffusion times per update | 6 on [0, 0.7] |
| sampling | DDIM, 20 steps |
| classifier refit | every 500 updates, all training events, batches of 32,768, plateau rule |
| validation | every 50 updates, one DDIM20 draw per event |

**GPU memory.** The paper runs used activation checkpointing
(`policy.evenet.activation_checkpointing: true`,
`activation_memory_mode: recompute`) to fit 2,048 × 6 candidates on one
80 GB A100. These flags only change how the backward pass stores activations,
not the results, but they need an EveNet build that provides
`evenet.utilities.partial_checkpointing` and `evenet.utilities.activation_offload`,
which the released `evenet-core` does not. They are therefore off in the
configs. With stock `evenet-core`, use a GPU with enough memory; with such an
EveNet build, set `policy.evenet.source` to it and turn the flags on.

To test the full pipeline quickly with the same command, add
`--set dgpo.refit_every=10 --set dgpo.steps=20`.
`scripts/perlmutter.sbatch` is a requeueable single-GPU SLURM template.

### 4. Evaluation

```sh
python scripts/summarize.py results/ztautau --output figures/ztautau
```

<!-- TODO before release: add the main results table / figure from the paper. -->

## Using your own model

DGPO only needs a supervised conditional diffusion policy; EveNet is one
choice. To use your own, set:

```yaml
policy:
  name: custom
  custom:
    function: my_analysis.models:load_policy   # package.module:name, importable from where you run
    params: {checkpoint: my_supervised.pt}     # passed as keyword arguments
    files: [my_supervised.pt]                  # sha256-hashed into the run ID
```

`load_policy(train_events, **params)` is called once on the training events and
returns an `nn.Module` implementing `evenet_align.core.types.Policy`:

| member | shape | role |
|---|---|---|
| `target_shape` | (n, F) | invisible objects per event and features per object |
| `normalize(target)` / `denormalize(x)` | (N, n, F) ↔ (N, n·F) | physical ↔ normalized diffusion space |
| `velocity(x_t, observed, t)` | (N, n·F) | velocity prediction at diffusion time t ∈ [0, 1] |
| `condition_embedding(observed)` | (N, E) | event features given to the classifier |
| `candidate_features(candidates, observed)` | (N, F_c) | features of clean candidates for the classifier; F_c may be 0 |

`observed` holds the arrays named in `dataset.observed`. Load weights from your
own supervised training: DGPO fine-tunes a policy, it does not train one from
scratch. `evenet_align/examples/custom_policy.py` is a working template that
wraps the reference MLP:

```sh
python scripts/run.py configs/smoke.yaml --set policy.name=custom \
    --set policy.custom.function=evenet_align.examples.custom_policy:mlp_policy
```

## Using your own data and oracle

### Your oracle

The oracle and its parameters come from the config:

```yaml
oracle:
  function: my_analysis.oracles:my_oracle    # package.module:name, importable from where you run
  params: {fixed_energy_GeV: 45.6}           # passed as keyword arguments
classifier:
  features:                                  # optional extra classifier features
    function: my_analysis.oracles:my_features
    params: {}
```

`my_oracle(train_events, **params)` is called once on the training events. It
returns an object with two methods:

- `distance(candidates, events)`: takes physical candidates of shape
  (K, B, n, F) and returns a non-negative (K, B) distance to the MC truth.
- `constants()`: returns the values the oracle pinned when it was built; they
  are saved with the run.

`my_features(train_events, **params)` returns a function
`(candidates, events) -> (K, B, F_x)`. Inside these functions, the MC target is
`events.target` (B, n, F), and every other array of your npz files is
`events.extra["<name>"]`.

The paper's two oracles, with matching classifier features, are in
`evenet_align/examples/`:

- `tau.py` (Z→ττ): each candidate tau gets the fixed energy E and momentum
  p₀ = √(E² − m²) along its predicted direction. The distance is the
  component-wise four-momentum error to the MC tau, divided by p₀² and
  averaged over both legs. Needs `extra["visible"]` and `extra["truth_p4"]`.
- `ttbar.py` (tt̄): each top is the reconstructed b + lepton plus either the
  candidate or the MC neutrino. The distance is the squared top four-momentum
  error, divided by scales [s_E, s, s, s] measured once on the training set,
  with one shared spatial scale. Needs `extra["visible"]`.

### Your events

For EveNet preprocessed parquet (flat `name:i:j` columns plus
`shape_metadata.json`), use `scripts/evenet_parquet_to_npz.py` as shown above.
It keeps events whose invisible objects are all valid, and also requires
`event_valid` > 0 when that column exists. `--derive` computes the arrays your
oracle needs: the tt̄ example builds each top's b + lepton from
`EXTRA/t*/{b,l}/*`; the Z→ττ example (`evenet_align.examples.tau:derive`)
reads `lead_*_visible_*` and `truth_tau_*`.

Otherwise, set `dataset.source: npz` and `dataset.path: <dir>`, a directory
holding `train.npz` and `validation.npz`:

| array | shape | role |
|---|---|---|
| names in `dataset.observed` (default `x`, `x_mask`, `conditions`, `conditions_mask`) | (N, ...) | EveNet inputs, the only thing the policy and classifier see |
| `dataset.target` (default `x_invisible`) | (N, n, F) | MC targets of the invisible objects |
| `x_invisible_mask` (optional) | (N, n) | must mark every object valid |
| `event_id` | (N,) | unique integers |
| `event_weight` (optional) | (N,) | positive MC weights for the classifier, default 1 |
| anything else | (N, ...) | passed to your oracle and features as `events.extra` |

Store the EveNet arrays in the representation EveNet trains on, the same values
as its preprocessed inputs, so that EveNet's own normalization applies
unchanged. Inputs are validated on load: shapes, finite values, positive
weights and unique IDs. Oracles can add their own checks; the tau example
verifies that the MC tau directions agree with the target offsets.

## Method

One policy update for B events:

1. **Candidates.** Draw K completed DDIM samples per event from the current
   policy, each conditioned on the same observed event.
2. **Hybrid reward.** A classifier trained to separate MC targets from
   current-policy samples gives each candidate a logit sᵢ; its weights are
   frozen during policy updates. Your oracle gives each candidate a distance to
   the MC truth, and the vote goes to the closest candidate (first minimum on a
   tie). The reward mixes both in probability space:

   rᵢ = log( 0.7 · softmax(s)ᵢ + 0.3 · 1[i = winner] )

3. **Advantage.** Unscaled leave-one-out: Aᵢ = rᵢ − mean_{j≠i} rⱼ.
4. **Loss.** Draw a diffusion time t ~ U[0, 0.7] and noise ε per event, shared
   by all candidates of the event, the policy and the frozen round reference.
   With Lᵢ the velocity-space denoising error of candidate i:

   loss = mean_{e,i}( w_e · A_{ie} · L_{ie} ) + 1.0 · ½‖v_θ − v_ref‖²,
   w_e = stopgrad σ( (β/K) Σᵢ A_{ie} (L_{ie} − L^{ref}_{ie}) ),  β = 1

   Six (t, ε) draws are averaged into one AdamW step, with constant learning
   rate 5×10⁻⁵ and gradient clipping at 1.0.

**Classifier rounds.** The classifier is fitted before the first update and
again every 500 updates, always on the same population: a fixed set of training
events, split once into fit and validation events (the paper uses the whole
training set). Each refit regenerates one current-policy sample per population
event. Each batch pairs every event's MC target (label 1) with its policy
sample (label 0), and the binary cross-entropy is weighted by the MC event
weight. Candidate features are standardized with the mean and standard
deviation of the first fit's targets and samples, fixed for the whole run.

The classifier sees the frozen supervised policy's view of the event and of
each candidate (for EveNet: the normalized inputs, and the network's features
of the clean candidate at t = 0), plus any extra features you configure. Each
fit keeps that frozen backbone and starts a fresh head from the same seed:
FiLM layers at identity and a zero final readout, so every candidate starts
with equal weight. The fit keeps the checkpoint with the best validation BCE
and stops at a plateau: at least 1,000 updates, patience 25 epochs, minimum
improvement 10⁻⁴, with a cosine learning rate from 10⁻⁴ to 10⁻⁵ over 1,500
epochs. A fit that never plateaus is not installed.

After each new classifier is installed, the round reference is reset to the
current policy. Policy batches walk through the training set in shuffled
epochs. The raw policy is validated at step 0 and every 50 updates with one
DDIM20 draw per event, before any refit due at the same step.

## Outputs and resuming

Each run writes only to `results/<experiment>/<run_id>/`. The `run_id`
combines the stage, a hash of every setting and input file that can change the
output, and the seed. A DGPO run directory contains:

```
config.yaml, provenance.json   resolved config; seed, git commit, dirty flag, package versions, loaded modules
metrics.jsonl                  validation, classifier-round and per-update diagnostics
setup.pt                       oracle constants and pinned classifier feature standardization
validation/truth.npz           validation MC targets and event IDs
validation/step_XXXXXXX.npz    one raw DDIM draw per validation event
classifier/round_XX.pt         installed classifier heads
policy_final.pt                final policy weights
COMPLETE                       written last
```

A run is built in `results/<experiment>/.partial/` and renamed into place after
its `COMPLETE` marker is written. A finished run is skipped. Nothing outside
this package's own outputs is ever overwritten: `run.py` refuses a `--results`
directory that holds files it did not create, the converters refuse an
existing output file, and `summarize.py` refuses an output directory holding
anything other than an earlier summary.

DGPO runs resume after an interruption such as a preempted job. Every
`dgpo.checkpoint_every` updates, and at every refit, the run saves
`checkpoint/state.pt`: the policy, its optimizer, the round reference, the
installed classifier head and feature scale, and the round index. During a
classifier fit, `checkpoint/fit.pt` is updated every epoch with the head,
optimizer, best checkpoint, plateau counters and random-number state.
Rerunning the same command continues from these files; an interrupted fit
continues within its round. Records written after the last checkpoint are
discarded and redone, so a resumed run ends in exactly the state of an
uninterrupted one (a test checks this for interruptions between updates and
in the middle of a fit). `provenance.json` lists the steps at which a run
resumed and the sha256 digests of the supervised checkpoint, its YAML and its
normalization file, which are also part of the run hash. The reference MLP's
own supervised run has no checkpoint and is redone from scratch.

## Reproducibility

Every random stream is keyed by its position in the run: each policy update,
each data epoch and each refit has its own seed. The classifier fit uses one
seed for every round, and validation reuses the same prior noise at every
step. A refit therefore never shifts the policy's random numbers, and a
resumed run needs no saved generator state except the fit's dropout state.

The same config and seed give bitwise-identical results on CPU.
`scripts/run.py` sets `MKL_CBWR=AUTO,STRICT` so that MKL kernels do not depend
on memory alignment. Bitwise reproducibility on GPU is not guaranteed, and no
tolerance has been established yet.

## Repository structure and tests

```
scripts/       entry points and the SLURM template; no logic
evenet_align/
  experiments/ thin shell: parse config, load events, policy and your functions, write run directories
  core/        algorithm, domain types and the EveNet / MLP policies; pure (no file I/O,
               randomness only via explicit seeds)
  examples/    the paper's oracles and classifier features (Z->tautau, ttbar), custom-policy template
  analysis/    reads completed run directories only
configs/       smoke.yaml, ttbar.yaml and ztautau.yaml: the only switch between runs
tests/         unit tests, end-to-end checks and the optional EveNet test
```

`python -m pytest` and `lint-imports` enforce the layer contracts, no file I/O
or printing in `core/`, no mode branches, determinism, kill-and-resume to the
same final state (between updates and during a fit), provenance fields, and
agreement between the npz and pseudo sources. CI runs both on every push.

The EveNet test needs `evenet-core` and an EveNet `share/` directory from the
EveNet-Full repository:

```sh
EVENET_SHARE=<EveNet-Full>/share python -m pytest -m evenet
```

It builds a tiny random EveNet "fine-tune" with EveNet's own config loader and
runs DGPO on it twice to check the plumbing and reproducibility; it says
nothing about physics.

## Citation

If you use this code, please cite:

```bibtex
@article{TODO,
  title   = {TODO},
  author  = {TODO},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

Please also cite [EveNet](https://github.com/EveNet-HEP/EveNet-Full), which
provides the supervised model.

## License

This project is released under the [MIT License](LICENSE).
