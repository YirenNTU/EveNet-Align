import torch

from evenet_align.core.classifier import FitData, binary_auc, fit_classifier
from evenet_align.core.models import build_classifier_head
from evenet_align.core.types import ClassifierConfig


def test_fresh_head_scores_every_candidate_equally():
    head = build_classifier_head(16, 5, 0.1, 30.0, seed=0).eval()
    logits = head(torch.randn(9, 16), torch.randn(9, 5))
    assert torch.equal(logits, torch.zeros(9))


def test_logit_is_bounded_by_log_ratio_bound():
    head = build_classifier_head(4, 3, 0.0, 30.0, seed=0).eval()
    torch.nn.init.constant_(head.readout.bias, 1e3)
    assert torch.allclose(head(torch.randn(2, 4), torch.randn(2, 3)), torch.full((2,), float(torch.log(torch.tensor(30.0)))))


def test_auc():
    labels = torch.tensor([1, 1, 0, 0, 1, 0])
    assert binary_auc(torch.tensor([0.9, 0.8, 0.1, 0.2, 0.7, 0.3]), labels) == 1.0
    assert binary_auc(torch.zeros(6), labels) == 0.5
    assert abs(binary_auc(torch.tensor([0.9, 0.1, 0.2, 0.8, 0.5, 0.5]), labels) - 4.5 / 9) < 1e-12


def _data(g, n):
    return FitData(torch.randn(n, 8, generator=g), torch.randn(n, 3, generator=g) + 1.0,
                   torch.randn(n, 3, generator=g), torch.ones(n))


CFG = ClassifierConfig(batch_size=256, learning_rate=1e-3, min_learning_rate=1e-4, lr_decay_epochs=50,
                       max_epochs=200, min_updates=20, patience=5)


def _run(train, val, resume=None):
    *states, result = list(fit_classifier(train, val, CFG, seed=0, resume=resume))
    return states, result


def test_fit_separates_and_reaches_plateau():
    g = torch.Generator().manual_seed(0)
    _, result = _run(_data(g, 2000), _data(g, 500))
    assert result.plateau and result.validation_auc > 0.8 and result.validation_bce < 0.6


def test_resumed_fit_equals_uninterrupted_fit():
    g = torch.Generator().manual_seed(1)
    train, val = _data(g, 2000), _data(g, 500)
    states, result = _run(train, val)
    _, resumed = _run(train, val, resume=states[2])
    assert resumed.epochs == result.epochs and resumed.updates == result.updates
    assert resumed.validation_bce == result.validation_bce
    for k, v in result.head.state_dict().items():
        assert torch.equal(v, resumed.head.state_dict()[k]), k
    _, finished = _run(train, val, resume=states[-1])  # resuming a finished fit only selects
    assert finished.validation_bce == result.validation_bce
