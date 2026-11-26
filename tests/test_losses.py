import torch

from evenet_align.core.losses import dgpo_loss, reference_anchor_loss


def test_gate_is_detached_and_formula():
    g = torch.Generator().manual_seed(1)
    current = torch.rand(5, 8, generator=g, requires_grad=True)
    reference = torch.rand(5, 8, generator=g)
    advantage = torch.randn(5, 8, generator=g)
    loss, diag = dgpo_loss(current, reference, advantage, beta=2.0)
    gate = torch.sigmoid((2.0 / 5) * (advantage * (current.detach() - reference)).sum(0))
    assert torch.allclose(loss, (gate * advantage * current).mean())
    loss.backward()
    # Gradient through the gate would add terms; detached gate gives exactly w_e * A / (K * B).
    assert torch.allclose(current.grad, gate * advantage / current.numel())


def test_anchor_loss_gradient_only_through_current():
    current = torch.randn(4, 3, requires_grad=True)
    reference = torch.randn(4, 3, requires_grad=True)
    reference_anchor_loss(current, reference).backward()
    assert reference.grad is None
    assert torch.allclose(current.grad, (current - reference).detach() / current.numel())
