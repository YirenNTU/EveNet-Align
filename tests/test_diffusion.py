import torch

from evenet_align.core.diffusion import alpha_sigma, ddim_sample, logsnr_cosine, noise_and_target
from evenet_align.core.types import make_generator


def test_schedule_endpoints_and_variance_preservation():
    t = torch.linspace(0, 1, 11, dtype=torch.float64)
    assert torch.allclose(logsnr_cosine(t[[0, -1]]), torch.tensor([20.0, -20.0], dtype=torch.float64))
    alpha, sigma = alpha_sigma(t)
    assert torch.allclose(alpha ** 2 + sigma ** 2, torch.ones_like(t))


def test_ddim_recovers_x0_with_oracle_velocity():
    """With the exact velocity of a point mass, deterministic DDIM lands on that point."""
    x0 = torch.tensor([[0.3, -1.2]], dtype=torch.float64).expand(5, 2)

    def velocity(x, t):
        alpha, sigma = (s[:, None] for s in alpha_sigma(t))
        eps = (x - alpha * x0) / sigma
        return alpha * eps - sigma * x0

    g = make_generator(0)
    out = ddim_sample(lambda x, t: velocity(x.double(), t.double()), 5, 2, 20, g)
    assert torch.allclose(out.double(), x0, atol=1e-4)


def test_noise_and_target_inverts():
    x0, eps, t = torch.randn(7, 3), torch.randn(7, 3), torch.rand(7)
    x_t, v = noise_and_target(x0, t, eps)
    alpha, sigma = (s[:, None] for s in alpha_sigma(t))
    assert torch.allclose(alpha * x_t - sigma * v, x0, atol=1e-5)
