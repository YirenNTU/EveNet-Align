import math

import torch

from evenet_align.core.reward import hybrid_reward, leave_one_out_advantage, vote_winner


def test_hybrid_reward_is_log_of_probability_mixture():
    g = torch.Generator().manual_seed(0)
    logits, distance = torch.randn(6, 50, generator=g), torch.rand(6, 50, generator=g)
    reward = hybrid_reward(logits, distance, 0.3)
    winner = distance.argmin(0)
    onehot = torch.zeros(6, 50, dtype=torch.float64).scatter_(0, winner[None], 1.0)
    expected = torch.log(0.7 * logits.double().softmax(0) + 0.3 * onehot)
    assert torch.allclose(reward, expected, atol=1e-12)
    assert torch.allclose(reward.exp().sum(0), torch.ones(50, dtype=torch.float64))


def test_zero_oracle_fraction_is_classifier_only():
    logits = torch.randn(4, 7)
    assert torch.allclose(hybrid_reward(logits, torch.rand(4, 7), 0.0), logits.double().log_softmax(0))


def test_vote_goes_to_first_minimum_on_tie():
    distance = torch.tensor([[0.5], [0.1], [0.1], [0.3]])
    assert int(vote_winner(distance)) == 1
    reward = hybrid_reward(torch.zeros(4, 1), distance, 0.3)
    assert math.isclose(float(reward[1, 0]), math.log(0.7 / 4 + 0.3))
    assert math.isclose(float(reward[2, 0]), math.log(0.7 / 4))


def test_leave_one_out_advantage():
    reward = torch.tensor([[1.0], [2.0], [6.0]])
    expected = torch.tensor([[1 - 4.0], [2 - 3.5], [6 - 1.5]])
    assert torch.allclose(leave_one_out_advantage(reward), expected)
