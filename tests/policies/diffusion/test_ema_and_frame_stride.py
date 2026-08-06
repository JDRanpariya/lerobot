"""Contracts for strided real-robot DP training and resumable EMA weights."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


def _tiny_ema_policy() -> DiffusionPolicy:
    policy = DiffusionPolicy.__new__(DiffusionPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        ema_update_after_step=0,
        ema_inv_gamma=1.0,
        ema_power=0.75,
        ema_min_decay=0.0,
        ema_max_decay=0.9999,
        ema_use_for_inference=False,
    )
    policy.diffusion = nn.Linear(2, 1, bias=False)
    nn.init.constant_(policy.diffusion.weight, 1.0)
    policy.ema_diffusion = deepcopy(policy.diffusion)
    policy.ema_diffusion.requires_grad_(False)
    policy.register_buffer("ema_optimization_step", torch.zeros((), dtype=torch.long))
    return policy


def test_frame_stride_scales_observation_action_and_padding_indices():
    config = DiffusionConfig(frame_stride=3)
    assert config.observation_delta_indices == [-3, 0]
    assert config.action_delta_indices == list(range(-3, 43, 3))
    assert len(config.action_delta_indices) == config.horizon
    assert config.drop_n_last_frames == 21


def test_frame_stride_must_be_positive():
    with pytest.raises(ValueError, match="frame_stride"):
        DiffusionConfig(frame_stride=0)


def test_ema_inference_requires_persisted_ema_model():
    with pytest.raises(ValueError, match="requires.*use_ema"):
        DiffusionConfig(use_ema=False, ema_use_for_inference=True)


def test_first_ema_update_copies_online_weights_then_uses_warmup_decay():
    policy = _tiny_ema_policy()
    policy.diffusion.weight.data.fill_(3.0)
    policy.update()
    torch.testing.assert_close(policy.ema_diffusion.weight, policy.diffusion.weight)
    assert policy.ema_optimization_step.item() == 1

    policy.diffusion.weight.data.fill_(5.0)
    expected_decay = 1.0 - 2.0**-0.75
    policy.update()
    expected = 3.0 * expected_decay + 5.0 * (1.0 - expected_decay)
    torch.testing.assert_close(
        policy.ema_diffusion.weight,
        torch.full_like(policy.ema_diffusion.weight, expected),
    )


def test_ema_state_and_update_count_are_checkpoint_resumable():
    trained = _tiny_ema_policy()
    trained.diffusion.weight.data.fill_(7.0)
    trained.update()
    trained.diffusion.weight.data.fill_(9.0)
    trained.update()

    resumed = _tiny_ema_policy()
    resumed.load_state_dict(trained.state_dict(), strict=True)
    assert resumed.ema_optimization_step.item() == 2
    torch.testing.assert_close(resumed.diffusion.weight, trained.diffusion.weight)
    torch.testing.assert_close(resumed.ema_diffusion.weight, trained.ema_diffusion.weight)

    resumed.config.ema_use_for_inference = True
    assert resumed.inference_model is resumed.ema_diffusion
    resumed.config.ema_use_for_inference = False
    assert resumed.inference_model is resumed.diffusion
