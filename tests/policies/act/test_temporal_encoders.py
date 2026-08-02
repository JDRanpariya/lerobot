"""Tests for thesis-specific ACT proprioceptive temporal encoders."""

from types import SimpleNamespace

import torch

from lerobot.policies.act.temporal_encoders import ExplicitFeatureEncoder


def test_explicit_encoder_persists_gravity_baseline():
    config = SimpleNamespace(
        proprio_current_indices=[6, 7, 8, 9, 10, 11],
        proprio_explicit_features=["raw", "residual"],
        proprio_K=9,
    )
    encoder = ExplicitFeatureEncoder(config, state_dim=12)
    expected = torch.arange(6, dtype=torch.float32)
    encoder.gravity_baseline.copy_(expected)

    state = encoder.state_dict()
    assert "gravity_baseline" in state

    restored = ExplicitFeatureEncoder(config, state_dim=12)
    restored.load_state_dict(state, strict=True)
    torch.testing.assert_close(restored.gravity_baseline, expected)

