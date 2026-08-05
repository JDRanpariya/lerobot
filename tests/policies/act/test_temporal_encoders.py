"""Tests for thesis-specific ACT proprioceptive temporal encoders."""

from types import SimpleNamespace

import torch

from lerobot.policies.act.diagnostics import _needs_history_window
from lerobot.policies.act.fusion_modules import TokenFusion
from lerobot.policies.act.temporal_encoders import (
    ExplicitFeatureEncoder,
    JointCNNTemporalEncoder,
    JointTokenEncoder,
)


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


def _joint_config(pooling="mean"):
    return SimpleNamespace(
        proprio_current_indices=[6, 7, 8, 9, 10, 11],
        proprio_K=9,
        proprio_cnn_channels=[4, 4, 2],
        proprio_cnn_kernel_sizes=[3, 3, 3],
        proprio_cnn_dilations=[1, 2, 4],
        proprio_cnn_pooling=pooling,
    )


def test_joint_token_encoder_preserves_joint_pairing_and_removes_current_bypass():
    encoder = JointTokenEncoder(_joint_config(), state_dim=12)
    positions = torch.arange(6, dtype=torch.float32).unsqueeze(0)
    currents = (10 + torch.arange(6, dtype=torch.float32)).unsqueeze(0)

    output = encoder({"observation.state": torch.cat((positions, currents), dim=-1)})

    assert output["observation.state"].shape == (1, 6)
    assert output["proprio_embedding"].shape == (1, 6, 2)
    torch.testing.assert_close(output["observation.state"], positions)
    torch.testing.assert_close(output["proprio_embedding"][0, :, 0], positions[0])
    torch.testing.assert_close(output["proprio_embedding"][0, :, 1], currents[0])
    assert encoder.embedding_tokens() == 6


def test_joint_cnn_keeps_histories_isolated_before_attention():
    encoder = JointCNNTemporalEncoder(_joint_config(), state_dim=12)
    for module in encoder.cnn:
        if isinstance(module, torch.nn.Conv1d):
            torch.nn.init.constant_(module.weight, 0.1)
            torch.nn.init.constant_(module.bias, 0.1)

    state = torch.zeros(1, 12)
    baseline_window = torch.zeros(1, 60)
    changed_window = baseline_window.view(1, 10, 6).clone()
    changed_window[:, :, 2] = 1.0
    changed_window = changed_window.reshape(1, 60)

    baseline = encoder(
        {
            "observation.state": state,
            "observation.state_window": baseline_window,
        }
    )["proprio_embedding"]
    changed = encoder(
        {
            "observation.state": state,
            "observation.state_window": changed_window,
        }
    )["proprio_embedding"]

    assert changed.shape == (1, 6, 3)
    torch.testing.assert_close(changed[:, [0, 1, 3, 4, 5]], baseline[:, [0, 1, 3, 4, 5]])
    assert not torch.equal(changed[:, 2], baseline[:, 2])
    assert encoder.embedding_tokens() == 6


def test_joint_cnn_contact_pooling_preserves_mean_max_and_latest_per_joint():
    encoder = JointCNNTemporalEncoder(
        _joint_config(pooling="mean_max_latest"), state_dim=12
    )
    encoder.cnn = torch.nn.Identity()
    encoder.out_dim = 1
    encoder.pool_factor = 3

    positions = torch.arange(6, dtype=torch.float32).unsqueeze(0)
    histories = torch.zeros(1, 10, 6)
    histories[0, 3, 2] = 4.0  # brief J3 contact peak
    histories[0, -1, 4] = 2.0  # latest J5 evidence
    output = encoder(
        {
            "observation.state": torch.cat((positions, torch.zeros_like(positions)), dim=-1),
            "observation.state_window": histories.reshape(1, -1),
        }
    )

    embedding = output["proprio_embedding"]
    assert embedding.shape == (1, 6, 4)
    torch.testing.assert_close(embedding[0, :, 0], positions[0])
    torch.testing.assert_close(embedding[0, 2, 1:], torch.tensor([0.4, 4.0, 0.0]))
    torch.testing.assert_close(embedding[0, 4, 1:], torch.tensor([0.2, 2.0, 2.0]))
    torch.testing.assert_close(embedding[0, [0, 1, 3, 5], 1:], torch.zeros(4, 3))
    assert encoder.embedding_dim() == 4


def test_joint_cnn_missing_pooling_field_preserves_historical_shape():
    legacy = _joint_config()
    del legacy.proprio_cnn_pooling
    encoder = JointCNNTemporalEncoder(legacy, state_dim=12)
    assert encoder.pooling == "mean"
    assert encoder.embedding_dim() == 3


def test_diagnostics_derive_history_requirement_from_encoder_capability():
    joint_tokens = SimpleNamespace(
        model=SimpleNamespace(temporal_encoder=JointTokenEncoder(_joint_config(), state_dim=12))
    )
    joint_cnn = SimpleNamespace(
        model=SimpleNamespace(temporal_encoder=JointCNNTemporalEncoder(_joint_config(), state_dim=12))
    )

    assert not _needs_history_window(joint_tokens)
    assert _needs_history_window(joint_cnn)


def test_token_fusion_exposes_six_addressable_joint_tokens():
    fusion = TokenFusion(SimpleNamespace(dim_model=8), temporal_dim=2, temporal_tokens=6)
    temporal = torch.arange(12, dtype=torch.float32).view(1, 6, 2)
    tokens = fusion(
        latent=torch.zeros(1, 8),
        state=torch.zeros(1, 8),
        vision_tokens=[],
        temporal_features=temporal,
    )

    assert len(tokens) == 8  # latent + position state + J1..J6
    assert fusion.get_extra_pos_embed().shape == (6, 1, 8)
    for joint_index in range(6):
        expected = fusion.temporal_proj(temporal[:, joint_index])
        torch.testing.assert_close(tokens[2 + joint_index], expected)
