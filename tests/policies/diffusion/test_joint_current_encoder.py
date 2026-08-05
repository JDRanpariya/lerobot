"""Focused contracts for the thesis DP learned-current path."""

from collections import deque
from types import SimpleNamespace

import torch

from lerobot.datasets.temporal_window import TemporalWindowDataset
from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionJointCurrentEncoder,
    DiffusionModel,
    DiffusionPolicy,
)


def _config(pooling: str = "mean"):
    return SimpleNamespace(
        proprio_temporal_encoder="joint_cnn",
        proprio_K=9,
        proprio_current_indices=[6, 7, 8, 9, 10, 11],
        proprio_cnn_channels=[4, 4],
        proprio_cnn_kernel_sizes=[3, 3],
        proprio_cnn_dilations=[1, 2],
        proprio_cnn_pooling=pooling,
        proprio_cnn_embedding_dim=3,
        n_obs_steps=2,
        image_features={},
        env_state_feature=None,
    )


def test_joint_current_cnn_shape_and_joint_isolation():
    encoder = DiffusionJointCurrentEncoder(_config())
    for module in encoder.cnn:
        if isinstance(module, torch.nn.Conv1d):
            torch.nn.init.constant_(module.weight, 0.1)
            torch.nn.init.constant_(module.bias, 0.1)
    torch.nn.init.constant_(encoder.projection.weight, 0.1)
    torch.nn.init.constant_(encoder.projection.bias, 0.0)

    baseline = torch.zeros(2, 2, 10, 6)
    changed = baseline.clone()
    changed[:, :, :, 3] = 1.0
    base_embedding = encoder(baseline)
    changed_embedding = encoder(changed)

    assert base_embedding.shape == (2, 2, 6, 3)
    torch.testing.assert_close(
        changed_embedding[:, :, [0, 1, 2, 4, 5]], base_embedding[:, :, [0, 1, 2, 4, 5]]
    )
    assert not torch.equal(changed_embedding[:, :, 3], base_embedding[:, :, 3])


def test_joint_current_cnn_contact_pooling_retains_peak_and_latest():
    encoder = DiffusionJointCurrentEncoder(_config("mean_max_latest"))
    output = encoder(torch.randn(1, 2, 10, 6))
    assert output.shape == (1, 2, 6, 3)
    assert encoder.projection.in_features == 12  # 3 summaries x final CNN width 4


def test_dp_conditioning_has_no_raw_current_bypass():
    config = _config()
    current_encoder = DiffusionJointCurrentEncoder(config)
    stub = SimpleNamespace(
        config=config,
        current_encoder=current_encoder,
        position_indices=[0, 1, 2, 3, 4, 5],
    )
    state = torch.randn(1, 2, 12)
    window = torch.randn(1, 2, 10, 6)
    batch = {"observation.state": state, "observation.state_window": window}
    reference = DiffusionModel._prepare_global_conditioning(stub, batch)

    raw_current_changed = dict(batch)
    raw_current_changed["observation.state"] = state.clone()
    raw_current_changed["observation.state"][..., 6:] += 1000
    torch.testing.assert_close(
        DiffusionModel._prepare_global_conditioning(stub, raw_current_changed), reference
    )

    current_history_changed = dict(batch)
    current_history_changed["observation.state_window"] = window.clone()
    current_history_changed["observation.state_window"][..., 2] += 1
    assert not torch.equal(
        DiffusionModel._prepare_global_conditioning(stub, current_history_changed), reference
    )


def test_dp_online_history_matches_two_observation_contract():
    config = _config()
    policy = SimpleNamespace(
        config=config,
        _current_history=deque(maxlen=config.proprio_K + config.n_obs_steps),
    )
    state = torch.arange(12, dtype=torch.float32).unsqueeze(0)
    first = DiffusionPolicy._update_current_history(policy, state)
    assert first.shape == (1, 2, 10, 6)
    torch.testing.assert_close(first[:, 0], torch.zeros_like(first[:, 0]))
    torch.testing.assert_close(first[:, 1, -1], state[:, 6:])

    second_state = state + 100
    second = DiffusionPolicy._update_current_history(policy, second_state)
    torch.testing.assert_close(second[:, 0, -1], state[:, 6:])
    torch.testing.assert_close(second[:, 1, -2], state[:, 6:])
    torch.testing.assert_close(second[:, 1, -1], second_state[:, 6:])


class _TinyBaseDataset:
    def __init__(self):
        self.rows = []
        for frame in range(3):
            state = torch.zeros(12)
            state[6:] = frame + 1
            self.rows.append(
                {
                    "observation.state": state,
                    "episode_index": torch.tensor(0),
                    "frame_index": torch.tensor(frame),
                }
            )
        self.hf_dataset = self.rows
        self.episode_data_index = {"from_index": torch.tensor([0]), "to_index": torch.tensor([3])}
        self.meta = SimpleNamespace(
            stats={"observation.state": {"min": torch.zeros(12), "max": torch.full((12,), 4.0)}}
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return dict(self.rows[index])


def test_dp_dataset_windows_are_per_observation_and_minmax_normalized():
    dataset = TemporalWindowDataset(
        _TinyBaseDataset(),
        K=1,
        state_indices=[6, 7, 8, 9, 10, 11],
        normalization_mode="MIN_MAX",
        observation_steps=2,
    )
    window = dataset[2]["observation.state_window"]
    assert window.shape == (2, 2, 6)
    # For observation t-1: [current(t-2), current(t-1)] = [1, 2] -> [-.5, 0].
    torch.testing.assert_close(window[0, :, 0], torch.tensor([-0.5, 0.0]))
    # For observation t: [2, 3] -> [0, .5].
    torch.testing.assert_close(window[1, :, 0], torch.tensor([0.0, 0.5]))
