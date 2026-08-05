"""Contracts for phase-adaptive chunk execution shared by ACT and DP."""

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.phase_adaptive import (
    ChunkTemporalEnsembler,
    GripperCyclePhaseDetector,
    PhaseUpdate,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def test_chunk_ensemble_aligns_predictions_by_control_step_age():
    ensemble = ChunkTemporalEnsembler(0.0, chunk_size=8, temporal_ensemble_window=8)
    first = torch.arange(8, dtype=torch.float32).view(1, 8, 1)
    torch.testing.assert_close(ensemble.update(first, output_steps=4), first[:, :4])
    ensemble.advance(4)

    second = (100 + torch.arange(8, dtype=torch.float32)).view(1, 8, 1)
    output = ensemble.update(second, output_steps=4)
    expected = torch.tensor([52.0, 53.0, 54.0, 55.0]).view(1, 4, 1)
    torch.testing.assert_close(output, expected)


def test_switching_chunk_window_discards_pre_transition_predictions():
    ensemble = ChunkTemporalEnsembler(0.01, chunk_size=8, temporal_ensemble_window=8)
    ensemble.update(torch.ones(1, 8, 1), output_steps=4)
    ensemble.advance(2)
    ensemble.set_window(4)
    post = 20 * torch.ones(1, 8, 1)
    torch.testing.assert_close(ensemble.update(post, output_steps=4), post[:, :4])


def test_act_episode_reset_restores_pre_grasp_ensemble_parameters():
    class _Ensembler:
        def __init__(self):
            self.parameters = None
            self.was_reset = False

        def set_runtime_parameters(self, **parameters):
            self.parameters = parameters

        def reset(self):
            self.was_reset = True

    class _PhaseDetector:
        def __init__(self):
            self.was_reset = False

        def reset(self):
            self.was_reset = True

    policy = SimpleNamespace(
        config=SimpleNamespace(
            temporal_ensemble_coeff=0.01,
            temporal_ensemble_window=None,
            proprio_temporal_encoder="none",
            proprio_K=0,
        ),
        temporal_ensembler=_Ensembler(),
        temporal_ensemble_phase_detector=_PhaseDetector(),
    )

    ACTPolicy.reset(policy)

    assert policy.temporal_ensembler.parameters == {
        "temporal_ensemble_coeff": 0.01,
        "temporal_ensemble_window": None,
    }
    assert policy.temporal_ensembler.was_reset
    assert policy.temporal_ensemble_phase_detector.was_reset


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_gripper_detector_locks_either_opening_polarity(direction):
    detector = GripperCyclePhaseDetector(
        min_open_excursion=0.2,
        stable_steps=2,
        stable_delta_fraction=0.05,
    )
    for value in [0.0, 0.3 * direction, 0.8 * direction, 1.0 * direction, 0.4 * direction]:
        detector.update(torch.tensor([[0, 0, 0, 0, 0, value]], dtype=torch.float32))
    assert not detector.post_grasp
    detector.update(torch.tensor([[0, 0, 0, 0, 0, 0.39 * direction]], dtype=torch.float32))
    update = detector.update(torch.tensor([[0, 0, 0, 0, 0, 0.38 * direction]], dtype=torch.float32))
    assert update.changed and update.post_grasp


class _Detector:
    def __init__(self, updates):
        self.updates = iter(updates)

    def update(self, _state):
        return next(self.updates)


def _policy_stub(chunks, detector_updates=None, deadline=None):
    config = SimpleNamespace(
        image_features={},
        proprio_temporal_encoder="none",
        temporal_ensemble_window=8,
        temporal_ensemble_replan_steps=4,
        temporal_ensemble_post_grasp_window=4,
        temporal_ensemble_post_grasp_replan_steps=2,
        temporal_ensemble_inference_deadline_ms=deadline,
        n_action_steps=8,
    )
    iterator = iter(chunks)
    stub = SimpleNamespace(
        config=config,
        _queues={OBS_STATE: deque(maxlen=2), ACTION: deque(maxlen=8)},
        temporal_ensembler=ChunkTemporalEnsembler(0.0, 8, 8),
        temporal_ensemble_phase_detector=(
            _Detector(detector_updates) if detector_updates is not None else None
        ),
        _active_replan_steps=4,
        predict_action_chunk=lambda _batch, noise=None: next(iterator),
        _last_inference_ms=None,
    )
    return stub


def test_dp_controller_replans_in_coherent_age_aligned_bursts():
    first = torch.arange(8, dtype=torch.float32).view(1, 8, 1)
    second = (100 + torch.arange(8, dtype=torch.float32)).view(1, 8, 1)
    policy = _policy_stub([first, second])
    batch = {OBS_STATE: torch.zeros(1, 6)}

    output = [DiffusionPolicy.select_action(policy, dict(batch)) for _ in range(5)]
    torch.testing.assert_close(torch.stack(output[:4]).flatten(), torch.arange(4.0))
    # The old chunk's index 4 is aligned with the new chunk's index 0.
    torch.testing.assert_close(output[4], torch.tensor([[52.0]]))


def test_dp_phase_transition_clears_fifo_before_selecting_transition_action():
    first = torch.arange(8, dtype=torch.float32).view(1, 8, 1)
    post = (100 + torch.arange(8, dtype=torch.float32)).view(1, 8, 1)
    policy = _policy_stub(
        [first, post],
        detector_updates=[
            PhaseUpdate(post_grasp=False, changed=False),
            PhaseUpdate(post_grasp=True, changed=True),
        ],
    )
    batch = {OBS_STATE: torch.zeros(1, 6)}

    torch.testing.assert_close(DiffusionPolicy.select_action(policy, dict(batch)), torch.tensor([[0.0]]))
    # Without the atomic clear this would be action 1 from the first chunk.
    torch.testing.assert_close(DiffusionPolicy.select_action(policy, dict(batch)), torch.tensor([[100.0]]))
    assert policy._active_replan_steps == 2


def test_dp_controller_aborts_before_queueing_actions_after_deadline(monkeypatch):
    chunk = torch.zeros(1, 8, 1)
    policy = _policy_stub([chunk], deadline=100.0)
    clock = iter([1.0, 1.2])
    monkeypatch.setattr(
        "lerobot.policies.diffusion.modeling_diffusion.time.perf_counter",
        lambda: next(clock),
    )

    with pytest.raises(RuntimeError, match="missed the development deadline"):
        DiffusionPolicy.select_action(policy, {OBS_STATE: torch.zeros(1, 6)})
    assert len(policy._queues[ACTION]) == 0
