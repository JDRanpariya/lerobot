"""Contracts for phase-adaptive chunk execution shared by ACT and DP."""

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.phase_adaptive import (
    ChunkTemporalEnsembler,
    GripperCyclePhaseDetector,
    PhaseUpdate,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def test_chunk_ensemble_aligns_predictions_by_control_step_age():
    ensemble = ChunkTemporalEnsembler(0.0, chunk_size=8)
    first = torch.arange(8, dtype=torch.float32).view(1, 8, 1)
    torch.testing.assert_close(ensemble.update(first, output_steps=4), first[:, :4])
    ensemble.advance(4)

    second = (100 + torch.arange(8, dtype=torch.float32)).view(1, 8, 1)
    output = ensemble.update(second, output_steps=4)
    expected = torch.tensor([52.0, 53.0, 54.0, 55.0]).view(1, 4, 1)
    torch.testing.assert_close(output, expected)


def test_act_episode_reset_restores_pre_grasp_temporal_ensemble():
    class _PhaseDetector:
        def __init__(self):
            self.was_reset = False

        def reset(self):
            self.was_reset = True

    policy = SimpleNamespace(
        config=SimpleNamespace(
            temporal_ensemble_coeff=0.01,
            temporal_ensemble_disable_after_grasp=True,
            chunk_size=8,
            proprio_temporal_encoder="none",
            proprio_K=0,
        ),
        temporal_ensembler=ACTTemporalEnsembler(0.01, 8),
        temporal_ensemble_phase_detector=_PhaseDetector(),
        _post_grasp_open_loop=True,
        _action_queue=deque([torch.ones(1, 1)], maxlen=8),
    )

    ACTPolicy.reset(policy)

    assert not policy._post_grasp_open_loop
    assert len(policy._action_queue) == 0
    assert policy.temporal_ensemble_phase_detector.was_reset


def test_negative_act_ensemble_coefficient_favors_newest_prediction():
    old_chunk = torch.zeros(1, 3, 1)
    new_chunk = 10 * torch.ones(1, 3, 1)
    outputs = {}
    for coefficient in (0.1, 0.0, -0.1):
        ensemble = ACTTemporalEnsembler(coefficient, 3)
        ensemble.update(old_chunk)
        outputs[coefficient] = ensemble.update(new_chunk).item()

    assert outputs[0.1] < outputs[0.0] < outputs[-0.1]


def test_act_switches_from_temporal_ensemble_to_unensembled_chunk_after_grasp():
    chunks = iter(
        [
            torch.arange(8, dtype=torch.float32).view(1, 8, 1),
            (100 + torch.arange(8, dtype=torch.float32)).view(1, 8, 1),
        ]
    )
    detector = _Detector(
        [
            PhaseUpdate(post_grasp=False, changed=False),
            PhaseUpdate(post_grasp=True, changed=True),
            PhaseUpdate(post_grasp=True, changed=False),
        ]
    )
    policy = SimpleNamespace(
        config=SimpleNamespace(
            temporal_ensemble_coeff=0.01,
            temporal_ensemble_disable_after_grasp=True,
            chunk_size=8,
            proprio_temporal_encoder="none",
            proprio_K=0,
        ),
        temporal_ensembler=ACTTemporalEnsembler(0.01, 8),
        temporal_ensemble_phase_detector=detector,
        _post_grasp_open_loop=False,
        _action_queue=deque([], maxlen=8),
        predict_action_chunk=lambda _batch: next(chunks),
        eval=lambda: None,
    )
    batch = {OBS_STATE: torch.zeros(1, 6)}

    torch.testing.assert_close(ACTPolicy.select_action(policy, dict(batch)), torch.tensor([[0.0]]))
    torch.testing.assert_close(ACTPolicy.select_action(policy, dict(batch)), torch.tensor([[100.0]]))
    # No re-query or blending post-grasp: consume the next action from the same chunk.
    torch.testing.assert_close(ACTPolicy.select_action(policy, dict(batch)), torch.tensor([[101.0]]))


def test_act_release_cannot_clear_post_grasp_episode_latch():
    chunks = iter(
        [
            torch.arange(8, dtype=torch.float32).view(1, 8, 1),
            (100 + torch.arange(8, dtype=torch.float32)).view(1, 8, 1),
        ]
    )
    detector = _Detector(
        [
            PhaseUpdate(post_grasp=False, changed=False),
            PhaseUpdate(post_grasp=True, changed=True),
            PhaseUpdate(post_grasp=False, changed=True),
        ]
    )
    policy = SimpleNamespace(
        config=SimpleNamespace(
            temporal_ensemble_coeff=0.01,
            temporal_ensemble_disable_after_grasp=True,
            chunk_size=8,
            proprio_temporal_encoder="none",
            proprio_K=0,
        ),
        temporal_ensembler=ACTTemporalEnsembler(0.01, 8),
        temporal_ensemble_phase_detector=detector,
        _post_grasp_open_loop=False,
        _action_queue=deque([], maxlen=8),
        predict_action_chunk=lambda _batch: next(chunks),
        eval=lambda: None,
    )
    batch = {OBS_STATE: torch.zeros(1, 6)}

    torch.testing.assert_close(ACTPolicy.select_action(policy, dict(batch)), torch.tensor([[0.0]]))
    torch.testing.assert_close(ACTPolicy.select_action(policy, dict(batch)), torch.tensor([[100.0]]))
    # Even a stale detector implementation reporting a reopen cannot discard
    # the post-grasp queue or resume temporal ensembling at release.
    torch.testing.assert_close(ACTPolicy.select_action(policy, dict(batch)), torch.tensor([[101.0]]))
    assert policy._post_grasp_open_loop


def test_act_post_grasp_unensembled_execution_requires_temporal_ensemble():
    with pytest.raises(ValueError, match="requires `temporal_ensemble_coeff`"):
        ACTConfig(
            chunk_size=50,
            n_action_steps=1,
            temporal_ensemble_disable_after_grasp=True,
        )


def test_act_post_grasp_unensembled_execution_requires_reference_coefficient():
    with pytest.raises(ValueError, match="requires `temporal_ensemble_coeff=0.01`"):
        ACTConfig(
            chunk_size=50,
            n_action_steps=1,
            temporal_ensemble_coeff=-0.1,
            temporal_ensemble_disable_after_grasp=True,
        )


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

    # Reopening at release never leaves post-grasp within the same episode.
    for value in [0.5 * direction, 0.9 * direction, 1.0 * direction] * 3:
        update = detector.update(torch.tensor([[0, 0, 0, 0, 0, value]], dtype=torch.float32))
        assert update.post_grasp and not update.changed
    detector.reset()
    assert not detector.post_grasp


class _Detector:
    def __init__(self, updates):
        self.updates = iter(updates)

    def update(self, _state):
        return next(self.updates)


def _policy_stub(chunks, deadline=None):
    config = SimpleNamespace(
        image_features={},
        proprio_temporal_encoder="none",
        temporal_ensemble_replan_steps=4,
        temporal_ensemble_inference_deadline_ms=deadline,
        n_action_steps=8,
    )
    iterator = iter(chunks)
    stub = SimpleNamespace(
        config=config,
        _queues={OBS_STATE: deque(maxlen=2), ACTION: deque(maxlen=8)},
        temporal_ensembler=ChunkTemporalEnsembler(0.0, 8),
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
