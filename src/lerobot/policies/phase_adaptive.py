#!/usr/bin/env python

# Copyright 2026 Jay Ranpariya. Thesis extension.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared phase-adaptive execution utilities for chunked policies.

The utilities in this module intentionally operate on already-normalized
policy inputs. The grasp detector uses ratios within the observed gripper
excursion, rather than absolute servo coordinates, so its open/close logic
is invariant to ACT's mean/std versus Diffusion Policy's min/max affine scale.
The minimum arming excursion remains a normalization-domain configuration.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PhaseUpdate:
    """Result of one causal gripper-phase update."""

    post_grasp: bool
    changed: bool


class GripperCyclePhaseDetector:
    """Detect and latch a completed open-then-close grasp cycle.

    The SO-101 demonstrations begin with the gripper near its closed baseline,
    open it around the object, and close it to a held-object aperture. The
    detector locks onto whichever opening direction first exceeds a minimum
    excursion from the initial value and enters ``post_grasp`` after the
    gripper retreats by ``close_fraction`` of that excursion for
    ``stable_steps`` consecutive observations. The post-grasp state is an
    episode latch: reopening for an intentional release does not change the
    controller. Only :meth:`reset` starts a new pre-grasp episode.

    Dynamic execution is a rollout-only facility and currently supports the
    single-environment batch used by ``lerobot-rollout``. The opening polarity
    is locked once per grasp attempt and validated by exact replay in the
    external human/gold calibration artifact. Rejecting larger
    batches avoids silently sharing one action queue across different phases.
    """

    def __init__(
        self,
        gripper_index: int = 5,
        min_open_excursion: float = 0.25,
        close_fraction: float = 0.5,
        stable_steps: int = 15,
        stable_delta_fraction: float = 0.02,
    ) -> None:
        if gripper_index < 0:
            raise ValueError("gripper_index must be non-negative")
        if min_open_excursion <= 0:
            raise ValueError("min_open_excursion must be positive")
        if not 0 < close_fraction < 1:
            raise ValueError("close_fraction must be between 0 and 1")
        if stable_steps <= 0:
            raise ValueError("stable_steps must be positive")
        if stable_delta_fraction <= 0:
            raise ValueError("stable_delta_fraction must be positive")

        self.gripper_index = gripper_index
        self.min_open_excursion = min_open_excursion
        self.close_fraction = close_fraction
        self.stable_steps = stable_steps
        self.stable_delta_fraction = stable_delta_fraction
        self.reset()

    def reset(self) -> None:
        self._baseline: Tensor | None = None
        self._peak_excursion: Tensor | None = None
        self._opening_direction: int | None = None
        self._post_grasp = False
        self._candidate_steps = 0
        self._previous: Tensor | None = None

    @property
    def post_grasp(self) -> bool:
        return self._post_grasp

    def update(self, state: Tensor) -> PhaseUpdate:
        if state.ndim != 2 or state.shape[0] != 1:
            raise ValueError(
                "phase-adaptive execution requires state shape (1, state_dim); "
                f"got {tuple(state.shape)}"
            )
        if self.gripper_index >= state.shape[-1]:
            raise ValueError(
                f"gripper_index={self.gripper_index} is outside state dim {state.shape[-1]}"
            )

        value = state[0, self.gripper_index].detach()
        if self._baseline is None:
            self._baseline = value.clone()
            self._peak_excursion = torch.zeros_like(value)
            self._previous = value.clone()
            return PhaseUpdate(post_grasp=False, changed=False)

        assert self._peak_excursion is not None
        positive_excursion = value - self._baseline
        negative_excursion = self._baseline - value
        if self._opening_direction is None:
            largest = torch.maximum(positive_excursion, negative_excursion)
            if bool(largest >= self.min_open_excursion):
                self._opening_direction = 1 if bool(positive_excursion >= negative_excursion) else -1
        direction = self._opening_direction or 1
        oriented_value = direction * (value - self._baseline)
        if self._opening_direction is not None:
            self._peak_excursion = torch.maximum(self._peak_excursion, oriented_value)
        excursion = self._peak_excursion
        assert self._previous is not None
        stable = bool((value - self._previous).abs() <= self.stable_delta_fraction * excursion)
        changed = False

        if not self._post_grasp:
            candidate = bool(
                self._opening_direction is not None
                and excursion >= self.min_open_excursion
                and oriented_value <= self.close_fraction * excursion
                and stable
            )
            self._candidate_steps = self._candidate_steps + 1 if candidate else 0
            if self._candidate_steps >= self.stable_steps:
                self._post_grasp = True
                self._candidate_steps = 0
                changed = True
        self._previous = value.clone()
        return PhaseUpdate(post_grasp=self._post_grasp, changed=changed)


@dataclass
class _ChunkPrediction:
    actions: Tensor
    age: int = 0


class ChunkTemporalEnsembler:
    """Blend aligned predictions from overlapping action chunks.

    Unlike ACT's optimized replan-every-step implementation, this class also
    supports policies that replan every ``R > 1`` control steps. Each retained
    chunk carries its age in control steps. At a replan, future action ``j`` is
    blended from each chunk at index ``age + j``. Predictions whose age reaches
    the chunk length are excluded.
    """

    def __init__(
        self,
        temporal_ensemble_coeff: float,
        chunk_size: int,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.temporal_ensemble_coeff = temporal_ensemble_coeff
        self.chunk_size = chunk_size
        self.reset()

    def reset(self) -> None:
        self._history: deque[_ChunkPrediction] = deque()

    def _retained(self) -> list[_ChunkPrediction]:
        return [entry for entry in self._history if entry.age < self.chunk_size]

    def update(self, actions: Tensor, output_steps: int) -> Tensor:
        if actions.ndim != 3 or actions.shape[1] != self.chunk_size:
            raise ValueError(
                "actions must have shape (batch, chunk_size, action_dim); got "
                f"{tuple(actions.shape)} with chunk_size={self.chunk_size}."
            )
        if not 1 <= output_steps <= self.chunk_size:
            raise ValueError(f"output_steps must be between 1 and {self.chunk_size}")

        self._history.append(_ChunkPrediction(actions=actions.clone()))
        retained = self._retained()
        self._history = deque(retained)
        outputs = []

        for future_offset in range(output_steps):
            eligible = [
                entry
                for entry in retained
                if entry.age + future_offset < self.chunk_size
            ]
            if not eligible:
                raise RuntimeError("no chunk predicts a requested output action")
            predictions = torch.stack(
                [entry.actions[:, entry.age + future_offset] for entry in eligible], dim=1
            )
            weights = torch.exp(
                -self.temporal_ensemble_coeff
                * torch.arange(len(eligible), device=actions.device, dtype=actions.dtype)
            ).view(1, -1, 1)
            outputs.append((predictions * weights).sum(dim=1) / weights.sum())

        return torch.stack(outputs, dim=1)

    def advance(self, steps: int = 1) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        for entry in self._history:
            entry.age += steps
        self._history = deque(self._retained())
