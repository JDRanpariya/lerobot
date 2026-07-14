#!/usr/bin/env python

# Copyright 2026 Jay Ranpariya. Thesis extension.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Temporal encoders for proprioceptive current signals in ACT-M variants.

Each encoder transforms a raw observation.state batch into enhanced features
that capture temporal contact dynamics. The output feeds into a FusionModule
that decides how to integrate proprioception with vision.

Design principle: encoders are pure functions over batches with a small
configurable interface:
  - output_state_dim(): dimension of observation.state after forward()
  - has_history_window(): whether the encoder needs observation.state_window
  - produces_embedding(): whether forward adds "proprio_embedding"
  - produces_contact_mask(): whether forward adds "contact_mask"

Note:
  - Encoders keep "observation.state" unchanged unless they explicitly expand it
    (early fusion).  Separable embeddings go into dedicated batch keys so that
    token/film/hybrid fusion can consume them without changing the state token.
"""

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from .configuration_act import ACTConfig


# Encoder capability matrix used by ACTConfig validation and ACT.__init__
TEMPORAL_ENCODER_CAPS = {
    "none":     {"embedding": False, "contact": False, "history": False},
    "history":  {"embedding": False, "contact": False, "history": True},
    "explicit": {"embedding": True,  "contact": False, "history": False},
    "cnn":      {"embedding": True,  "contact": False, "history": True},
    "trigger":  {"embedding": True,  "contact": True,  "history": False},
}


def _central_diff(x: Tensor, dt: float) -> Tensor:
    """Central finite difference along the time dimension (last dim).

    Args:
        x: (..., T) tensor
        dt: timestep in seconds
    Returns:
        (..., T) tensor with boundary values copied
    """
    dx = torch.zeros_like(x)
    dx[..., 1:-1] = (x[..., 2:] - x[..., :-2]) / (2.0 * dt)
    dx[..., 0] = x[..., 1] - x[..., 0]
    dx[..., -1] = x[..., -1] - x[..., -2]
    return dx / dt


class BaseTemporalEncoder(nn.Module):
    """Abstract base for temporal encoders."""

    def __init__(self, config: ACTConfig, state_dim: int):
        super().__init__()
        self.config = config
        self.state_dim = state_dim
        self.current_indices = config.proprio_current_indices
        self.n_current = len(self.current_indices)
        self.n_position = state_dim - self.n_current
        self._dt = 1.0 / 30.0

    def output_state_dim(self) -> int:
        """Return the dimension of observation.state after forward()."""
        raise NotImplementedError

    def has_history_window(self) -> bool:
        """True if this encoder needs observation.state_window in the batch."""
        return False

    def produces_embedding(self) -> bool:
        """True if forward adds batch['proprio_embedding']."""
        return False

    def produces_contact_mask(self) -> bool:
        """True if forward adds batch['contact_mask']."""
        return False

    def embedding_dim(self) -> int | None:
        """Dimension of batch['proprio_embedding'] if produced, else None.

        Fusion modules use this to size their projection layers eagerly (at
        construction time) so the optimizer can see those parameters.
        """
        return None

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError

    def _get_currents(self, state: Tensor) -> Tensor:
        """Extract current channels from state vector."""
        return state[..., self.current_indices]

    def _get_positions(self, state: Tensor) -> Tensor:
        """Extract position channels (indices not in current_indices)."""
        all_idx = set(range(state.shape[-1]))
        pos_idx = sorted(list(all_idx - set(self.current_indices)))
        return state[..., pos_idx]

    def _get_state_window(self, batch: Dict[str, Tensor]) -> Tensor | None:
        """Return (B, (K+1)*n_current) current history window if present."""
        return batch.get("observation.state_window")


class IdentityEncoder(BaseTemporalEncoder):
    """T0: No-op encoder for ACT-V and ACT-M-instant."""

    def output_state_dim(self) -> int:
        return self.state_dim

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return batch


class HistoryStackEncoder(BaseTemporalEncoder):
    """T1: History stacking for early fusion.

    Expects observation.state_window to contain K+1 timesteps of the
    n_current channels, concatenated. Replaces observation.state with
    [positions_t, currents_t, currents_{t-1}, ..., currents_{t-K}].
    """

    def __init__(self, config: ACTConfig, state_dim: int):
        super().__init__(config, state_dim)
        self.K = config.proprio_K

    def output_state_dim(self) -> int:
        return self.n_position + (self.K + 1) * self.n_current

    def has_history_window(self) -> bool:
        return True

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        window = self._get_state_window(batch)
        if window is None:
            raise KeyError(
                "HistoryStackEncoder requires 'observation.state_window'. "
                "Use TemporalWindowDataset or equivalent."
            )
        expected = (self.K + 1) * self.n_current
        if window.shape[-1] != expected:
            raise ValueError(
                f"HistoryStackEncoder expected state_window dim {expected}, got {window.shape[-1]}"
            )

        batch = dict(batch)
        positions = self._get_positions(batch["observation.state"])
        # Reshape window into (B, K+1, n_current) and flatten to (B, (K+1)*n_current)
        B = window.shape[0]
        currents_history = window.view(B, self.K + 1, self.n_current)
        currents_history = currents_history.view(B, -1)
        batch["observation.state"] = torch.cat([positions, currents_history], dim=-1)
        return batch


class ExplicitFeatureEncoder(BaseTemporalEncoder):
    """T2: Physics-motivated explicit temporal features for early fusion.

    Optionally uses observation.state_window to compute real finite-difference
    derivatives; otherwise the derivative/power features are zero (still useful
    for diagnosing whether the architecture itself removes collapse).

    Expanded state (default features) = positions + currents + derivative +
    residual + power + variance + peak + impulse.
    """

    def __init__(self, config: ACTConfig, state_dim: int):
        super().__init__(config, state_dim)
        self.features = config.proprio_explicit_features
        self.K = config.proprio_K

        # Learnable gravity baseline per joint (initialized to zero)
        self.register_buffer(
            "gravity_baseline",
            torch.zeros(self.n_current),
            persistent=False,
        )
        self.register_buffer(
            "baseline_ema_momentum",
            torch.tensor(0.99),
            persistent=False,
        )

    def output_state_dim(self) -> int:
        # positions + one slot per feature name, each n_current wide
        return self.n_position + len(self.features) * self.n_current

    def produces_embedding(self) -> bool:
        # We also expose an embedding for FiLM/token/hybrid usage
        return True

    def embedding_dim(self) -> int:
        # Embedding = every current-side feature concatenated (positions excluded).
        # Each feature name contributes n_current channels.
        return len(self.features) * self.n_current

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        batch = dict(batch)
        state = batch["observation.state"]
        B = state.shape[0]
        device = state.device

        currents = self._get_currents(state)  # (B, n_current)
        positions = self._get_positions(state)  # (B, n_position)

        feature_list = [positions]

        # Cache derivative/velocity from window for temporalized features
        window = self._get_state_window(batch)
        if window is not None:
            expected = (max(self.K, 1) + 1) * self.n_current
            if window.shape[-1] >= expected:
                w = window.view(B, -1, self.n_current)
                dI = _central_diff(w.transpose(-1, -2), self._dt).transpose(-1, -2)
                dI = dI[..., -1, :]  # present derivative
                dq = _central_diff(
                    positions.unsqueeze(-1).transpose(-1, -2), self._dt
                ).transpose(-1, -2).squeeze(-2)
            else:
                dI = torch.zeros_like(currents)
                dq = torch.zeros_like(positions)
        else:
            dI = torch.zeros_like(currents)
            dq = torch.zeros_like(positions)

        if "raw" in self.features:
            feature_list.append(currents)

        if "derivative" in self.features:
            feature_list.append(dI)

        if "residual" in self.features:
            with torch.no_grad():
                is_free_space = (currents.abs() < self.config.proprio_free_space_threshold).float().mean(dim=-1) > 0.5
                if is_free_space.any():
                    mean_free = currents[is_free_space].mean(dim=0)
                    self.gravity_baseline.mul_(self.baseline_ema_momentum).add_(
                        mean_free * (1 - self.baseline_ema_momentum)
                    )
            I_resid = currents - self.gravity_baseline
            feature_list.append(I_resid)

        if "power" in self.features:
            omega = dq
            # Align shapes: assume first n_current position velocities map to currents
            if omega.shape[-1] >= self.n_current:
                omega = omega[..., : self.n_current]
            else:
                omega = torch.zeros_like(currents)
            P = currents * omega
            feature_list.append(P)

        if any(f in self.features for f in ("variance", "peak", "impulse")):
            if window is not None:
                w = window.view(B, -1, self.n_current)
                if "variance" in self.features:
                    feature_list.append(w.var(dim=1))
                if "peak" in self.features:
                    feature_list.append(w.abs().amax(dim=1))
                if "impulse" in self.features:
                    feature_list.append(w.sum(dim=1))
            else:
                if "variance" in self.features:
                    feature_list.append(torch.zeros_like(currents))
                if "peak" in self.features:
                    feature_list.append(torch.zeros_like(currents))
                if "impulse" in self.features:
                    feature_list.append(torch.zeros_like(currents))

        # Build expanded state and a compact embedding (all current-side features)
        expanded_state = torch.cat(feature_list, dim=-1)
        batch["observation.state"] = expanded_state
        # Embedding = all current features concatenated (positions excluded)
        batch["proprio_embedding"] = torch.cat(feature_list[1:], dim=-1)
        return batch


class CNNTemporalEncoder(BaseTemporalEncoder):
    """T3: Learned 1D-CNN over current history for token/film/hybrid fusion.

    Keeps observation.state unchanged; the CNN consumes observation.state_window
    and writes proprio_embedding.
    """

    def __init__(self, config: ACTConfig, state_dim: int):
        super().__init__(config, state_dim)
        self.K = config.proprio_K
        channels = config.proprio_cnn_channels
        kernels = config.proprio_cnn_kernel_sizes
        dilations = config.proprio_cnn_dilations

        layers = []
        in_ch = self.n_current
        for out_ch, k, d in zip(channels, kernels, dilations):
            pad = (k - 1) * d // 2
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=d, padding=pad),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = channels[-1]

    def output_state_dim(self) -> int:
        return self.state_dim

    def has_history_window(self) -> bool:
        return True

    def produces_embedding(self) -> bool:
        return True

    def embedding_dim(self) -> int:
        return self.out_dim

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        window = self._get_state_window(batch)
        if window is None:
            raise KeyError(
                "CNNTemporalEncoder requires 'observation.state_window'. "
                "Use TemporalWindowDataset or equivalent."
            )
        expected = (self.K + 1) * self.n_current
        if window.shape[-1] != expected:
            raise ValueError(
                f"CNNTemporalEncoder expected state_window dim {expected}, got {window.shape[-1]}"
            )

        B = window.shape[0]
        # window: (B, (K+1)*n_current) -> (B, n_current, K+1)
        current_history = window.view(B, self.K + 1, self.n_current).transpose(1, 2)
        features = self.cnn(current_history)  # (B, channels[-1], L)
        embedding = self.global_pool(features).squeeze(-1)  # (B, channels[-1])

        batch = dict(batch)
        batch["proprio_embedding"] = embedding
        return batch


class TriggerEncoder(BaseTemporalEncoder):
    """T4: Contact-event trigger for hybrid fusion.

    Detects contact from gripper closure + current threshold and writes
    contact_mask plus proprio_embedding (contact features).  Does not modify
    observation.state.
    """

    def __init__(self, config: ACTConfig, state_dim: int):
        super().__init__(config, state_dim)
        self.threshold_I = config.proprio_contact_threshold_I
        self.threshold_dI = config.proprio_contact_threshold_dI
        self.gripper_idx = config.proprio_gripper_idx

    def output_state_dim(self) -> int:
        return self.state_dim

    def produces_embedding(self) -> bool:
        return True

    def produces_contact_mask(self) -> bool:
        return True

    def embedding_dim(self) -> int:
        # contact_features has the same width as the current channels.
        return self.n_current

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        batch = dict(batch)
        state = batch["observation.state"]
        currents = self._get_currents(state)

        # Use the gripper current channel as the contact signal.
        # `gripper_idx` is a *global* state index; translate it to its position
        # within the extracted current channels. Fall back to the last current
        # channel if the gripper is not among the current indices.
        if self.gripper_idx in self.current_indices:
            gidx = self.current_indices.index(self.gripper_idx)
        else:
            gidx = -1
        I_gripper = currents[..., gidx]
        contact_mask = I_gripper > self.threshold_I  # (B,)

        # Contact features = all currents, zeroed for non-contact samples
        contact_features = currents * contact_mask.float().unsqueeze(-1)

        batch["contact_mask"] = contact_mask
        batch["contact_features"] = contact_features
        # HybridFusion expects temporal_features; alias contact_features
        batch["proprio_embedding"] = contact_features
        return batch


# Registry for factory
TEMPORAL_ENCODERS = {
    "none": IdentityEncoder,
    "history": HistoryStackEncoder,
    "explicit": ExplicitFeatureEncoder,
    "cnn": CNNTemporalEncoder,
    "trigger": TriggerEncoder,
}


def build_temporal_encoder(config: ACTConfig, state_dim: int) -> BaseTemporalEncoder:
    """Factory function to instantiate the correct temporal encoder."""
    name = config.proprio_temporal_encoder
    if name not in TEMPORAL_ENCODERS:
        raise ValueError(
            f"Unknown temporal encoder '{name}'. Available: {list(TEMPORAL_ENCODERS.keys())}"
        )
    return TEMPORAL_ENCODERS[name](config, state_dim)
