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

Design principle: encoders are pure functions over batches. They do NOT modify
the model architecture; they only prepare the batch dict for downstream use.
"""

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from .configuration_act import ACTConfig


class BaseTemporalEncoder(nn.Module):
    """Abstract base for temporal encoders.

    Args:
        config: ACTConfig with temporal encoder settings.
        state_dim: Total dimension of observation.state.
    """

    def __init__(self, config: ACTConfig, state_dim: int):
        super().__init__()
        self.config = config
        self.state_dim = state_dim
        self.current_indices = config.proprio_current_indices
        self.n_current = len(self.current_indices)

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


class IdentityEncoder(BaseTemporalEncoder):
    """T0: No-op encoder for ACT-V and ACT-M-instant.

    Passes batch through unchanged. Compatible with any fusion stage.
    """

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return batch


class HistoryStackEncoder(BaseTemporalEncoder):
    """T1: History stacking (FIR filter learning).

    Expects observation.state to already contain K+1 timesteps concatenated
    by the TemporalWindowDataset wrapper. No computation here — just verifies
    shape.
    """

    def __init__(self, config: ACTConfig, state_dim: int):
        super().__init__(config, state_dim)
        self.K = config.proprio_K
        # Expected state dim after dataset wrapper concatenates history
        # Original state dim + K * n_current (past current timesteps)
        # But we allow any shape — the projection layer adapts
        self.expected_dim = state_dim  # after dataset wrapper expansion

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        state = batch["observation.state"]
        # Just verify that state has been expanded by dataset wrapper
        assert state.shape[-1] >= self.state_dim, (
            f"HistoryStackEncoder expects state dim >= {self.state_dim}, "
            f"got {state.shape[-1]}. Make sure dataset wraps with TemporalWindowDataset."
        )
        return batch


class ExplicitFeatureEncoder(BaseTemporalEncoder):
    """T2: Physics-motivated explicit temporal features.

    Computes per-joint features from instantaneous current and position:
    - raw: I(t) [already in state]
    - derivative: dI/dt via Savitzky-Golay or finite difference
    - residual: I(t) - gravity_baseline(q)
    - variance: std(I over local window)
    - peak: max(I over local window)
    - power: I * omega (approximate)
    - impulse: cumulative sum over window

    The gravity baseline is learned online as a running mean of free-space
    current (heuristic: when current is low and stable, update baseline).
    """

    def __init__(self, config: ACTConfig, state_dim: int):
        super().__init__(config, state_dim)
        self.features = config.proprio_explicit_features
        self.K = config.proprio_K  # window for variance/peak/impulse
        self._dt = 1.0 / 30.0  # policy rate 30Hz

        # Learnable gravity baseline per joint (initialized to zero)
        # Updated online via EMA when current is stable (free-space)
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

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        state = batch["observation.state"]
        B = state.shape[0]
        device = state.device

        currents = self._get_currents(state)  # (B, n_current)
        positions = self._get_positions(state)  # (B, n_pos)

        feature_list = [positions]  # always keep positions

        if "raw" in self.features:
            feature_list.append(currents)

        if "derivative" in self.features:
            # Finite difference: (I_t - I_{t-1}) / dt
            # Store last current in buffer for next step (inference)
            # For training batch, we compute diff along batch dim if sequential
            # But batches are shuffled — use per-sample diff if state contains history
            dI = self._compute_derivative(state)
            feature_list.append(dI)

        if "residual" in self.features:
            # Online EMA of gravity baseline
            with torch.no_grad():
                # Update baseline for samples that look like free-space
                # Heuristic: current < 20 counts above baseline for 3 joints
                is_free_space = (currents.abs() < 20).float().mean(dim=-1) > 0.5
                if is_free_space.any():
                    mean_free = currents[is_free_space].mean(dim=0)
                    self.gravity_baseline.mul_(self.baseline_ema_momentum).add_(
                        mean_free * (1 - self.baseline_ema_momentum)
                    )
            I_resid = currents - self.gravity_baseline
            feature_list.append(I_resid)

        if "power" in self.features:
            # Approximate omega from position finite difference
            # omega = (q_t - q_{t-1}) / dt  (rad/s)
            # power_proxy = I * omega
            omega = self._compute_velocity(positions)
            P = currents * omega  # element-wise, same shape via broadcasting or slicing
            if P.shape != currents.shape:
                # If positions and currents have different counts, need to align
                # For gripper-focused: use gripper current (idx 5) * gripper vel
                P = currents * omega[..., : self.n_current]
            feature_list.append(P)

        if "variance" in self.features or "peak" in self.features or "impulse" in self.features:
            # These need history — for batch training, use batch statistics as proxy
            # At inference, maintain rolling window
            I_var, I_peak, I_impulse = self._compute_window_features(currents)
            if "variance" in self.features:
                feature_list.append(I_var)
            if "peak" in self.features:
                feature_list.append(I_peak)
            if "impulse" in self.features:
                feature_list.append(I_impulse)

        batch["observation.state"] = torch.cat(feature_list, dim=-1)
        return batch

    def _compute_derivative(self, state: Tensor) -> Tensor:
        """Compute dI/dt. For training batches, use zero (no sequential guarantee)."""
        currents = self._get_currents(state)
        # During training with shuffled batches, finite diff is meaningless.
        # Use zero placeholder; at inference, policy maintains history buffer.
        return torch.zeros_like(currents)

    def _compute_velocity(self, positions: Tensor) -> Tensor:
        """Compute approximate joint velocity. Same limitation as derivative."""
        return torch.zeros_like(positions)

    def _compute_window_features(self, currents: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return (variance, peak, impulse) over local window.
        For training batch, use batch-level stats as proxy."""
        # Variance across batch as proxy for temporal variance
        I_var = currents.var(dim=0, keepdim=True).expand_as(currents)
        I_peak = currents.abs().amax(dim=0, keepdim=True).expand_as(currents)
        I_impulse = currents.sum(dim=0, keepdim=True).expand_as(currents)
        return I_var, I_peak, I_impulse


class CNNTemporalEncoder(BaseTemporalEncoder):
    """T3: Learned 1D-CNN over current history.

    Expects state to contain K+1 timesteps of current (from dataset wrapper
    or policy rolling buffer). Builds a compact contact embedding.
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
            # Padding "same" for variable-length sequences
            # At 30Hz policy rate, K+1 steps = 300ms + current
            pad = (k - 1) * d // 2
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=d, padding=pad),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = channels[-1]

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        state = batch["observation.state"]
        # Extract current history: assumed to be last (K+1)*n_current dims
        history_len = self.K + 1
        expected_hist_dim = history_len * self.n_current
        assert state.shape[-1] >= expected_hist_dim, (
            f"CNNTemporalEncoder expects state dim >= {expected_hist_dim}, got {state.shape[-1]}"
        )
        # Current history is at end of state vector
        current_history = state[..., -expected_hist_dim:]  # (B, (K+1)*n_current)
        # Reshape to (B, n_current, K+1)
        B = state.shape[0]
        current_history = current_history.view(B, self.n_current, history_len)
        # CNN forward
        features = self.cnn(current_history)  # (B, channels[-1], L)
        embedding = self.global_pool(features).squeeze(-1)  # (B, channels[-1])
        batch["proprio_embedding"] = embedding
        return batch


class TriggerEncoder(BaseTemporalEncoder):
    """T4: Contact-event trigger (used with hybrid fusion).

    Detects contact from gripper closure + current threshold + onset rate.
    Adds 'contact_mask' (B,) bool and 'contact_features' (B, D) to batch.
    """

    def __init__(self, config: ACTConfig, state_dim: int):
        super().__init__(config, state_dim)
        self.threshold_I = config.proprio_contact_threshold_I
        self.threshold_dI = config.proprio_contact_threshold_dI
        self.gripper_idx = config.proprio_gripper_idx

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        state = batch["observation.state"]
        currents = self._get_currents(state)
        gripper_pos = state[..., self.gripper_idx]

        # Contact detection: I_resid > threshold
        # Since we don't have baseline here, use absolute threshold
        # (simplified; full version uses ExplicitFeatureEncoder's baseline)
        I_gripper = currents[..., -1] if currents.shape[-1] > 0 else torch.zeros_like(gripper_pos)
        contact_mask = I_gripper > self.threshold_I  # (B,)

        # Build contact features: [I, dI/dt, residual] for active samples
        contact_features = currents.clone()
        # Zero out for non-contact samples (hybrid fusion handles gating)
        contact_features = contact_features * contact_mask.float().unsqueeze(-1)

        batch["contact_mask"] = contact_mask
        batch["contact_features"] = contact_features
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
