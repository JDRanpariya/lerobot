#!/usr/bin/env python

# Copyright 2026 Jay Ranpariya. Thesis extension.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Adaptive training strategies for multimodal fusion.

FACTR curriculum (Liu et al., 2025): gradually degrade vision early in training
to force the policy to rely on proprioception, then restore visual acuity.

OGM-GE (Peng et al., 2022): dynamically modulate per-modality gradients based
on relative convergence rates to prevent one modality from suppressing another.

These are training-time interventions, not architectural changes. They wrap
around the standard training loop.

SO-101 collapse-axis note (thesis, 2026-07): vanilla FACTR/OGM target the
VISION -> proprioception collapse (vision is the strong modality on a Franka with
good cameras). On the SO-101 the per-modality attribution shows the opposite: the
policy is proprioception-dominated (joint POSITION ~90% of action sensitivity),
vision is weak (~3%), and the bus-servo CURRENT channel is ignored. The collapse
axis is therefore POSITION -> current, *within* proprioception. Blurring vision
(vanilla FACTR) only shifts weight onto position, not current; and the 2-way OGM
below lumps position+current into one "proprio" group, so it cannot separate them.
`ProprioDegradationCurriculum` (degrade position, not vision) and the 3-way OGM
methods below re-target both interventions to the position->current axis. See
`research/current-attending-fusion-design-2026-07.md`.
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class FACTRCurriculum:
    """Factory for FACTR-style vision-degradation curriculum.

    During the first T_decay steps, images are Gaussian-blurred with a
    step-dependent standard deviation:
        sigma(t) = sigma_max * cos(pi/2 * t / T_decay)

    At t=0: sigma = sigma_max (strongest blur, vision least reliable)
    At t=T_decay: sigma = 0 (sharp vision restored)
    """

    def __init__(self, T_decay: int = 30000, sigma_max: float = 8.0):
        self.T_decay = T_decay
        self.sigma_max = sigma_max
        self.kernel_size = int(2 * math.ceil(sigma_max * 3) + 1)  # ~6*sigma, odd

    def __call__(self, images: Tensor, step: int) -> Tensor:
        """Apply curriculum blur to images at given training step.

        Args:
            images: (B, C, H, W) normalized float32 images
            step: current global training step
        Returns:
            Blurred images (or original if step >= T_decay)
        """
        if step >= self.T_decay:
            return images

        sigma = self.sigma_max * math.cos(math.pi / 2 * step / self.T_decay)
        if sigma < 0.1:
            return images

        # Gaussian blur per image
        # F.gaussian_blur expects (B, C, H, W)
        blurred = F.gaussian_blur(
            images, kernel_size=self.kernel_size, sigma=sigma
        )
        return blurred


class ProprioDegradationCurriculum:
    """FACTR curriculum re-targeted to the SO-101 position->current collapse axis.

    Vanilla `FACTRCurriculum` blurs vision to force proprioception reliance. Here
    proprioception (joint POSITION) is already the dominant modality and the weak
    bus-servo CURRENT channel is ignored, so blurring vision would only shift weight
    onto position. Instead we degrade the DOMINANT proprioceptive channels (position,
    i.e. every observation.state channel that is NOT a current index) with
    step-annealed Gaussian noise, forcing the policy to extract signal from current
    early in training:

        sigma(t) = sigma_max * cos(pi/2 * t / T_decay)     (normalized state units)

    At t=0: sigma=sigma_max (position least reliable -> lean on current).
    At t=T_decay: sigma=0 (clean position restored). Current channels are never
    corrupted. Combine with `FACTRCurriculum` if you also want to suppress vision.
    """

    def __init__(self, T_decay: int = 30000, sigma_max: float = 3.0,
                 current_indices=(6, 7, 8, 9, 10, 11)):
        self.T_decay = T_decay
        self.sigma_max = sigma_max
        self.current_indices = set(int(i) for i in current_indices)

    def __call__(self, state: Tensor, step: int) -> Tensor:
        """Add curriculum noise to the position channels of observation.state.

        Args:
            state: (..., D) normalized observation.state (position + current).
            step: current global training step.
        Returns:
            state with position channels noised (current channels untouched).
        """
        if step >= self.T_decay:
            return state
        sigma = self.sigma_max * math.cos(math.pi / 2 * step / self.T_decay)
        if sigma < 1e-3:
            return state
        D = state.shape[-1]
        pos_idx = [i for i in range(D) if i not in self.current_indices]
        if not pos_idx:
            return state
        out = state.clone()
        noise = torch.randn_like(out[..., pos_idx]) * sigma
        out[..., pos_idx] = out[..., pos_idx] + noise
        return out


class OGMGradientModulator:
    """On-the-fly Gradient Modulation (OGM-GE).

    Tracks per-modality loss convergence and scales gradients to prevent
    the strong modality (vision) from suppressing the weak (proprioception).

    Reference: Peng et al., "Balanced Multimodal Learning via On-the-fly
    Gradient Modulation", CVPR 2022.

    The 2-way methods group all proprioception (position+current) together. On the
    SO-101 the collapse is position->current *inside* proprioception, so the 3-way
    methods (`*_3way`) treat current as its own modality group. Using them requires
    the current pathway to have separable parameters (a dedicated current encoder,
    e.g. `proprio_temporal_encoder != none` with a distinct `current_param_prefix`).
    """

    def __init__(self, alpha: float = 1.0, warmup_steps: int = 1000):
        self.alpha = alpha  # modulation strength
        self.warmup_steps = warmup_steps
        # Running averages of per-modality losses
        self.vision_loss_ema = None
        self.proprio_loss_ema = None
        self._ema3 = None   # 3-way EMAs {vision, position, current}
        self.momentum = 0.9

    def compute_modulation_weights(
        self,
        vision_loss: float,
        proprio_loss: float,
        step: int,
    ) -> Tuple[float, float]:
        """Compute gradient scale factors for each modality.

        Returns (vision_weight, proprio_weight) in [0, 1].
        The weaker modality (higher loss relative to EMA) gets up-weighted.
        """
        if step < self.warmup_steps:
            return 1.0, 1.0

        # Update EMAs
        if self.vision_loss_ema is None:
            self.vision_loss_ema = vision_loss
            self.proprio_loss_ema = proprio_loss
        else:
            self.vision_loss_ema = (
                self.momentum * self.vision_loss_ema
                + (1 - self.momentum) * vision_loss
            )
            self.proprio_loss_ema = (
                self.momentum * self.proprio_loss_ema
                + (1 - self.momentum) * proprio_loss
            )

        # Relative difficulty: higher loss = harder to learn = weaker signal
        v_diff = vision_loss / (self.vision_loss_ema + 1e-8)
        p_diff = proprio_loss / (self.proprio_loss_ema + 1e-8)

        # Modulation direction: if vision converges faster (lower diff),
        # reduce its gradient; boost proprioception
        if v_diff < p_diff:
            # Vision is easier — suppress it, boost proprio
            ratio = v_diff / (p_diff + 1e-8)
            vision_weight = ratio ** self.alpha
            proprio_weight = 1.0
        else:
            # Proprio is easier — suppress it, boost vision
            ratio = p_diff / (v_diff + 1e-8)
            vision_weight = 1.0
            proprio_weight = ratio ** self.alpha

        return vision_weight, proprio_weight

    def apply_to_gradients(
        self,
        model,
        vision_weight: float,
        proprio_weight: float,
        vision_param_prefix: str = "model.backbone",
        proprio_param_prefix: str = "model.encoder_robot_state",
    ):
        """Scale gradients in-place by modality weights.

        Args:
            model: the policy model
            vision_weight: scale factor for vision/backbone params
            proprio_weight: scale factor for state/proprio params
        """
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if name.startswith(vision_param_prefix):
                param.grad.mul_(vision_weight)
            elif name.startswith(proprio_param_prefix):
                param.grad.mul_(proprio_weight)
            # MLP/attention layers that process both are left unchanged

    # ---- 3-way (SO-101 position->current axis) ----

    def compute_modulation_weights_3way(
        self,
        vision_loss: float,
        position_loss: float,
        current_loss: float,
        step: int,
    ) -> Tuple[float, float, float]:
        """OGM with CURRENT as its own group (not lumped with position).

        Each modality's loss is passed relative to its own EMA; the RELATIVELY
        hardest modality (highest loss/EMA = slowest to learn = weakest signal) is
        kept at weight 1.0 and the others are scaled down toward it. On the SO-101
        this keeps the ignored current channel's gradient from being suppressed by
        the fast-converging position channel.

        Returns (vision_weight, position_weight, current_weight) in (0, 1].
        """
        if step < self.warmup_steps:
            return 1.0, 1.0, 1.0
        losses = {"vision": vision_loss, "position": position_loss, "current": current_loss}
        if self._ema3 is None:
            self._ema3 = dict(losses)
        else:
            for k in losses:
                self._ema3[k] = self.momentum * self._ema3[k] + (1 - self.momentum) * losses[k]
        diff = {k: losses[k] / (self._ema3[k] + 1e-8) for k in losses}
        hardest = max(diff, key=diff.get)  # weakest modality -> protect it
        w = {k: (1.0 if k == hardest else (diff[k] / (diff[hardest] + 1e-8)) ** self.alpha)
             for k in diff}
        return w["vision"], w["position"], w["current"]

    def apply_to_gradients_3way(
        self,
        model,
        vision_weight: float,
        position_weight: float,
        current_weight: float,
        vision_param_prefix: str = "model.backbone",
        position_param_prefix: str = "model.encoder_robot_state",
        current_param_prefix: str = "model.encoder_current",
    ):
        """Scale gradients in-place by 3-way modality weights.

        Requires a separable current pathway: params whose name starts with
        `current_param_prefix` (e.g. a dedicated current temporal encoder). If the
        architecture lumps current into the state encoder, current shares
        `position_param_prefix` and this degrades to the 2-way behaviour.
        """
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if name.startswith(current_param_prefix):
                param.grad.mul_(current_weight)
            elif name.startswith(vision_param_prefix):
                param.grad.mul_(vision_weight)
            elif name.startswith(position_param_prefix):
                param.grad.mul_(position_weight)
            # shared MLP/attention layers are left unchanged
