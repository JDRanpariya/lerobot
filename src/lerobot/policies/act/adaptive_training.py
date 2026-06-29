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


class OGMGradientModulator:
    """On-the-fly Gradient Modulation (OGM-GE).

    Tracks per-modality loss convergence and scales gradients to prevent
    the strong modality (vision) from suppressing the weak (proprioception).

    Reference: Peng et al., "Balanced Multimodal Learning via On-the-fly
    Gradient Modulation", CVPR 2022.
    """

    def __init__(self, alpha: float = 1.0, warmup_steps: int = 1000):
        self.alpha = alpha  # modulation strength
        self.warmup_steps = warmup_steps
        # Running averages of per-modality losses
        self.vision_loss_ema = None
        self.proprio_loss_ema = None
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
