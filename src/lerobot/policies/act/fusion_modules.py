#!/usr/bin/env python

# Copyright 2026 Jay Ranpariya. Thesis extension.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Fusion modules for combining proprioceptive temporal features with vision.

Each fusion module takes the processed temporal features and decides HOW they
enter the transformer encoder. This is independent of the temporal encoder
that produced the features.

Fusion stages:
  F0 early:    concatenate temporal features into state vector
  F1 token:    add dedicated token(s) to encoder sequence
  F2 film:     modulate ResNet visual features via (gamma, beta)
  F3 hybrid:   phase-gated token insertion (only during contact)
"""

from typing import Dict, List, Tuple

import einops
import torch
import torch.nn as nn
from torch import Tensor

from .configuration_act import ACTConfig


class BaseFusionModule(nn.Module):
    """Abstract base for fusion strategies.

    All subclasses implement:
        forward(latent, state, vision_tokens, temporal_features=None,
                contact_mask=None, film_params=None)
        → (encoder_tokens, pos_embeds) as lists of (seq, B, dim) tensors
    """

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config
        self.dim_model = config.dim_model

    def forward(
        self,
        latent: Tensor,  # (B, latent_dim)
        state: Tensor,   # (B, state_dim)
        vision_tokens: List[Tensor],  # list of (seq, B, dim) from backbone
        temporal_features: Tensor | None = None,  # (B, temporal_dim) from temporal encoder
        contact_mask: Tensor | None = None,  # (B,) bool for hybrid
        film_params: Tuple[Tensor, Tensor] | None = None,  # (gamma, beta) for FiLM
    ) -> Tuple[List[Tensor], List[Tensor]]:
        raise NotImplementedError


class EarlyFusion(BaseFusionModule):
    """F0: Early fusion — concatenate temporal features into state vector.

    The state projection layer (encoder_robot_state_input_proj) sees a wider
    vector and learns a joint embedding. This is the simplest fusion: the
    temporal encoder has already expanded the state dimension, so we just
    project it as usual.
    """

    def __init__(self, config: ACTConfig):
        super().__init__(config)
        # No extra parameters — state projection is already in ACT.__init__
        # We just need to know the expected state dimension after temporal encoding

    def forward(self, latent, state, vision_tokens, **kwargs):
        """Returns tokens in the order: latent, state, [vision_tokens...]."""
        # Latent token
        tokens = [latent]  # caller will apply encoder_latent_input_proj
        # State token (already enhanced by temporal encoder)
        tokens.append(state)
        # Vision tokens (already projected by backbone)
        tokens.extend(vision_tokens)
        return tokens

    def n_non_vision_tokens(self) -> int:
        """Number of non-vision tokens: latent + state = 2."""
        return 2


class TokenFusion(BaseFusionModule):
    """F1: Intermediate fusion — add temporal features as separate encoder token(s).

    The temporal encoder produces an embedding (e.g., CNN output). We project
    it to dim_model and insert it as a separate token in the encoder sequence,
    with its own positional embedding.
    """

    def __init__(self, config: ACTConfig):
        super().__init__(config)
        # Temporal projection: output of temporal encoder → dim_model
        # Actual input dim depends on temporal encoder; we use a flexible linear
        # that can be resized if needed, or assume temporal encoder outputs dim_model
        self.temporal_dim = None  # set on first forward
        self.temporal_proj = None
        # Extra positional embedding for temporal token
        self.temporal_pos_embed = nn.Embedding(1, self.dim_model)

    def _ensure_proj(self, temporal_features: Tensor):
        """Lazy init of projection to match temporal feature dimension."""
        if self.temporal_proj is None:
            self.temporal_dim = temporal_features.shape[-1]
            self.temporal_proj = nn.Linear(self.temporal_dim, self.dim_model).to(
                temporal_features.device
            )

    def forward(self, latent, state, vision_tokens, temporal_features=None, **kwargs):
        tokens = [latent, state]
        if temporal_features is not None:
            self._ensure_proj(temporal_features)
            temporal_token = self.temporal_proj(temporal_features)  # (B, dim_model)
            temporal_token = temporal_token.unsqueeze(0)  # (1, B, dim_model)
            tokens.append(temporal_token)
        tokens.extend(vision_tokens)
        return tokens

    def n_non_vision_tokens(self) -> int:
        return 3  # latent + state + temporal

    def get_extra_pos_embed(self) -> Tensor:
        """Return positional embedding for temporal token."""
        return self.temporal_pos_embed.weight.unsqueeze(1)  # (1, 1, dim_model)


class FiLMFusion(BaseFusionModule):
    """F2: FiLM conditioning — use temporal features to modulate backbone outputs.

    The temporal encoder produces (gamma, beta) vectors that scale and shift
    the ResNet feature maps before they enter the transformer. This avoids
    adding tokens to the sequence (no attention competition).

    Implementation: We apply FiLM at specific ResNet layers via forward hooks.
    Since this module doesn't directly return modified vision tokens, it stores
    film_params in the batch dict and the ACT model applies them in backbone forward.
    """

    def __init__(self, config: ACTConfig):
        super().__init__(config)
        # FiLM parameters are computed by the temporal encoder
        # This module mainly acts as a pass-through with metadata
        self.film_layers = config.proprio_film_layers

    def forward(self, latent, state, vision_tokens, film_params=None, **kwargs):
        tokens = [latent, state]
        # FiLM is applied to vision tokens BEFORE they reach here
        # (ACT model applies film to backbone output if film_params present)
        tokens.extend(vision_tokens)
        return tokens

    def n_non_vision_tokens(self) -> int:
        return 2  # latent + state (no extra temporal token)

    def apply_film(self, feature_map: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        """Apply FiLM: feature_map * gamma + beta.

        Args:
            feature_map: (B, C, H, W)
            gamma: (B, C) broadcast to (B, C, 1, 1)
            beta: (B, C) broadcast to (B, C, 1, 1)
        Returns:
            (B, C, H, W) modulated feature map
        """
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return feature_map * (1 + gamma) + beta


class HybridFusion(BaseFusionModule):
    """F3: Hybrid fusion — phase-gated token insertion.

    During free-space motion: same as early fusion (state only).
    During contact: add temporal features as extra tokens (like token fusion).
    The contact mask selects which batch items get the extra token.

    Implementation: For simplicity, we add the temporal token to ALL samples
    but zero it out for non-contact samples. The transformer learns to ignore
    zeroed tokens (standard behavior with masking).
    """

    def __init__(self, config: ACTConfig):
        super().__init__(config)
        # Lazy init projection
        self.temporal_proj = None
        self.temporal_pos_embed = nn.Embedding(1, self.dim_model)

    def _ensure_proj(self, temporal_features: Tensor):
        if self.temporal_proj is None:
            temporal_dim = temporal_features.shape[-1]
            self.temporal_proj = nn.Linear(temporal_dim, self.dim_model).to(
                temporal_features.device
            )

    def forward(self, latent, state, vision_tokens, temporal_features=None,
                contact_mask=None, **kwargs):
        tokens = [latent, state]
        if temporal_features is not None:
            self._ensure_proj(temporal_features)
            temporal_token = self.temporal_proj(temporal_features)  # (B, dim_model)
            # Zero out for non-contact samples
            if contact_mask is not None:
                temporal_token = temporal_token * contact_mask.float().unsqueeze(-1)
            temporal_token = temporal_token.unsqueeze(0)  # (1, B, dim_model)
            tokens.append(temporal_token)
        tokens.extend(vision_tokens)
        return tokens

    def n_non_vision_tokens(self) -> int:
        return 3  # latent + state + conditional temporal

    def get_extra_pos_embed(self) -> Tensor:
        return self.temporal_pos_embed.weight.unsqueeze(1)


# Registry
FUSION_MODULES = {
    "early": EarlyFusion,
    "token": TokenFusion,
    "film": FiLMFusion,
    "hybrid": HybridFusion,
}


def build_fusion_module(config: ACTConfig) -> BaseFusionModule:
    """Factory function to instantiate the correct fusion module."""
    name = config.proprio_fusion_stage
    if name not in FUSION_MODULES:
        raise ValueError(
            f"Unknown fusion stage '{name}'. Available: {list(FUSION_MODULES.keys())}"
        )
    return FUSION_MODULES[name](config)
