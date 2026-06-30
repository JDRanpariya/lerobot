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
  F0 early:    concatenate temporal features into the state vector
  F1 token:    add a dedicated token to the encoder sequence
  F2 film:     modulate ResNet visual features via (gamma, beta)
  F3 hybrid:   phase-gated token insertion (only during contact)

All modules return a list of tensors of shape (B, dim_model).  When stacked
by ACT.forward, this gives (S, B, dim_model).

Projection layers (token/hybrid temporal projections, FiLM MLP) are created
EAGERLY in ``__init__`` using dimensions passed by the caller. This is critical:
if they were created lazily on the first forward pass, they would be missing
from the module at optimizer-construction time and would never receive
gradient updates.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .configuration_act import ACTConfig


class BaseFusionModule(nn.Module):
    """Abstract base for fusion strategies.

    Args:
        config: ACT config.
        temporal_dim: width of ``batch['proprio_embedding']`` (the temporal
            encoder's embedding). Passed in at construction time so projection
            layers can be created eagerly and registered with the optimizer.
        backbone_channels: channel count of the vision backbone feature map,
            needed by FiLM to size its (gamma, beta) head eagerly.
    """

    def __init__(
        self,
        config: ACTConfig,
        temporal_dim: int | None = None,
        backbone_channels: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.dim_model = config.dim_model
        self.temporal_dim = temporal_dim
        self.backbone_channels = backbone_channels

    def forward(
        self,
        latent: Tensor,  # (B, dim_model)
        state: Tensor,   # (B, dim_model)
        vision_tokens: List[Tensor],  # each (B, dim_model)
        temporal_features: Tensor | None = None,  # (B, temporal_dim)
        contact_mask: Tensor | None = None,  # (B,)
    ) -> List[Tensor]:
        raise NotImplementedError


class EarlyFusion(BaseFusionModule):
    """F0: Early fusion — temporal features already live in observation.state.

    The temporal encoder has already expanded the state vector, so we just
    project it as usual and append vision tokens.
    """

    def forward(self, latent, state, vision_tokens, **kwargs):
        tokens = [latent, state]
        tokens.extend(vision_tokens)
        return tokens

    def n_non_vision_tokens(self) -> int:
        return 2


class TokenFusion(BaseFusionModule):
    """F1: Intermediate fusion — temporal embedding as a separate token."""

    def __init__(self, config: ACTConfig, temporal_dim: int | None = None, **kwargs):
        super().__init__(config, temporal_dim=temporal_dim, **kwargs)
        if self.temporal_dim is None:
            raise ValueError(
                "TokenFusion requires `temporal_dim` at construction time so its "
                "projection layer is registered before the optimizer is built. "
                "The temporal encoder must expose embedding_dim()."
            )
        self.temporal_proj = nn.Linear(self.temporal_dim, self.dim_model)
        self.temporal_pos_embed = nn.Embedding(1, self.dim_model)

    def forward(self, latent, state, vision_tokens, temporal_features=None, **kwargs):
        tokens = [latent, state]
        if temporal_features is not None:
            temporal_token = self.temporal_proj(temporal_features)  # (B, dim_model)
            tokens.append(temporal_token)
        tokens.extend(vision_tokens)
        return tokens

    def n_non_vision_tokens(self) -> int:
        return 3

    def get_extra_pos_embed(self) -> Tensor:
        return self.temporal_pos_embed.weight.unsqueeze(1)  # (1, 1, dim_model)


class FiLMFusion(BaseFusionModule):
    """F2: FiLM conditioning of the ResNet backbone.

    Generates per-channel gamma and beta from temporal features.  The ACT
    model applies them to the backbone feature map before projection.
    """

    def __init__(
        self,
        config: ACTConfig,
        temporal_dim: int | None = None,
        backbone_channels: int | None = None,
        **kwargs,
    ):
        super().__init__(
            config, temporal_dim=temporal_dim, backbone_channels=backbone_channels
        )
        self.film_layers = config.proprio_film_layers
        if self.temporal_dim is None or self.backbone_channels is None:
            raise ValueError(
                "FiLMFusion requires both `temporal_dim` and `backbone_channels` "
                "at construction time so its FiLM MLP is registered before the "
                "optimizer is built."
            )
        self.film_mlp = nn.Sequential(
            nn.Linear(self.temporal_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.backbone_channels * 2),
        )

    def compute_film_params(
        self,
        temporal_features: Tensor,
        backbone_channels: int,
    ) -> Tuple[Tensor, Tensor]:
        """Return (gamma, beta) with shape (B, backbone_channels)."""
        if backbone_channels != self.backbone_channels:
            raise ValueError(
                f"FiLMFusion was built for {self.backbone_channels} backbone "
                f"channels but received {backbone_channels}."
            )
        film_out = self.film_mlp(temporal_features)  # (B, 2*C)
        gamma, beta = film_out.chunk(2, dim=-1)
        return gamma, beta

    def apply_film(self, feature_map: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        """Apply FiLM: feature_map * (1 + gamma) + beta."""
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return feature_map * (1 + gamma) + beta

    def forward(self, latent, state, vision_tokens, **kwargs):
        # FiLM is applied to the backbone feature map in ACT.forward.
        # This module only returns tokens; vision_tokens are already modulated.
        tokens = [latent, state]
        tokens.extend(vision_tokens)
        return tokens

    def n_non_vision_tokens(self) -> int:
        return 2


class HybridFusion(BaseFusionModule):
    """F3: Hybrid fusion — phase-gated token insertion.

    During contact, a temporal token is added; otherwise it is zeroed so the
    transformer learns to ignore it.  No dynamic sequence lengths required.
    """

    def __init__(self, config: ACTConfig, temporal_dim: int | None = None, **kwargs):
        super().__init__(config, temporal_dim=temporal_dim, **kwargs)
        if self.temporal_dim is None:
            raise ValueError(
                "HybridFusion requires `temporal_dim` at construction time so its "
                "projection layer is registered before the optimizer is built. "
                "The temporal encoder must expose embedding_dim()."
            )
        self.temporal_proj = nn.Linear(self.temporal_dim, self.dim_model)
        self.temporal_pos_embed = nn.Embedding(1, self.dim_model)

    def forward(self, latent, state, vision_tokens, temporal_features=None,
                contact_mask=None, **kwargs):
        tokens = [latent, state]
        if temporal_features is not None:
            temporal_token = self.temporal_proj(temporal_features)  # (B, dim_model)
            if contact_mask is not None:
                temporal_token = temporal_token * contact_mask.float().unsqueeze(-1)
            tokens.append(temporal_token)
        tokens.extend(vision_tokens)
        return tokens

    def n_non_vision_tokens(self) -> int:
        return 3

    def get_extra_pos_embed(self) -> Tensor:
        return self.temporal_pos_embed.weight.unsqueeze(1)


# Registry
FUSION_MODULES = {
    "early": EarlyFusion,
    "token": TokenFusion,
    "film": FiLMFusion,
    "hybrid": HybridFusion,
}


def build_fusion_module(
    config: ACTConfig,
    temporal_dim: int | None = None,
    backbone_channels: int | None = None,
) -> BaseFusionModule:
    """Factory function to instantiate the correct fusion module.

    Args:
        config: ACT config.
        temporal_dim: width of the temporal encoder's embedding (proprio_embedding).
            Required for token/hybrid/film fusion so projections are built eagerly.
        backbone_channels: vision backbone feature-map channel count. Required
            for film fusion.
    """
    name = config.proprio_fusion_stage
    if name not in FUSION_MODULES:
        raise ValueError(
            f"Unknown fusion stage '{name}'. Available: {list(FUSION_MODULES.keys())}"
        )
    return FUSION_MODULES[name](
        config, temporal_dim=temporal_dim, backbone_channels=backbone_channels
    )
