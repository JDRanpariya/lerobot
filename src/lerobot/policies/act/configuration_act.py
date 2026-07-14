#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig


@PreTrainedConfig.register_subclass("act")
@dataclass
class ACTConfig(PreTrainedConfig):
    """Configuration class for the Action Chunking Transformers policy.

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_features` and `output_features`.

    Notes on the inputs and outputs:
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.images." they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            This should be no greater than the chunk size. For example, if the chunk size size 100, you may
            set this to 50. This would mean that the model predicts 100 steps worth of actions, runs 50 in the
            environment, and throws the other 50 out.
        input_features: A dictionary defining the PolicyFeature of the input data for the policy. The key represents
            the input data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy. The key represents
            the output data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        normalization_mapping: A dictionary that maps from a str value of FeatureType (e.g., "STATE", "VISUAL") to
            a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX)
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a dilated
            convolution.
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        dim_model: The transformer blocks' main hidden dimension.
        n_heads: The number of heads to use in the transformer blocks' multi-head attention.
        dim_feedforward: The dimension to expand the transformer's hidden dimension to in the feed-forward
            layers.
        feedforward_activation: The activation to use in the transformer block's feed-forward layers.
        n_encoder_layers: The number of transformer layers to use for the transformer encoder.
        n_decoder_layers: The number of transformer layers to use for the transformer decoder.
        use_vae: Whether to use a variational objective during training. This introduces another transformer
            which is used as the VAE's encoder (not to be confused with the transformer encoder - see
            documentation in the policy class).
        latent_dim: The VAE's latent dimension.
        n_vae_encoder_layers: The number of transformer layers to use for the VAE's encoder.
        temporal_ensemble_coeff: Coefficient for the exponential weighting scheme to apply for temporal
            ensembling. Defaults to None which means temporal ensembling is not used. `n_action_steps` must be
            1 when using this feature, as inference needs to happen at every step to form an ensemble. For
            more information on how ensembling works, please see `ACTTemporalEnsembler`.
        dropout: Dropout to use in the transformer layers (see code for details).
        kl_weight: The weight to use for the KL-divergence component of the loss if the variational objective
            is enabled. Loss is then calculated as: `reconstruction_loss + kl_weight * kld_loss`.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
    # that means only the first layer is used. Here we match the original implementation by setting this to 1.
    # See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    # Note: the value used in ACT when temporal ensembling is enabled is 0.01.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    # === TEMPORAL ENCODING for proprioceptive signals ===
    # What: how the current time-series is represented before fusion
    proprio_temporal_encoder: str = "none"  # none | history | explicit | cnn
    proprio_K: int = 9                      # history window steps (Method 1)
    # Which state indices contain current channels (for temporal encoders)
    # Default: indices 6-11 in act-m view (position 0-5, current 6-11)
    proprio_current_indices: list[int] = field(default_factory=lambda: [6, 7, 8, 9, 10, 11])
    # Explicit features to compute (Method 2)
    proprio_explicit_features: list[str] = field(default_factory=lambda: [
        "raw", "derivative", "residual", "variance", "peak", "power", "impulse"
    ])
    # CNN architecture (Method 3)
    proprio_cnn_channels: list[int] = field(default_factory=lambda: [16, 16, 8])
    proprio_cnn_kernel_sizes: list[int] = field(default_factory=lambda: [3, 3, 3])
    proprio_cnn_dilations: list[int] = field(default_factory=lambda: [1, 2, 4])

    # === FUSION STAGE ===
    # Where: at what architectural depth temporal features fuse with vision
    proprio_fusion_stage: str = "early"  # early | token | film | hybrid

    # Thresholds are in normalized observation-state units (after the
    # checkpoint preprocessor), not raw STS3215 register counts.
    # Contact detection thresholds (for hybrid fusion, Method 4)
    proprio_contact_threshold_I: float = 2.0
    proprio_contact_threshold_dI: float = 2.0
    proprio_free_space_threshold: float = 2.0
    # Gripper position index in state vector for contact detection
    proprio_gripper_idx: int = 5

    # FiLM conditioning layers (1-indexed ResNet blocks)
    proprio_film_layers: list[int] = field(default_factory=lambda: [1, 2, 3])

    # === ADAPTIVE TRAINING (FACTR + OGM-GE reserve) ===
    use_factr: bool = False
    factr_T_decay: int = 30000
    factr_sigma_max: float = 8.0
    use_ogm_ge: bool = False
    # Position->current collapse axis (SO-101-specific; see
    # research/current-attending-fusion-design-2026-07.md). Vanilla FACTR degrades
    # vision to force proprioception; here proprioception (position) is already
    # dominant and current is ignored, so the curriculum degrades POSITION channels
    # and the 3-way OGM treats current as its own gradient group.
    use_proprio_curriculum: bool = False
    proprio_curriculum_T_decay: int = 30000
    proprio_curriculum_sigma_max: float = 3.0  # normalized state units
    ogm_three_way: bool = False                # split vision / position / current

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        # Validate temporal encoder
        valid_temporal = {"none", "history", "explicit", "cnn", "trigger"}
        if self.proprio_temporal_encoder not in valid_temporal:
            raise ValueError(
                f"`proprio_temporal_encoder` must be one of {valid_temporal}. "
                f"Got '{self.proprio_temporal_encoder}'."
            )
        # Validate fusion stage
        valid_fusion = {"early", "token", "film", "hybrid"}
        if self.proprio_fusion_stage not in valid_fusion:
            raise ValueError(
                f"`proprio_fusion_stage` must be one of {valid_fusion}. "
                f"Got '{self.proprio_fusion_stage}'."
            )

        # Capability matrix: which encoder can feed which fusion stage
        # embedding = produces a proprio_embedding vector usable by token/film
        # contact   = produces contact_mask usable by hybrid
        # history   = needs observation.state_window (K-step history of currents)
        temporal_caps = {
            "none":    {"embedding": False, "contact": False, "history": False},
            "history": {"embedding": False, "contact": False, "history": True},
            "explicit":{"embedding": True,  "contact": False, "history": False},
            "cnn":     {"embedding": True,  "contact": False, "history": True},
            "trigger": {"embedding": True,  "contact": True,  "history": False},
        }
        caps = temporal_caps[self.proprio_temporal_encoder]

        # Validate temporal+fusion compatibility
        if self.proprio_temporal_encoder == "none" and self.proprio_fusion_stage != "early":
            raise ValueError(
                "When temporal_encoder='none', fusion_stage must be 'early'."
            )
        if self.proprio_fusion_stage in ("token", "film") and not caps["embedding"]:
            raise ValueError(
                f"Temporal encoder '{self.proprio_temporal_encoder}' does not produce an embedding, "
                f"so fusion_stage='{self.proprio_fusion_stage}' is not supported."
            )
        if self.proprio_fusion_stage == "hybrid" and not caps["contact"]:
            raise ValueError(
                f"Temporal encoder '{self.proprio_temporal_encoder}' does not produce a contact mask, "
                f"so fusion_stage='hybrid' is not supported."
            )
        if caps["history"] and self.proprio_K <= 0:
            raise ValueError(
                f"Temporal encoder '{self.proprio_temporal_encoder}' requires proprio_K > 0."
            )
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
