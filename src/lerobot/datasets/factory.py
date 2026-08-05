#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
from pprint import pformat

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, IMAGENET_STATS, OBS_PREFIX, REWARD

from .dataset_metadata import LeRobotDatasetMetadata
from .lerobot_dataset import LeRobotDataset
from .multi_dataset import MultiLeRobotDataset
from .streaming_dataset import StreamingLeRobotDataset


def resolve_delta_timestamps(
    cfg: PreTrainedConfig | RewardModelConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the config.

    Args:
        cfg (PreTrainedConfig | RewardModelConfig): The config to read delta_indices from. Both
            ``PreTrainedConfig`` and concrete ``RewardModelConfig`` subclasses expose the
            ``{observation,action,reward}_delta_indices`` properties used below.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, ds_meta)
        if not cfg.dataset.streaming:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                return_uint8=True,
                tolerance_s=cfg.tolerance_s,
            )
        else:
            dataset = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.tolerance_s,
                return_uint8=True,
            )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    # === THESIS EXTENSION: temporal-window wrapper for history-based encoders ===
    # When the policy uses a temporal encoder that needs a K-step current
    # history (history / cnn / explicit), wrap the dataset so each item
    # also carries an `observation.state_window` of past currents. The
    # wrapper preserves episode boundaries (zero-pad before episode start).
    _enc = getattr(getattr(cfg, "trainable_config", None), "proprio_temporal_encoder", "none")
    _K = getattr(getattr(cfg, "trainable_config", None), "proprio_K", 0)
    if _enc in ("history", "cnn", "joint_cnn", "explicit") and _K and _K > 0:
        try:
            from .temporal_window import TemporalWindowDataset

            _cur_idx = getattr(cfg.trainable_config, "proprio_current_indices", None)
            # The window is consumed *after* the policy preprocessor.  Apply
            # the same state normalization here so the temporal path sees the
            # exact normalized current values as the direct state path.  This
            # is particularly important for DP: it uses MIN_MAX, whereas ACT
            # uses MEAN_STD.  The dataset view has already applied any thesis
            # q99.5 current transform before these statistics are computed.
            _norm_mapping = getattr(cfg.trainable_config, "normalization_mapping", {})
            _norm_mode = _norm_mapping.get("STATE", "MEAN_STD")
            # ``joint_cnn`` also exists for ACT, whose temporal encoder
            # expects the legacy flat one-observation window.  DP is the only
            # current consumer that needs one history for each n_obs_steps
            # observation.
            from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

            _obs_steps = (
                int(cfg.trainable_config.n_obs_steps)
                if _enc == "joint_cnn" and isinstance(cfg.trainable_config, DiffusionConfig)
                else 1
            )
            dataset = TemporalWindowDataset(
                dataset,
                K=int(_K),
                state_indices=_cur_idx,
                normalization_mode=_norm_mode,
                observation_steps=_obs_steps,
            )
            logging.info(
                f"Wrapped dataset with TemporalWindowDataset "
                f"(K={_K}, enc={_enc}, n_current={len(_cur_idx) if _cur_idx is not None else 'all'}, "
                f"normalization={getattr(_norm_mode, 'value', _norm_mode)}, observations={_obs_steps})"
            )
        except Exception as e:  # pragma: no cover - depends on external dataset metadata
            raise RuntimeError(
                "TemporalWindowDataset wrapping was requested but failed; refusing to train "
                "with a missing or incorrectly normalized current-history input."
            ) from e
    # ============================================================================

    return dataset
