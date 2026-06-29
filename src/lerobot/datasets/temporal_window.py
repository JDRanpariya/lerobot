#!/usr/bin/env python

# Copyright 2026 Jay Ranpariya. Thesis extension.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Temporal window dataset wrapper for history-stacking temporal encoders.

Wraps a LeRobotDataset to provide K-step history of state features.
This is needed for Methods 1 (history stacking) and 3 (1D-CNN) where
the temporal encoder expects a window of past timesteps.

The wrapper modifies __getitem__ to return an expanded state vector:
  Original:    state = [q_t, I_t]                    (12-D for act-m)
  With K=9:    state = [q_t, I_t, I_{t-1}, ..., I_{t-9}]  (66-D)

Episode boundaries are respected: frames before episode start are zero-padded
rather than pulling from the previous episode.

Usage:
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.temporal_window import TemporalWindowDataset

    base_ds = LeRobotDataset(repo_id, root=dataset_dir)
    ds = TemporalWindowDataset(base_ds, K=9, state_indices=[6,7,8,9,10,11])
    # state_indices specifies which channels to history-stack
"""

import torch
from torch.utils.data import Dataset


class TemporalWindowDataset(Dataset):
    """Wraps a LeRobotDataset to concatenate K past state timesteps.

    Args:
        base_dataset: The underlying LeRobotDataset.
        K: Number of past timesteps to include (history window length).
        state_key: The key in the dataset item dict for the state tensor.
        episode_key: The key for episode index (default "episode_index").
        frame_key: The key for frame index within episode (default "frame_index").
    """

    def __init__(
        self,
        base_dataset: Dataset,
        K: int = 9,
        state_key: str = "observation.state",
        episode_key: str = "episode_index",
        frame_key: str = "frame_index",
    ):
        self.base = base_dataset
        self.K = K
        self.state_key = state_key
        self.episode_key = episode_key
        self.frame_key = frame_key

        # Precompute episode boundaries for fast lookup
        # Build a map: episode_idx -> (start_frame_idx, end_frame_idx)
        self._episode_bounds = {}
        if hasattr(base_dataset, "episode_data_index"):
            # LeRobotDataset stores this
            self._episode_bounds = self._build_episode_bounds_from_index()
        else:
            self._episode_bounds = self._build_episode_bounds_scan()

    def _build_episode_bounds_from_index(self) -> dict:
        """Use LeRobotDataset's episode_data_index if available."""
        idx = self.base.episode_data_index
        bounds = {}
        for ep_idx in range(len(idx["from_index"])):
            start = int(idx["from_index"][ep_idx])
            end = int(idx["to_index"][ep_idx])
            bounds[ep_idx] = (start, end)
        return bounds

    def _build_episode_bounds_scan(self) -> dict:
        """Fallback: scan all frames to find episode boundaries."""
        bounds = {}
        for i in range(len(self.base)):
            item = self.base[i]
            ep_idx = int(item[self.episode_key].item())
            if ep_idx not in bounds:
                bounds[ep_idx] = [i, i]
            else:
                bounds[ep_idx][1] = i
        return {k: (v[0], v[1] + 1) for k, v in bounds.items()}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        """Return item with K-step history concatenated into state.

        For each past timestep k in [K, K-1, ..., 0]:
          - If within same episode: use that frame's state
          - If before episode start: zero-pad
        """
        item = self.base[idx]
        current_ep = int(item[self.episode_key].item())

        # Find episode start index
        ep_start, ep_end = self._episode_bounds.get(current_ep, (0, len(self.base)))
        frame_in_ep = idx - ep_start

        # Collect K past + current state
        state_list = []
        for k in range(self.K, -1, -1):  # K, K-1, ..., 0
            past_frame_in_ep = frame_in_ep - k
            if past_frame_in_ep < 0:
                # Before episode start: zero-pad
                template = item[self.state_key]
                past_state = torch.zeros_like(template)
            else:
                past_idx = ep_start + past_frame_in_ep
                past_item = self.base[past_idx]
                # Sanity check: same episode
                past_ep = int(past_item[self.episode_key].item())
                if past_ep != current_ep:
                    template = item[self.state_key]
                    past_state = torch.zeros_like(template)
                else:
                    past_state = past_item[self.state_key]
            state_list.append(past_state)

        # Concatenate along last dimension
        item = dict(item)  # shallow copy
        item[self.state_key] = torch.cat(state_list, dim=-1)
        return item
