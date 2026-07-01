#!/usr/bin/env python

# Copyright 2026 Jay Ranpariya. Thesis extension.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Temporal window dataset wrapper for history-stacking temporal encoders.

Adds a new key `observation.state_window` containing a concatenated history
of selected state channels (typically motor currents).  The original
`observation.state` is left unchanged, so token-/FiLM-based encoders can keep
a compact state while still accessing temporal context.

Usage:
    base_ds = LeRobotDataset(repo_id, root=dataset_dir)
    ds = TemporalWindowDataset(
        base_ds,
        K=9,
        state_indices=[6, 7, 8, 9, 10, 11],  # current channels
    )

The wrapper guarantees episode boundary respect: frames before episode start
are zero-padded rather than pulling from a previous episode.
"""

import torch
from torch.utils.data import Dataset


class TemporalWindowDataset(Dataset):
    """Wraps a LeRobotDataset to append a K-step state window.

    Args:
        base_dataset: The underlying LeRobotDataset.
        K: Number of past timesteps to include (history window length).
        state_indices: Which state channels to history-stack. If None, all
            channels are stacked (legacy behavior; not recommended).
        state_key: The key in the dataset item dict for the state tensor.
        window_key: The new key to write the window into.
        episode_key: The key for episode index.
        frame_key: The key for frame index within episode.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        K: int = 9,
        state_indices: list[int] | None = None,
        state_key: str = "observation.state",
        window_key: str = "observation.state_window",
        episode_key: str = "episode_index",
        frame_key: str = "frame_index",
    ):
        self.base = base_dataset
        self.K = K
        self.state_indices = state_indices
        self.state_key = state_key
        self.window_key = window_key
        self.episode_key = episode_key
        self.frame_key = frame_key

        self._episode_bounds = {}
        if hasattr(base_dataset, "episode_data_index"):
            self._episode_bounds = self._build_episode_bounds_from_index()
        else:
            self._episode_bounds = self._build_episode_bounds_scan()

    # ------------------------------------------------------------------
    # Attribute passthrough: forward any attribute the training pipeline
    # expects on a LeRobotDataset (`.meta`, `.num_frames`, `.num_episodes`,
    # `.episodes`, `.meta.stats`, `.meta.camera_keys`, `.hf_dataset`, ...)
    # to the wrapped base dataset. Without this, lerobot_train.py raises
    # AttributeError on `dataset.meta` etc. because this wrapper only
    # overrides __getitem__/__len__.
    # ------------------------------------------------------------------
    def __getattr__(self, name):
        # __getattr__ is only called when normal lookup fails, so this
        # safely forwards everything not found on the wrapper itself.
        # `self.base` is set in __init__ before any external access, but
        # guard against early-init access (e.g. copy/pickle) to avoid
        # infinite recursion.
        if name == "base":
            raise AttributeError(name)
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def _build_episode_bounds_from_index(self) -> dict:
        idx = self.base.episode_data_index
        bounds = {}
        for ep_idx in range(len(idx["from_index"])):
            start = int(idx["from_index"][ep_idx])
            end = int(idx["to_index"][ep_idx])
            bounds[ep_idx] = (start, end)
        return bounds

    def _build_episode_bounds_scan(self) -> dict:
        # Use the parquet-backed tabular layer to avoid per-frame video decodes.
        try:
            tabular = self.base.hf_dataset
        except Exception:
            tabular = None
        bounds = {}
        n = len(self.base)
        for i in range(n):
            ep_idx = (int(tabular[i][self.episode_key].item()) if tabular is not None
                       else int(self.base[i][self.episode_key].item()))
            if ep_idx not in bounds:
                bounds[ep_idx] = [i, i]
            else:
                bounds[ep_idx][1] = i
        return {k: (v[0], v[1] + 1) for k, v in bounds.items()}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        current_ep = int(item[self.episode_key].item())
        ep_start, ep_end = self._episode_bounds.get(current_ep, (0, len(self.base)))
        frame_in_ep = idx - ep_start

        state = item[self.state_key]
        n_channels = state.shape[-1]
        indices = self.state_indices if self.state_indices is not None else list(range(n_channels))

        window_list = []
        # Cache the parquet-backed tabular layer so past-state reads do NOT
        # trigger a video decode. base[idx] decodes the image at idx (~12ms);
        # base.hf_dataset[idx] reads only the tabular row (~0.03ms, ~400x faster).
        # We only need past CURRENTS from the state vector, so the image is
        # discarded work — read the table directly instead.
        try:
            tabular = self.base.hf_dataset
        except Exception:
            tabular = None  # fall back to full base[idx] if no hf_dataset

        for k in range(self.K, -1, -1):  # K, K-1, ..., 0
            past_frame_in_ep = frame_in_ep - k
            if past_frame_in_ep < 0:
                # Before episode start: zero-pad (matching stat shape)
                past_state = torch.zeros_like(state)
            else:
                past_idx = ep_start + past_frame_in_ep
                if tabular is not None:
                    # Cheap tabular read; episode boundary already guaranteed
                    # by past_idx being within [ep_start, ep_end).
                    past_state = tabular[past_idx][self.state_key]
                else:
                    past_item = self.base[past_idx]
                    past_ep = int(past_item[self.episode_key].item())
                    if past_ep != current_ep:
                        past_state = torch.zeros_like(state)
                    else:
                        past_state = past_item[self.state_key]
            window_list.append(past_state[..., indices])

        item = dict(item)  # shallow copy
        item[self.window_key] = torch.cat(window_list, dim=-1)
        return item
