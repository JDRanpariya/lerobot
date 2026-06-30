#!/usr/bin/env python

# Copyright 2026 Jay Ranpariya. Thesis extension.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Multimodal dataset factory with temporal window support."""

from pathlib import Path
from typing import Dict, Optional

import torch

from lerobot.datasets import LeRobotDataset

from .temporal_window import TemporalWindowDataset


class MultimodalDataset(LeRobotDataset):
    """Extended dataset with optional temporal window for proprioception."""

    def __init__(
        self,
        repo_id: str,
        root: Optional[str | Path] = None,
        temporal_window: int = 0,
        state_window_indices: Optional[list[int]] = None,
        **kwargs,
    ):
        super().__init__(repo_id=repo_id, root=root, **kwargs)
        self.temporal_window = temporal_window
        self.state_window_indices = state_window_indices

        if temporal_window > 0:
            self._windowed = TemporalWindowDataset(
                base_dataset=self,
                K=temporal_window,
                state_indices=state_window_indices,
            )
        else:
            self._windowed = None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._windowed is not None:
            return self._windowed[idx]
        return super().__getitem__(idx)

    def __len__(self) -> int:
        return len(self._windowed) if self._windowed else super().__len__()
