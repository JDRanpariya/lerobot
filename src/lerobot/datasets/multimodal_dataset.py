#!/usr/bin/env python

# Copyright 2026 Jay Ranpariya. Thesis extension.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Multimodal dataset factory with temporal window support.

This module provides a dataset class that wraps LeRobotDataset with
temporal window capabilities for training ACT-M variants.

Usage in training config:
    dataset:
      _class_: lerobot.datasets.multimodal_dataset.MultimodalDataset
      repo_id: local/so101-peg-...
      root: /path/to/dataset
      temporal_window: 9  # K-step history
"""

from pathlib import Path
from typing import Dict, Optional

import torch
from datasets import Dataset as HFDataset

from lerobot.datasets import LeRobotDataset

from .temporal_window import TemporalWindowDataset


class MultimodalDataset(LeRobotDataset):
    """Extended dataset with optional temporal window and proprioception views.

    Inherits from LeRobotDataset and adds:
      - temporal_window: K-step history stacking for state features
      - view: which state subset to use (act-v, act-m, etc.)

    When temporal_window > 0, each sample's observation.state is concatenated
    with K past timesteps of the specified state channels.
    """

    def __init__(
        self,
        repo_id: str,
        root: Optional[str | Path] = None,
        temporal_window: int = 0,
        state_channels: Optional[list] = None,
        **kwargs,
    ):
        """Initialize multimodal dataset.

        Args:
            repo_id: Dataset repository identifier.
            root: Root directory containing the dataset.
            temporal_window: K-step history. 0 means no history (default).
            state_channels: Which state indices to history-stack. None means all.
            **kwargs: Additional arguments passed to LeRobotDataset.
        """
        super().__init__(repo_id=repo_id, root=root, **kwargs)
        self.temporal_window = temporal_window
        self.state_channels = state_channels

        if temporal_window > 0:
            # Wrap with temporal window
            self._windowed = TemporalWindowDataset(
                base_dataset=self,
                K=temporal_window,
            )
        else:
            self._windowed = None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._windowed is not None:
            return self._windowed[idx]
        return super().__getitem__(idx)

    def __len__(self) -> int:
        return len(self._windowed) if self._windowed else super().__len__()
