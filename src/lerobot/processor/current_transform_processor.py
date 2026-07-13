#!/usr/bin/env python

# Copyright 2026 Jay Ranpariya. Thesis extension.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Inference-time current transform for the normalized ACT-M views (Rung-0).

The `act-m-{norm,sqrt,lognorm,binned}` dataset views are built offline by
`experiments/scripts/make_lognorm_dataset.py`, which bakes a per-channel
transform into `observation.state`'s current channels (indices 6..11) and
recomputes `meta.stats` in transform-space. A policy trained on such a view
therefore has a `NormalizerProcessorStep` whose mean/std live in transform
space -- but at deployment the robot streams *raw* current counts. Without a
matching transform the normalizer would see out-of-distribution values.

This step replays the SAME transform on the live current, inserted immediately
BEFORE the normalizer in the policy preprocessor, so hardware/offline eval is
consistent with training. Parameters come verbatim from the view's
`view_manifest.json` `transform` block (winsor_caps_counts, bin_edges_counts),
so the transform travels with the checkpoint. Positions (0..5) are untouched.

op semantics (must match make_lognorm_dataset.py exactly):
  none   : winsorize only (caps), normalizer's MEAN_STD does the scaling.
  log1p  : winsorize then log1p.
  sqrt   : winsorize then sqrt.
  bin    : per-channel ordinal level = digitize(edges) / len(edges), in [0,1]
           (normalization-free; the trained normalizer for a binned view is
           an approximate identity on these channels).
"""
from dataclasses import dataclass, field
from typing import Any

import torch

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="current_transform")
class CurrentTransformProcessorStep(ObservationProcessorStep):
    """Replay a baked current-view transform on live current, before normalization.

    Attributes:
        op: one of {"none", "log1p", "sqrt", "bin"}. "none" = winsorize-only.
        current_indices: indices into observation.state holding current (default 6..11).
        winsor_caps: per-channel upper cap in COUNT units (len == len(current_indices)),
            or None for no winsorize. Applied for every op except "bin".
        bin_edges: per-channel digitize edges in COUNT units (list of lists), for op="bin".
        state_key: observation key holding the flat state vector.
    """

    op: str = "none"
    current_indices: list[int] = field(default_factory=lambda: list(range(6, 12)))
    winsor_caps: list[float] | None = None
    bin_edges: list[list[float]] | None = None
    state_key: str = "observation.state"

    def __post_init__(self):
        if self.op not in ("none", "log1p", "sqrt", "bin"):
            raise ValueError(f"CurrentTransformProcessorStep: unknown op {self.op!r}")
        if self.op == "bin" and not self.bin_edges:
            raise ValueError("CurrentTransformProcessorStep: op='bin' requires bin_edges")
        if self.winsor_caps is not None and len(self.winsor_caps) != len(self.current_indices):
            raise ValueError("winsor_caps length must match current_indices")

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        s = observation.get(self.state_key)
        if s is None or not torch.is_tensor(s):
            return observation
        x = s.clone()
        idx = self.current_indices

        # winsorize (glitch clip) for every op except bin
        if self.op != "bin" and self.winsor_caps is not None:
            for k, j in enumerate(idx):
                x[..., j] = x[..., j].clamp(max=float(self.winsor_caps[k]))

        if self.op == "log1p":
            for j in idx:
                x[..., j] = torch.log1p(x[..., j])
        elif self.op == "sqrt":
            for j in idx:
                x[..., j] = torch.sqrt(x[..., j].clamp(min=0.0))
        elif self.op == "bin":
            for k, j in enumerate(idx):
                e = torch.tensor(self.bin_edges[k], dtype=x.dtype, device=x.device)
                levels = e.numel()
                # right=True matches np.digitize(..., right=False): index = #edges <= value
                lvl = torch.bucketize(x[..., j].contiguous(), e, right=True)
                x[..., j] = lvl.to(x.dtype) / levels

        observation[self.state_key] = x
        return observation

    def get_config(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "current_indices": list(self.current_indices),
            "winsor_caps": self.winsor_caps,
            "bin_edges": self.bin_edges,
            "state_key": self.state_key,
        }

    def transform_features(self, features):
        """Identity: the transform reshapes values in place, not the state shape."""
        return features
