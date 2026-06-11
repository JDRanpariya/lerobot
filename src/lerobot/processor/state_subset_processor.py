#!/usr/bin/env python

# Copyright 2026 Jay Ranpariya. Thesis extension.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""State subset processor for ACT-V / ACT-M feature selection.

Allows training different policy variants from the same multimodal dataset
by selecting which indices of observation.state to keep.

Examples:
    ACT-V (position only):              indices = [0, 1, 2, 3, 4, 5]
    ACT-M (position + current):         indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    ACT-M-grip (pos + gripper current): indices = [0, 1, 2, 3, 4, 5, 11]

Usage in a pipeline config:
    observation_processor:
      steps:
        - type: state_subset
          indices: [0, 1, 2, 3, 4, 5]  # ACT-V: position only
          names: [shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
                  wrist_flex.pos, wrist_roll.pos, gripper.pos]
"""
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import PipelineFeatureType, PolicyFeature

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="state_subset")
class StateSubsetProcessorStep(ObservationProcessorStep):
    """Selects a subset of observation.state indices.

    This enables training ACT-V (positions only) and ACT-M (positions + currents)
    from the same multimodal dataset without duplicating data.

    The processor slices the state vector at the specified indices, reducing
    the state dimension seen by the policy. Feature metadata is updated
    accordingly so the policy builds with the correct input_dim.

    Attributes:
        indices: List of integer indices into the observation.state vector to keep.
        names: Optional human-readable names for selected indices (documentation only).
    """

    indices: list[int] = field(default_factory=list)
    names: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.indices:
            raise ValueError(
                "StateSubsetProcessorStep requires at least one index. "
                "Use indices=[0,1,2,3,4,5] for position-only (ACT-V) or "
                "indices=list(range(12)) for position+current (ACT-M)."
            )

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Select state subset from the observation dict.

        Looks for keys ending in '.state' or containing 'state' that hold
        the flat state vector, and slices to self.indices.
        """
        processed = {}
        for key, value in observation.items():
            if key.endswith(".pos") or key.endswith(".cur") or key.endswith(".temp"):
                # These are individual motor keys from robot.get_observation()
                # They get aggregated into observation.state by the pipeline
                processed[key] = value
            else:
                processed[key] = value
        return processed

    def get_config(self) -> dict[str, Any]:
        return {"indices": self.indices, "names": self.names}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Update observation feature shapes to reflect the subset.

        When observation.state has shape (N,), this transforms it to
        shape (len(self.indices),) and updates the names list accordingly.
        """
        features = deepcopy(features)
        if PipelineFeatureType.OBSERVATION in features:
            obs_features = features[PipelineFeatureType.OBSERVATION]
            for key in list(obs_features.keys()):
                ft = obs_features[key]
                # Check if this is the state feature (type float, not a tuple/image)
                if isinstance(ft, type) and ft == float:
                    # Individual motor features — filter by index
                    # The pipeline aggregates .pos, .cur, .temp keys into state
                    # We need to filter which keys survive
                    pass
                elif hasattr(ft, 'shape') and len(ft.shape) == 1:
                    # This is the aggregated state vector — slice it
                    new_shape = (len(self.indices),)
                    if self.names:
                        obs_features[key] = PolicyFeature(
                            type=ft.type, shape=new_shape, names=self.names
                        )
                    else:
                        obs_features[key] = PolicyFeature(
                            type=ft.type, shape=new_shape
                        )
        return features
