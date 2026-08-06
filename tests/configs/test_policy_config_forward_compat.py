# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Loading checkpoints whose config.json predates a config-field removal.

A checkpoint persists every config field that existed when it was trained. If a
field is later deleted from the config dataclass, a strict parse rejects the whole
file and every older checkpoint silently becomes unloadable. These tests pin the
tolerant behaviour: unknown keys are dropped with a warning, known values survive,
and defaults are never overwritten by stale data.
"""

import json
import logging

import draccus
import pytest

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


def _encode(**kwargs):
    with draccus.config_type("json"):
        return draccus.encode(ACTConfig(**kwargs))


def _write_config(directory, **overrides):
    config = _encode(chunk_size=25, n_action_steps=25)
    config["type"] = "act"
    config.update(overrides)
    (directory / "config.json").write_text(json.dumps(config))
    return config


def test_unknown_fields_are_dropped_and_config_still_loads(tmp_path, caplog):
    """A field removed from ACTConfig must not make the checkpoint unloadable."""
    _write_config(
        tmp_path,
        temporal_ensemble_window=None,
        temporal_ensemble_post_grasp_window=None,
        temporal_ensemble_reopen_fraction=0.8,
        some_field_from_a_future_refactor=123,
    )
    with caplog.at_level(logging.WARNING):
        config = PreTrainedConfig.from_pretrained(str(tmp_path))

    assert config.type == "act"
    # Real values survive the filtering.
    assert config.chunk_size == 25
    assert config.n_action_steps == 25
    # The dropped keys are reported, so this can never fail silently.
    assert "temporal_ensemble_reopen_fraction" in caplog.text
    assert "some_field_from_a_future_refactor" in caplog.text


def test_known_fields_are_not_dropped(tmp_path, caplog):
    """A config with no stale keys must load without any warning."""
    _write_config(tmp_path)
    with caplog.at_level(logging.WARNING):
        config = PreTrainedConfig.from_pretrained(str(tmp_path))
    assert config.chunk_size == 25
    assert "ignoring" not in caplog.text


def test_dropped_field_falls_back_to_current_default(tmp_path):
    """Stale values must not leak into fields the class still defines."""
    default = ACTConfig().temporal_ensemble_coeff
    _write_config(tmp_path, removed_knob_that_no_longer_exists="stale")
    config = PreTrainedConfig.from_pretrained(str(tmp_path))
    assert config.temporal_ensemble_coeff == default
    assert not hasattr(config, "removed_knob_that_no_longer_exists")


@pytest.mark.parametrize("policy_type", ["not_a_registered_policy", ""])
def test_unresolvable_type_is_left_to_draccus(tmp_path, policy_type):
    """If the subclass cannot be resolved we must not silently filter anything."""
    config = _encode()
    config["type"] = policy_type
    assert PreTrainedConfig._drop_unknown_fields(dict(config), "test") == config
