# Copyright 2026 Jay Ranpariya. Thesis extension.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Relocate absolute asset paths recorded by a checkpoint onto the local machine.

A checkpoint trained on a cluster persists absolute paths to the frozen encoders
and tokenizers it was built from (for example a CLIP snapshot under
``/data/beegfs/...``). Those paths do not exist on a workstation, so the policy
cannot be constructed there at all -- neither for offline analysis nor for a
robot rollout.

Set ``LEROBOT_ASSET_SEARCH_PATH`` to one or more local directories (``:``
separated) holding copies of those assets. When a recorded absolute path is
missing, it is replaced by a directory of the *same basename* found in one of
those directories.

Matching on the exact basename is deliberate: HuggingFace snapshot directory
names embed the revision hash, so a different revision of the same encoder cannot
be silently substituted for the weights the policy was trained against. Nothing
on disk is rewritten -- checkpoints and their hashes are untouched.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

ASSET_SEARCH_ENV = "LEROBOT_ASSET_SEARCH_PATH"

logger = logging.getLogger(__name__)


def asset_search_dirs() -> list[Path]:
    """Directories to search, from ``LEROBOT_ASSET_SEARCH_PATH``."""
    raw = os.environ.get(ASSET_SEARCH_ENV, "")
    return [Path(part) for part in raw.split(os.pathsep) if part]


def relocate_asset_path(value: object) -> object:
    """Return a local stand-in for an unreachable absolute asset path.

    Values that are not absolute paths, or that already exist, are returned
    unchanged, so this is safe to apply indiscriminately.
    """
    if not isinstance(value, str) or not value.startswith("/"):
        return value
    original = Path(value)
    if original.exists():
        return value
    for directory in asset_search_dirs():
        candidate = directory / original.name
        if candidate.is_dir():
            logger.warning(
                "Asset path %s is unreachable; using local copy %s "
                "(matched by exact name, so the revision is preserved).",
                value,
                candidate,
            )
            return str(candidate)
    return value


def relocate_mapping(config: dict, key_suffixes: tuple[str, ...] = ()) -> dict:
    """Relocate string values in a flat config mapping.

    With ``key_suffixes``, only keys ending in one of them are considered;
    otherwise every string value is examined.
    """
    result = {}
    for key, value in config.items():
        if key_suffixes and not key.endswith(key_suffixes):
            result[key] = value
            continue
        result[key] = relocate_asset_path(value)
    return result
