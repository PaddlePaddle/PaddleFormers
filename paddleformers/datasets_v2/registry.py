# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Dataset registry: metadata definitions and registration.

Provides a central registry for known datasets, enabling lookup by name
and bulk registration from JSON configuration files.

Corresponds to ms-swift's dataset/register.py + dataset/dataset_meta.py.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

from .schema import DATASET_TYPE

logger = logging.getLogger(__name__)


@dataclass
class DatasetMeta:
    """Metadata descriptor for a registered dataset.

    Attributes:
        name: Unique identifier for the dataset in the registry.
        path: Local file/directory path or HuggingFace Hub repo ID.
              If None, `name` is used as the Hub ID when loading.
        subset: HuggingFace dataset config/subset name.
        split: Default split to load.
        preprocessor: Preprocessor to apply after loading.
            - "auto": use AutoPreprocessor (default)
            - "messages": use MessagesPreprocessor
            - "response": use ResponsePreprocessor
            - "alpaca": use AlpacaPreprocessor
            - None: skip preprocessing
            - A callable: used directly
        columns: Column rename mapping {src_name: dst_name}.
        tags: Metadata tags for categorization.
    """

    name: str
    path: Optional[str] = None
    subset: Optional[str] = None
    split: str = "train"
    preprocessor: Union[str, Callable[..., DATASET_TYPE], None] = "auto"
    columns: Optional[Dict[str, str]] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class DatasetSpec:
    """Parsed result of a dataset string specification.

    Syntax: "dataset_name_or_path#sample_count"

    Examples:
        "alpaca"                   -> name="alpaca", sample=None
        "alpaca#500"               -> name="alpaca", sample=500
        "/path/to/data.jsonl#1000" -> name="/path/to/data.jsonl", sample=1000
    """

    name: str
    sample: Optional[int] = None


# ============================================================
# Module-level registry
# ============================================================

_DATASET_REGISTRY: Dict[str, DatasetMeta] = {}


# ============================================================
# Registration functions
# ============================================================


def register_dataset(meta: DatasetMeta, *, exist_ok: bool = False) -> None:
    """Register a DatasetMeta into the global registry.

    Args:
        meta: The dataset metadata to register.
        exist_ok: If True, silently overwrites existing entries.

    Raises:
        ValueError: If the name is already registered and exist_ok=False.
    """
    if not exist_ok and meta.name in _DATASET_REGISTRY:
        raise ValueError(f"Dataset '{meta.name}' is already registered. " f"Use exist_ok=True to overwrite.")
    _DATASET_REGISTRY[meta.name] = meta


def get_dataset_meta(name: str) -> Optional[DatasetMeta]:
    """Look up a registered dataset by name.

    Returns:
        The DatasetMeta if found, otherwise None.
    """
    return _DATASET_REGISTRY.get(name)


def list_datasets(tag: Optional[str] = None) -> List[str]:
    """List all registered dataset names, optionally filtered by tag.

    Args:
        tag: If provided, only return datasets containing this tag.

    Returns:
        Sorted list of dataset names.
    """
    if tag is None:
        return sorted(_DATASET_REGISTRY.keys())
    return sorted(name for name, meta in _DATASET_REGISTRY.items() if tag in meta.tags)


def register_dataset_info(json_path: str) -> List[DatasetMeta]:
    """Bulk-register datasets from a JSON file.

    The JSON file should contain a list of objects with fields matching
    DatasetMeta: name, path, subset, split, preprocessor, columns, tags.
    Only "name" is required; others use defaults.

    Relative paths in "path" are resolved relative to the JSON file's directory.

    Args:
        json_path: Path to the JSON file.

    Returns:
        List of registered DatasetMeta objects.

    Raises:
        FileNotFoundError: If json_path does not exist.
        ValueError: If an entry is missing the "name" field.
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Dataset info file not found: {json_path}")

    base_dir = os.path.dirname(os.path.abspath(json_path))

    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        raise ValueError(f"Expected a JSON list in {json_path}, got {type(entries).__name__}")

    registered = []
    for entry in entries:
        if "name" not in entry:
            raise ValueError(f"Dataset entry missing 'name' field: {entry}")

        # Resolve relative paths
        path = entry.get("path")
        if path and not os.path.isabs(path):
            # Hub IDs contain "/" but are not relative paths (e.g. "org/repo")
            # Only resolve paths that don't contain "/" or start with "./"
            is_hub_id = "/" in path and not path.startswith(("./", "../"))
            if not is_hub_id:
                path = os.path.join(base_dir, path)

        meta = DatasetMeta(
            name=entry["name"],
            path=path,
            subset=entry.get("subset"),
            split=entry.get("split", "train"),
            preprocessor=entry.get("preprocessor", "auto"),
            columns=entry.get("columns"),
            tags=entry.get("tags", []),
        )
        register_dataset(meta, exist_ok=True)
        registered.append(meta)

    logger.info(f"Registered {len(registered)} datasets from {json_path}")
    return registered


# ============================================================
# String syntax parsing
# ============================================================


def parse_dataset_string(dataset_str: str) -> DatasetSpec:
    """Parse a dataset specification string.

    Supports syntax: "name_or_path#N" where #N is optional sample count.

    Args:
        dataset_str: The dataset string to parse.

    Returns:
        A DatasetSpec with parsed name and optional sample count.

    Raises:
        ValueError: If the sample count is not a positive integer.
    """
    dataset_str = dataset_str.strip()
    if "#" in dataset_str:
        name, sample_str = dataset_str.rsplit("#", 1)
        try:
            sample = int(sample_str)
        except ValueError:
            raise ValueError(
                f"Invalid sample count '{sample_str}' in '{dataset_str}'. " f"Expected a positive integer after '#'."
            )
        if sample <= 0:
            raise ValueError(f"Sample count must be positive, got {sample} in '{dataset_str}'.")
        return DatasetSpec(name=name.strip(), sample=sample)
    return DatasetSpec(name=dataset_str)


# ============================================================
# Built-in registry loading
# ============================================================

_BUILTIN_LOADED = False


def _load_builtin_registry() -> None:
    """Load the built-in dataset_info.json shipped with this package.

    Called once on first import. Safe to call multiple times (idempotent).
    """
    global _BUILTIN_LOADED
    if _BUILTIN_LOADED:
        return
    _BUILTIN_LOADED = True

    builtin_path = os.path.join(os.path.dirname(__file__), "dataset_info.json")
    if os.path.isfile(builtin_path):
        register_dataset_info(builtin_path)
        logger.debug(f"Loaded built-in dataset registry from {builtin_path}")
    else:
        logger.debug("No built-in dataset_info.json found, skipping.")
