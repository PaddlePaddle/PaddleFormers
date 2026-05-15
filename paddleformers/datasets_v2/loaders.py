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

"""Dataset loading: local files, directories, and HuggingFace Hub.

Provides a unified load_dataset() entry point that handles:
- Local files (json/jsonl/csv/tsv/parquet/arrow/txt)
- Local directories (auto-detect data files)
- HuggingFace Hub datasets (by repo_id)
- Streaming mode (IterableDataset)
- Automatic preprocessing via registry metadata
"""

import logging
import os
from typing import Dict, List, Optional, Union

from datasets import Dataset as HfMapDataset
from datasets import IterableDataset as HfIterableDataset  # noqa: F401
from datasets import load_dataset as hf_load_dataset

from .registry import DatasetMeta, get_dataset_meta, parse_dataset_string
from .schema import DATASET_TYPE

logger = logging.getLogger(__name__)

# ============================================================
# File extension mapping
# ============================================================

_EXT_TO_FORMAT = {
    "json": "json",
    "jsonl": "json",
    "csv": "csv",
    "tsv": "csv",
    "parquet": "parquet",
    "arrow": "arrow",
    "txt": "text",
}


# ============================================================
# Internal helpers
# ============================================================


def _detect_file_format(path: str) -> str:
    """Detect the HF-compatible format name from file extension."""
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    fmt = _EXT_TO_FORMAT.get(ext)
    if fmt is None:
        raise ValueError(f"Unsupported file format '.{ext}' for: {path}. " f"Supported: {list(_EXT_TO_FORMAT.keys())}")
    return fmt


def _resolve_source(name_or_path: str) -> str:
    """Determine the source type of a dataset identifier.

    Returns one of: "file", "directory", "hub"
    """
    if os.path.isfile(name_or_path):
        return "file"
    if os.path.isdir(name_or_path):
        return "directory"
    return "hub"


def _load_local_file(
    path: str,
    *,
    split: str = "train",
    streaming: bool = False,
    **kwargs,
) -> DATASET_TYPE:
    """Load a dataset from a local file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    fmt = _detect_file_format(path)
    load_kwargs = {"split": split, "streaming": streaming, **kwargs}

    if fmt == "csv":
        load_kwargs.setdefault("na_filter", False)
        if path.endswith(".tsv"):
            load_kwargs.setdefault("delimiter", "\t")

    dataset = hf_load_dataset(fmt, data_files=path, **load_kwargs)
    logger.info(f"Loaded local file: {path} (format={fmt}, streaming={streaming})")
    return dataset


def _load_local_directory(
    directory: str,
    *,
    split: str = "train",
    streaming: bool = False,
    **kwargs,
) -> DATASET_TYPE:
    """Load a dataset from a local directory.

    If the directory contains a loading script or dataset_infos.json,
    delegates directly to HF. Otherwise, auto-detects data files.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Check if HF can handle the directory directly (has script or metadata)
    has_script = any(f.endswith(".py") for f in os.listdir(directory))
    has_info = os.path.isfile(os.path.join(directory, "dataset_infos.json"))
    has_info = has_info or os.path.isfile(os.path.join(directory, "dataset_info.json"))

    if has_script or has_info:
        dataset = hf_load_dataset(directory, split=split, streaming=streaming, **kwargs)
        logger.info(f"Loaded directory (HF script/metadata): {directory}")
        return dataset

    # Auto-detect data files by extension
    data_files = []
    detected_fmt = None
    for fname in sorted(os.listdir(directory)):
        ext = os.path.splitext(fname)[1].lstrip(".").lower()
        if ext in _EXT_TO_FORMAT:
            if detected_fmt is None:
                detected_fmt = _EXT_TO_FORMAT[ext]
            elif _EXT_TO_FORMAT[ext] == detected_fmt:
                pass
            else:
                # Mixed formats: use the first one found
                continue
            data_files.append(os.path.join(directory, fname))

    if not data_files:
        raise ValueError(
            f"No supported data files found in directory: {directory}. "
            f"Supported extensions: {list(_EXT_TO_FORMAT.keys())}"
        )

    load_kwargs = {"split": split, "streaming": streaming, **kwargs}
    if detected_fmt == "csv":
        load_kwargs.setdefault("na_filter", False)

    dataset = hf_load_dataset(detected_fmt, data_files=data_files, **load_kwargs)
    logger.info(f"Loaded directory: {directory} ({len(data_files)} files, format={detected_fmt})")
    return dataset


def _load_hub_dataset(
    repo_id: str,
    *,
    subset: Optional[str] = None,
    split: str = "train",
    streaming: bool = False,
    token: Optional[str] = None,
    **kwargs,
) -> DATASET_TYPE:
    """Load a dataset from HuggingFace Hub."""
    load_kwargs = {
        "split": split,
        "streaming": streaming,
        "trust_remote_code": True,
        **kwargs,
    }
    if subset is not None:
        load_kwargs["name"] = subset
    if token is not None:
        load_kwargs["token"] = token

    dataset = hf_load_dataset(repo_id, **load_kwargs)
    logger.info(f"Loaded from Hub: {repo_id} (subset={subset}, split={split}, streaming={streaming})")
    return dataset


def _get_preprocessor(meta: DatasetMeta):
    """Resolve preprocessor from DatasetMeta.

    Returns a callable preprocessor instance, or None if no preprocessing.
    """
    preprocessor = meta.preprocessor
    if preprocessor is None:
        return None
    if callable(preprocessor):
        return preprocessor

    # Lazy import to avoid circular dependencies
    from .preprocessors import (
        AlpacaPreprocessor,
        AutoPreprocessor,
        MessagesPreprocessor,
        ResponsePreprocessor,
    )

    _PREPROCESSOR_MAP = {
        "auto": AutoPreprocessor,
        "messages": MessagesPreprocessor,
        "response": ResponsePreprocessor,
        "alpaca": AlpacaPreprocessor,
    }

    if preprocessor not in _PREPROCESSOR_MAP:
        raise ValueError(f"Unknown preprocessor '{preprocessor}'. " f"Available: {list(_PREPROCESSOR_MAP.keys())}")

    cls = _PREPROCESSOR_MAP[preprocessor]
    columns = meta.columns or {}
    return cls(columns=columns)


# ============================================================
# Main entry points
# ============================================================


def load_dataset(
    dataset: str,
    *,
    split: Optional[str] = None,
    subset: Optional[str] = None,
    streaming: bool = False,
    token: Optional[str] = None,
    num_proc: int = 1,
    preprocess: bool = True,
    columns: Optional[Dict[str, str]] = None,
    strict: bool = False,
) -> DATASET_TYPE:
    """Load a dataset from any source: local file, directory, or HuggingFace Hub.

    Supports:
    - Registered dataset names: "alpaca"
    - Local file paths: "/data/train.jsonl"
    - Local directories: "/data/my_dataset/"
    - HuggingFace Hub IDs: "tatsu-lab/alpaca"
    - Sample syntax: "dataset_name#500" (sample 500 rows after loading)

    Args:
        dataset: Dataset specification string.
        split: Split to load. Overrides registry default.
        subset: HF subset/config. Overrides registry default.
        streaming: If True, returns an IterableDataset.
        token: HuggingFace Hub token for private datasets.
        num_proc: Number of workers for preprocessing (map-style only).
        preprocess: Whether to apply the registered preprocessor.
        columns: Column rename mapping. Merged with registry columns.
        strict: If True, raise on preprocessing errors.

    Returns:
        The loaded (and optionally preprocessed) dataset.
    """
    # 1. Parse string syntax
    spec = parse_dataset_string(dataset)

    # 2. Registry lookup
    meta = get_dataset_meta(spec.name)

    # 3. Resolve effective parameters
    effective_split = split or (meta.split if meta else "train")
    effective_subset = subset or (meta.subset if meta else None)
    effective_path = (meta.path if meta else None) or spec.name

    # Merge column mappings: registry columns + explicit columns (explicit wins)
    effective_columns = {}
    if meta and meta.columns:
        effective_columns.update(meta.columns)
    if columns:
        effective_columns.update(columns)

    # 4. Determine source and load
    source_type = _resolve_source(effective_path)

    if source_type == "file":
        ds = _load_local_file(effective_path, split=effective_split, streaming=streaming)
    elif source_type == "directory":
        ds = _load_local_directory(effective_path, split=effective_split, streaming=streaming)
    else:
        ds = _load_hub_dataset(
            effective_path,
            subset=effective_subset,
            split=effective_split,
            streaming=streaming,
            token=token,
        )

    # 5. Apply preprocessing
    if preprocess:
        if meta and meta.preprocessor is None:
            # Explicitly disabled preprocessing via registry
            preprocessor = None
        elif meta:
            preprocessor = _get_preprocessor(meta)
        else:
            # No registry meta: default to AutoPreprocessor
            from .preprocessors import AutoPreprocessor

            preprocessor = AutoPreprocessor(columns=effective_columns)

        if preprocessor is not None:
            proc_num_proc = num_proc if not streaming else 1
            ds = preprocessor(ds, num_proc=proc_num_proc, strict=strict)

    # 6. Apply sampling
    if spec.sample is not None:
        if isinstance(ds, HfMapDataset):
            from .ops import sample_dataset

            ds = sample_dataset(ds, spec.sample)
        else:
            ds = ds.take(spec.sample)

    return ds


def load_datasets(
    datasets: Union[str, List[str]],
    *,
    split: Optional[str] = None,
    streaming: bool = False,
    token: Optional[str] = None,
    num_proc: int = 1,
    preprocess: bool = True,
    columns: Optional[Dict[str, str]] = None,
    strict: bool = False,
) -> DATASET_TYPE:
    """Load multiple datasets and concatenate them.

    Args:
        datasets: A single dataset string or list of dataset strings.
        split: Default split for all datasets.
        streaming: If True, all datasets are loaded in streaming mode.
        token: HuggingFace Hub token.
        num_proc: Number of workers for preprocessing.
        preprocess: Whether to apply preprocessors.
        columns: Column rename mapping applied to all datasets.
        strict: Strict preprocessing mode.

    Returns:
        A single concatenated dataset.
    """
    if isinstance(datasets, str):
        datasets = [datasets]

    loaded = []
    for ds_str in datasets:
        ds = load_dataset(
            ds_str,
            split=split,
            streaming=streaming,
            token=token,
            num_proc=num_proc,
            preprocess=preprocess,
            columns=columns,
            strict=strict,
        )
        loaded.append(ds)

    if len(loaded) == 1:
        return loaded[0]

    from .ops import concat_datasets

    return concat_datasets(loaded)
