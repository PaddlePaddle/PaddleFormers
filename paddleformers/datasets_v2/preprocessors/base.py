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

"""Base preprocessor for datasets_v2.

Corresponds to ms-swift's RowPreprocessor (preprocessor/core.py:25).
Designed to work with HuggingFace Dataset.map().
"""

import logging
import traceback
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset as HfMapDataset

from ..schema import (
    DATASET_TYPE,
    STANDARD_KEYS,
    cast_images,
    cast_media_list,
    check_messages,
)

logger = logging.getLogger(__name__)


class BasePreprocessor:
    """Base class for all dataset preprocessors.

    Subclasses must implement `preprocess(row) -> row | list[row] | None`.

    Usage:
        preprocessor = MyPreprocessor(columns={'instruction': 'query'})
        dataset = preprocessor(dataset)
    """

    DEFAULT_COLUMN_ALIASES = {
        "image": "images",
        "video": "videos",
        "audio": "audios",
    }

    def __init__(
        self,
        *,
        columns: Optional[Dict[str, str]] = None,
        traceback_limit: int = 10,
    ) -> None:
        self.columns = {**self.DEFAULT_COLUMN_ALIASES, **(columns or {})}
        self._traceback_counter = 0
        self.traceback_limit = traceback_limit

    def preprocess(self, row: Dict[str, Any]) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """Process a single row. Must be implemented by subclasses.

        Returns:
            - A dict: the processed row
            - A list of dicts: one row expanded into multiple
            - None: row is skipped
        """
        raise NotImplementedError

    def prepare_dataset(self, dataset: DATASET_TYPE) -> DATASET_TYPE:
        """Hook for subclasses to do dataset-level setup (e.g. download media)."""
        return dataset

    # ================================================================
    # Batched processing (for Dataset.map)
    # ================================================================

    def _batched_preprocess(self, batched_row: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
        """Process a batch of rows. Used as the fn for Dataset.map(batched=True)."""
        rows = self._batched_to_rows(batched_row)
        new_rows = []

        for row in rows:
            try:
                result = self.preprocess(row)
                if result is None:
                    result = []
                if isinstance(result, dict):
                    result = [result]
                for r in result:
                    self._check_and_cast(r)
                new_rows += result
            except Exception:
                if strict:
                    raise
                if self._traceback_counter < self.traceback_limit:
                    logger.warning(f"Row preprocessing error (row will be skipped):\n{traceback.format_exc()}")
                    self._traceback_counter += 1

        res = self._rows_to_batched(new_rows)
        if len(res) == 0:
            res["messages"] = []
        return res

    # ================================================================
    # __call__: the main entry point
    # ================================================================

    def __call__(
        self,
        dataset: DATASET_TYPE,
        *,
        num_proc: int = 1,
        batch_size: int = 1000,
        strict: bool = False,
    ) -> DATASET_TYPE:
        """Apply this preprocessor to a dataset.

        Args:
            dataset: HuggingFace Dataset or IterableDataset
            num_proc: number of parallel workers for map
            batch_size: rows per batch in map
            strict: if True, raise on first error instead of skipping
        """
        dataset = self._rename_columns(dataset, self.columns)
        dataset = self.prepare_dataset(dataset)

        map_kwargs = {
            "batched": True,
            "batch_size": batch_size,
            "fn_kwargs": {"strict": strict},
            "remove_columns": self._columns_to_remove(dataset),
        }
        if isinstance(dataset, HfMapDataset):
            map_kwargs["num_proc"] = num_proc

        dataset = dataset.map(self._batched_preprocess, **map_kwargs)
        return dataset

    # ================================================================
    # Checks and casting (per row)
    # ================================================================

    @staticmethod
    def _check_and_cast(row: Dict[str, Any]) -> None:
        """Validate and normalize a single processed row in-place."""
        if "messages" in row:
            check_messages(row["messages"])

        for key in ["images", "rejected_images"]:
            if key in row and row[key] is not None:
                row[key] = cast_images(row[key])

        for key in ["videos", "audios"]:
            if key in row and row[key] is not None:
                row[key] = cast_media_list(row[key])

    # ================================================================
    # Column utilities
    # ================================================================

    @staticmethod
    def _rename_columns(dataset: DATASET_TYPE, columns: Dict[str, str]) -> DATASET_TYPE:
        """Rename columns, skipping those not present in dataset."""
        col_renames = {k: v for k, v in columns.items() if k in dataset.column_names and k != v}
        col_renames = {k: v for k, v in col_renames.items() if v not in dataset.column_names}
        return dataset.rename_columns(col_renames) if col_renames else dataset

    @staticmethod
    def _columns_to_remove(dataset: DATASET_TYPE) -> List[str]:
        """Get list of columns to remove after map (non-standard columns)."""
        return [c for c in dataset.column_names if c not in STANDARD_KEYS]

    # ================================================================
    # Batch <-> rows conversion
    # ================================================================

    @staticmethod
    def _batched_to_rows(batched_row: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert a batched dict to a list of row dicts."""
        return [dict(zip(batched_row, vals)) for vals in zip(*batched_row.values())]

    @staticmethod
    def _rows_to_batched(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert a list of row dicts to a batched dict."""
        if not rows:
            return {}
        all_keys = set().union(*rows)
        return {k: [row.get(k) for row in rows] for k in all_keys}
