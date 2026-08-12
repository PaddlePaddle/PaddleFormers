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

"""AutoPreprocessor: automatically selects the right preprocessor based on dataset columns."""

from typing import Dict, Optional

from ..schema import DATASET_TYPE
from .base import BasePreprocessor
from .extra import AlpacaPreprocessor, TextPreprocessor
from .messages import MessagesPreprocessor
from .response import ResponsePreprocessor


class AutoPreprocessor:
    """Auto-detect dataset format and dispatch to the appropriate preprocessor.

    Detection logic:
        1. Has 'messages' / 'conversation' / 'conversations' column → MessagesPreprocessor
        2. Has 'instruction' + 'input' columns → AlpacaPreprocessor
        3. Otherwise → ResponsePreprocessor
    """

    def __init__(self, *, columns: Optional[Dict[str, str]] = None, **kwargs) -> None:
        self.columns = columns or {}
        self.kwargs = kwargs

    def _get_preprocessor(self, dataset: DATASET_TYPE) -> BasePreprocessor:
        col_names = dataset.column_names
        # IterableDataset may have column_names=None; use features or peek first row
        if col_names is None:
            # Try .features first (non-destructive)
            features = getattr(dataset, "features", None)
            if features is not None:
                col_names = list(features.keys())
            else:
                # Last resort: peek first row. Note: this loses one row for
                # IterableDatasets without features, but such cases are rare
                # (HF streaming datasets always have features defined).
                first_row = next(iter(dataset))
                col_names = list(first_row.keys())
        for key in ["messages", "conversation", "conversations"]:
            if key in col_names:
                return MessagesPreprocessor(columns=self.columns, **self.kwargs)
        if "instruction" in col_names and "input" in col_names:
            return AlpacaPreprocessor(columns=self.columns, **self.kwargs)
        # Pure text pretrain datasets (has "text" but no SFT-characteristic columns)
        sft_indicators = {"query", "response", "instruction", "input", "output", "prompt", "answer"}
        if "text" in col_names and not sft_indicators.intersection(col_names):
            return TextPreprocessor(columns=self.columns, **self.kwargs)
        return ResponsePreprocessor(columns=self.columns, **self.kwargs)

    def __call__(
        self,
        dataset: DATASET_TYPE,
        *,
        num_proc: int = 1,
        batch_size: int = 1000,
        strict: bool = False,
    ) -> DATASET_TYPE:
        dataset = BasePreprocessor._rename_columns(dataset, self.columns)
        preprocessor = self._get_preprocessor(dataset)
        return preprocessor(dataset, num_proc=num_proc, batch_size=batch_size, strict=strict)
