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

"""Extra preprocessors: thin wrappers over the base preprocessors.

Contains AlpacaPreprocessor and other lightweight derived classes.
"""

from typing import Any, Dict, Optional

from .response import ResponsePreprocessor


class AlpacaPreprocessor(ResponsePreprocessor):
    """Preprocessor for Alpaca-style datasets (instruction/input/output).

    Concatenates instruction + input into query, then delegates to ResponsePreprocessor.

    Expected input:
        {"instruction": "...", "input": "...", "output": "..."}
    """

    def __init__(self, *, columns: Optional[Dict[str, str]] = None, **kwargs) -> None:
        super().__init__(columns=columns, **kwargs)
        # Remove these from column rename map — we handle them manually in preprocess
        for key in ("instruction", "input", "output"):
            self.columns.pop(key, None)

    @staticmethod
    def concat_inst_input(instruction: Optional[str], input_: Optional[str]) -> str:
        if instruction and input_:
            return f"{instruction}\n{input_}"
        query = instruction or input_
        assert isinstance(query, str), f"query must be str, got: {query}"
        return query

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        instruction = row.pop("instruction", None)
        input_ = row.pop("input", None)
        output = row.pop("output", None)
        if output is not None:
            row["response"] = output
        row["query"] = self.concat_inst_input(instruction, input_)
        return super().preprocess(row)
