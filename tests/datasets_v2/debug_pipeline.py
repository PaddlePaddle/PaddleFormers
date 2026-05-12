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

"""Debug script: observe the full dataset loading pipeline step by step.

Usage:
  1. Set breakpoints at the print() lines below
  2. Run via VSCode debugger using "Debug datasets_v2 Pipeline" config
  3. Or run directly: python tests/datasets_v2/debug_pipeline.py
"""

import importlib
import json

# Workaround for broken torchcodec
_original_find_spec = importlib.util.find_spec


def _patched_find_spec(name, *args, **kwargs):
    if name == "torchcodec":
        return None
    return _original_find_spec(name, *args, **kwargs)


importlib.util.find_spec = _patched_find_spec

# ============================================================
# Step 0: Prepare test data
# ============================================================

test_file = "/tmp/debug_pipeline_data.jsonl"
test_data = [
    {"query": "什么是机器学习?", "response": "机器学习是人工智能的一个分支。"},
    {"query": "1+1=?", "response": "2", "system": "你是一个数学助手"},
    {"query": "hello", "response": "你好"},
]

with open(test_file, "w", encoding="utf-8") as f:
    for row in test_data:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"=== Test data written to: {test_file} ===")
print(f"    Rows: {len(test_data)}")
print()

# ============================================================
# Step 1: Raw loading (no preprocessing)
# ============================================================

from paddleformers.datasets_v2.loaders import _load_local_file

raw_ds = _load_local_file(test_file)
print("=== Step 1: Raw loaded dataset ===")
print(f"    Columns: {raw_ds.column_names}")
print(f"    Num rows: {len(raw_ds)}")
for i, row in enumerate(raw_ds):
    print(f"    Row {i}: {row}")
print()

# ============================================================
# Step 2: AutoPreprocessor detects format
# ============================================================

from paddleformers.datasets_v2.preprocessors import AutoPreprocessor

auto = AutoPreprocessor()

# See what preprocessor it picks
from paddleformers.datasets_v2.preprocessors.base import BasePreprocessor

renamed_ds = BasePreprocessor._rename_columns(raw_ds, auto.columns)
print("=== Step 2: After column rename ===")
print(f"    Columns: {renamed_ds.column_names}")
for i, row in enumerate(renamed_ds):
    print(f"    Row {i}: {row}")
print()

# Check which preprocessor AutoPreprocessor would choose
preprocessor = auto._get_preprocessor(renamed_ds)
print(f"    Selected preprocessor: {type(preprocessor).__name__}")
print()

# ============================================================
# Step 3: Apply preprocessing row by row (manual simulation)
# ============================================================

print("=== Step 3: Row-by-row preprocessing ===")
for i in range(len(renamed_ds)):
    row = dict(renamed_ds[i])  # Make a mutable copy
    print(f"  [Before] Row {i}: {row}")
    result = preprocessor.preprocess(row)
    print(f"  [After]  Row {i}: {result}")
    print()

# ============================================================
# Step 4: Full pipeline via load_dataset
# ============================================================

from paddleformers.datasets_v2 import load_dataset

final_ds = load_dataset(test_file)
print("=== Step 4: Final dataset (full pipeline) ===")
print(f"    Columns: {final_ds.column_names}")
print(f"    Num rows: {len(final_ds)}")
for i, row in enumerate(final_ds):
    print(f"    Row {i}: {row}")
print()

# ============================================================
# Step 5: With registered dataset + sampling
# ============================================================

from paddleformers.datasets_v2 import DatasetMeta, register_dataset

register_dataset(
    DatasetMeta(
        name="debug_test",
        path=test_file,
        preprocessor="response",
        tags=["debug"],
    )
)

sampled_ds = load_dataset("debug_test#2")
print("=== Step 5: Registered + sampled (2 rows) ===")
print(f"    Columns: {sampled_ds.column_names}")
print(f"    Num rows: {len(sampled_ds)}")
for i, row in enumerate(sampled_ds):
    print(f"    Row {i}: {row}")

print("\n=== Pipeline debug complete ===")
