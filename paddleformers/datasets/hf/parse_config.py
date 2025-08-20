# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
"""Config for parse dataset to same format"""
import os

DATASET_WORKROOT = os.getenv("ERNIE_DATASET_WORKROOT", os.path.abspath(os.path.join(os.path.dirname(__file__))))
DATASET_DOWNLOAD_ROOT = os.path.join(DATASET_WORKROOT, "download")
DATASET_OUTPUT_ROOT = os.path.join(DATASET_WORKROOT, "output")

DATA_INFO_FILE = os.path.join(DATASET_WORKROOT, "data_info.json")

DEFAULT_DOC_FORMATTING = "json"

DEFAULT_ALPACA_COLUMNS_MAPPING = {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system",
}
DEFAULT_COLUMN_VALUE_MAPPING = {"prompt": "", "query": "", "response": ""}
DEFAULT_ALPACA_DPO_COLUMNS_MAPPING = {
    "prompt": "instruction",
    "query": "input",
    "chosen": "chosen",
    "rejected": "rejected",
    "system": "system",
}
DEFAULT_ALPACA_DPO_COLUMNS_VALUE_MAPPING = {
    "prompt": "",
    "query": "",
    "chosen": "",
    "rejected": "",
}
DEFAULT_DATASET_COLUMNS_MAPPING = {"alpaca": DEFAULT_ALPACA_COLUMNS_MAPPING}

DEFAULT_OUTPUT_JSON_INDENT = 2

ALPACA_COLUMNS_EMPTY_CHECK_LIST = ["prompt", "query", "response"]

DEBUG_DATASET_OUTPUT_FORMATTED_FILE = True
