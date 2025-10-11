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

import json
import os

from .base_reader import BaseReader
from .download_manager import HuggingFaceDownload
from .file_reader import FileListReader, FileReader

DATA_INFO_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data_info.json")
DATASET_WORKROOT = os.getenv("DATASET_WORKROOT", "/root/.cache/paddleformers")
DATASET_DOWNLOAD_ROOT = os.path.join(DATASET_WORKROOT, "download")


def get_hf_dataset_config(file_path):
    with open(DATA_INFO_FILE) as fp:
        hf_repo_config_map = json.load(fp)
    hf_dataset_config = hf_repo_config_map.get(file_path, None)
    return hf_dataset_config


class HuggingFaceReader(BaseReader):
    def __init__(self, file_path, file_type="alpaca", shuffle_file=True):
        # download
        config_map = get_hf_dataset_config(file_path)
        if config_map is not None:
            HuggingFaceDownload(file_path)
            download_dir = os.path.join(DATASET_DOWNLOAD_ROOT, file_path)
            file_name = config_map.get("file_name", "")
            download_file_path = os.path.join(download_dir, file_name)
            download_file_type = config_map.get("formatting", file_type)
            if os.path.isdir(download_file_path):
                self.file_reader = FileListReader(download_file_path, download_file_type, shuffle_file)
            else:
                self.file_reader = FileReader(download_file_path, download_file_type, shuffle_file)

            self.file_reader.__init__()

        else:
            raise ValueError(f"Unsupported huggingface dataset {file_path}")

    def read(self):
        return self.file_reader.read()
