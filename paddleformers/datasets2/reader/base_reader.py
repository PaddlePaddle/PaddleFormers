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
""" Basic data reader implement. """

import os
from abc import ABC, abstractmethod

from .convertor import erniekit_convertor
from .io import load_csv, load_json, load_jsonl, load_parquet, load_txt
from .download_manager import HuggingFaceDownload


class BaseReader(ABC):
    def __init__(self, file_path, file_type, shuffle_file=True):
        self._file_path = file_path
        self._file_type = file_type  # erniekit, alpaca, ...
        self._shuffle_file = shuffle_file
        self.loader_map = {
            ".json": load_json,
            ".jsonl": load_jsonl,
            ".txt": load_txt,
            ".csv": load_csv,
            ".parquet": load_parquet,
        }
        self.convertor_map = {
            "erniekit": erniekit_convertor,
        }

    @abstractmethod
    def read(self):
        pass

    def _get_extension(self):
        _, ext = os.path.splitext(self._file_path)
        return ext.lower()

