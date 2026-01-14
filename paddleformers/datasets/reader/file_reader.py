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

import copy
import json
import os

from paddle.io import IterableDataset

from paddleformers.utils.log import logger

from .convertor import erniekit_convertor, messages_convertor
from .download_manager import HuggingFaceDownload
from .io import load_csv, load_json, load_parquet, load_txt

DATA_INFO_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data_info.json")
DATASET_WORKROOT = os.getenv("DATASET_WORKROOT", "/root/.cache/paddleformers")
DATASET_DOWNLOAD_ROOT = os.path.join(DATASET_WORKROOT, "download")


class BaseReader(IterableDataset):
    """Basic data reader implement."""

    def __init__(self, file_path, file_type, shuffle_file=True, split_multi_turn=False, template_backend="jinja"):
        self._file_path = file_path
        self._file_type = file_type  # erniekit, alpaca, ...
        self._shuffle_file = shuffle_file
        self._split_multi_turn = split_multi_turn
        self._template_backend = template_backend
        self.loader_map = {
            ".json": load_json,
            ".jsonl": load_json,
            ".txt": load_txt,
            ".csv": load_csv,
            ".parquet": load_parquet,
        }
        self.convertor_map = {
            "erniekit": erniekit_convertor,
            "messages": messages_convertor,
        }


class FileReader(BaseReader):
    def __init__(self, file_path, file_type, shuffle_file=True, split_multi_turn=False, template_backend="jinja"):
        super().__init__(
            file_path=file_path,
            file_type=file_type,
            shuffle_file=shuffle_file,
            split_multi_turn=split_multi_turn,
            template_backend=template_backend,
        )

    def __iter__(self):
        ext = self._get_extension()

        # load file
        if ext not in self.loader_map:
            raise ValueError(f"Unsupported file extension: {ext}. Supported extension are: {self.loader_map.keys()}")
        res = self.loader_map[ext](self._file_path)

        # data preprocess
        if self._file_type not in self.convertor_map:
            raise ValueError(
                f"Unsupported file type: {self._file_type}. Supported types are: {self.convertor_map.keys()}"
            )
        for item in res:
            try:
                convert_data = self.convertor_map[self._file_type](item)
                checked_data = self.data_check(convert_data)
            except Exception as e:
                logger.warning(f"preprocess data error: {e}, data: {str(item)[:30]}")
                continue
            if not checked_data:
                # ignore invalid example
                continue

            if self._split_multi_turn:
                assistant_index = 0
                for index, turn in enumerate(checked_data["messages"]):
                    if "assistant" in turn["role"] and checked_data["label"][assistant_index]:
                        new_data = copy.deepcopy(checked_data)
                        new_data["messages"] = checked_data["messages"][: index + 1]
                        new_data["label"] = [1]
                        yield new_data
                        assistant_index += 1
            else:
                yield checked_data

    def _get_extension(self):
        _, ext = os.path.splitext(self._file_path)
        return ext.lower()

    def data_check(self, data):
        if not data:
            return None

        if len(data["messages"]) == 0:
            raise ValueError("Ignore example with empty messages.")

        if self._template_backend != "jinja":
            ROLE_MAPPING = {
                "tool": "observation",
                "tool_response": "observation",
                "tool_call": "function",
                "tool_calls": "function",
                "function_call": "function",
                "function_calls": "function",
            }

            key_list = ["messages", "chosen_response", "rejected_response"]

            for key in key_list:
                if key not in data:
                    continue

                for item in data[key]:
                    # Update role names using the mapping
                    if item["role"] in ROLE_MAPPING:
                        item["role"] = ROLE_MAPPING[item["role"]]

                    # Convert content to string if needed
                    if item["role"] in ("observation", "function") and not isinstance(item["content"], str):
                        item["content"] = json.dumps(item["content"])

                    # Convert tool_calls to string if present and not already a string
                    if "tool_calls" in item and not isinstance(item["tool_calls"], str):
                        item["tool_calls"] = json.dumps(item["tool_calls"])

        # Convert the content of tool list into a string
        if "tools" in data and not isinstance(data["tools"], str):
            data["tools"] = json.dumps(data["tools"], ensure_ascii=False)

        # If no label is input, it means each response needs to be learned.
        if "label" not in data:
            data["label"] = [
                1 for turn in data["messages"] if ("assistant" in turn["role"] or "function" in turn["role"])
            ]

        system = ""
        if self._template_backend != "jinja" and "system" in data["messages"][0]["role"]:
            # extract system message when template_backend is not jinja
            system = data["messages"][0]["content"]
            if not isinstance(system, str):
                raise ValueError("System field must be a string.")
            data["messages"] = data["messages"][1:]
        data["system"] = system

        # Convert the relative paths of multimode data into absolute paths
        if "images" in data:
            for idx in range(len(data["images"])):
                if data["images"][idx].startswith("http") or os.path.isabs(data["images"][idx]):
                    pass
                else:
                    data["images"][idx] = os.path.join(os.path.dirname(self._file_path), data["images"][idx])
        if "videos" in data:
            for idx in range(len(data["videos"])):
                if data["videos"][idx].startswith("http") or os.path.isabs(data["videos"][idx]):
                    pass
                else:
                    data["videos"][idx] = os.path.join(os.path.dirname(self._file_path), data["videos"][idx])
        if "audios" in data:
            for idx in range(len(data["audios"])):
                if data["audios"][idx].startswith("http") or os.path.isabs(data["audios"][idx]):
                    pass
                else:
                    data["audios"][idx] = os.path.join(os.path.dirname(self._file_path), data["audios"][idx])

        return data


class FileListReader(BaseReader):
    def __init__(self, file_path, file_type, shuffle_file=True, split_multi_turn=False, template_backend="jinja"):
        if not os.path.isdir(file_path):
            raise ValueError(f"Directory not found: {file_path}")
        super().__init__(
            file_path=file_path,
            file_type=file_type,
            shuffle_file=shuffle_file,
            split_multi_turn=split_multi_turn,
            template_backend=template_backend,
        )

    def __iter__(self):
        for file_path in self._get_files():
            # all files under the path must be of the same data type
            reader = FileReader(
                file_path, self._file_type, self._shuffle_file, self._split_multi_turn, self._template_backend
            )
            yield from reader

    def _get_files(self):
        files = []
        for filename in os.listdir(self._file_path):
            file_path = os.path.join(self._file_path, filename)
            if os.path.isfile(file_path):
                files.append(file_path)
        return files


def get_hf_dataset_config(file_path):
    with open(DATA_INFO_FILE) as fp:
        hf_repo_config_map = json.load(fp)
    hf_dataset_config = hf_repo_config_map.get(file_path, None)
    return hf_dataset_config


class HuggingFaceReader(BaseReader):
    def __init__(
        self, file_path, file_type="alpaca", shuffle_file=True, split_multi_turn=False, template_backend="jinja"
    ):
        # download
        config_map = get_hf_dataset_config(file_path)
        if config_map is not None:
            # download hf dataset
            download_dir = os.path.join(DATASET_DOWNLOAD_ROOT, file_path)
            HuggingFaceDownload(file_path, download_dir)
            # read hf data file
            file_name = config_map.get("file_name", "")
            download_file_path = os.path.join(download_dir, file_name)
            download_file_type = config_map.get("formatting", file_type)
            if os.path.isdir(download_file_path):
                self.file_reader = FileListReader(
                    download_file_path,
                    download_file_type,
                    shuffle_file,
                    split_multi_turn,
                    template_backend,
                )
            else:
                self.file_reader = FileReader(
                    download_file_path, download_file_type, shuffle_file, split_multi_turn, template_backend
                )
        else:
            raise ValueError(f"Unsupported huggingface dataset {file_path}")

    def __iter__(self):
        yield from self.file_reader
