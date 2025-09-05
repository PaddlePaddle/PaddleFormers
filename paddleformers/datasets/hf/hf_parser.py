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
""" HuggingFace datasets implement. """
import json
import os
import random

from huggingface_hub import snapshot_download
from paddle.io import IterableDataset

from paddleformers.datasets.hf import errors, parse_config


class BaseDatasetParser(IterableDataset):
    """Base class for file parser."""

    def __init__(
        self,
        file_path,
        formatting,
        doc_formatting,
        columns,
        process_fn=None,
        shuffle_file=False,
    ):
        super().__init__()
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.train_type = ""
        self.formatting = formatting
        self.doc_formatting = doc_formatting
        self.columns = columns
        self.r_columns = {}
        for k, v in self.columns.items():
            self.r_columns[v] = k

        self.data = []

        self.process_fn = process_fn
        self.shuffle_file = shuffle_file

    def update_columns(self, columns):
        """Update columns for parser."""
        self.columns = columns
        self.r_columns = {}
        for k, v in self.columns.items():
            self.r_columns[v] = k

    def _alpaca_sft_to_erine(self, item):
        """Transform alpaca formatted sft data to ernie formatting"""
        src = [
            item.get("prompt", "") + item.get("query", ""),
        ]
        tgt = [
            item.get("response", ""),
        ]
        output = {
            "src": src,
            "tgt": tgt,
        }
        system = item.get("system", None)
        if isinstance(system, str):
            output["system"] = system
        # history
        history = item.get("history", None)
        if isinstance(history, list):
            for each_diag in history[::-1]:
                output["src"].insert(0, each_diag[0])
                output["tgt"].insert(0, each_diag[1])
        return output

    def _alpaca_dpo_to_erine(self, item):
        """Transform alpaca formatted dpo data to ernie formatting"""
        src = [
            item.get("prompt", "") + item.get("query", ""),
        ]
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")
        response = [chosen, rejected]
        output = {
            "src": src,
            "tgt": [],
            "response": response,
            "sort": [1, 0],
        }
        system = item.get("system", None)
        if isinstance(system, str):
            output["system"] = system
        # history
        history = item.get("history", None)
        if isinstance(history, list):
            for each_diag in history[:-1]:
                output["src"].insert(0, each_diag[0])
                output["tgt"].insert(0, each_diag[1])
        return output

    def _alpaca_to_erine(self, item):
        """
        "If train_type is defined as either 'sft' or 'dpo', parse accordingly based on train_type.
        """
        if self.train_type == "dpo":
            return self._alpaca_dpo_to_erine(item)
        return self._alpaca_sft_to_erine(item)

    def _sharegpt_sft_to_erine(self, item):
        """Transform sharegpt formatted sft data to ernie formatting"""
        output = {"src": [], "tgt": []}
        broken_data = False

        odd_tags = ("human", "observation")
        even_tags = ("gpt", "function_call")
        accept_tags = (odd_tags, even_tags)

        for turn_idx, message in enumerate(item["messages"]):
            if "role" in message:
                key_1 = "role"
                key_2 = "content"
            else:
                key_1 = "from"
                key_2 = "value"
            if message[key_1] not in accept_tags[turn_idx % 2]:
                print("Invalid role tag.")
                broken_data = True
                break
            if message[key_1] == "human" or message[key_1] == "observation":
                output["src"].append(message[key_2])
            elif message[key_1] == "gpt" or message[key_1] == "function_call":
                output["tgt"].append(message[key_2])

        if broken_data:
            output = {"src": [], "tgt": []}

        return output

    def _sharegpt_dpo_to_erine(self, item):
        """Transform sharegpt formatted sft data to ernie formatting"""
        output = {
            "src": [],
            "tgt": [],
            "response": "",
            "sort": [1, 0],
        }

        message = item.get("messages", "")
        assert isinstance(message, list)
        output["src"] = message[0]["value"]

        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")
        output["response"] = [chosen["value"], rejected["value"]]

        return output

    def _sharegpt_to_ernie(self, item):
        """
        If train_type is defined as either 'sft' or 'dpo', parse accordingly based on train_type.
        """
        if self.train_type == "dpo":
            return self._sharegpt_dpo_to_erine(item)
        return self._sharegpt_sft_to_erine(item)

    def __iter__(self):
        """Iterator function for dataset."""
        self.run()
        if self.shuffle_file:
            random.shuffle(self.data)
        for item in self.data:
            if self.formatting == "alpaca":
                ex = self._alpaca_to_erine(item)
            else:
                ex = self._sharegpt_to_ernie(item)
            if self.process_fn is not None:
                try:
                    ex = self.process_fn(ex, self.file_name)
                except Exception as e:
                    print(f"Skip parsing error data in {self.file_name}. Error message: {e}")
                    continue
            # ignore invalid example
            if ex is None:
                continue
            yield ex

    def add_dict_row(self, dict_row):
        """
        Mapping the raw dict into the columns.
        For alpaca formatting, if train_type is not defined, then:
            1. Check if the required fields 'chosen' and 'rejected' exist.
            2. If these fields are present, treat it as DPO data; otherwise, treat it as SFT data."
        All items will be treated the same train_type as the first item.
        """
        if self.formatting == "alpaca" and self.train_type == "":
            if dict_row.get("chosen", None) is not None and dict_row.get("rejected", None) is not None:
                self.train_type = "dpo"
                self.update_columns(parse_config.DEFAULT_ALPACA_DPO_COLUMNS_MAPPING)
            else:
                self.train_type = "sft"
                self.update_columns(parse_config.DEFAULT_ALPACA_SFT_COLUMNS_MAPPING)
        elif self.formatting == "sharegpt" and self.train_type == "":
            if dict_row.get("chosen", None) is not None and dict_row.get("rejected", None) is not None:
                self.train_type = "dpo"
                self.update_columns(parse_config.DEFAULT_SHAREGPT_DPO_COLUMNS_MAPPING)
            else:
                self.train_type = "sft"
                self.update_columns(parse_config.DEFAULT_SHAREGPT_SFT_COLUMNS_MAPPING)

        row = {}
        for input_key, output_key in self.r_columns.items():
            value = dict_row.get(input_key, None)
            if value is not None:
                row[output_key] = value
        return row

    def parse_json_file(self):
        """
        Parse the json-format file into data.

        Returns:
            bool (bool): True means success. False means failed.

        Raises:
            errors.DataSetFileCannotOpenError (OSError): Cannot open the file.
            errors.DataSetParseError (json.decoder.JSONDecodeError): Cannot open the file using json parser.
        """
        try:
            with open(self.file_path) as fp:
                json_data = json.load(fp)
                if isinstance(json_data, list):
                    for item in json_data:
                        self.data.append(self.add_dict_row(item))
                elif isinstance(json_data, dict):
                    self.data.append(self.add_dict_row(json_data))
                else:
                    return False
        except Exception as e:
            print("Fail to load file : %s" % str(e))
            return False
        return True

    def parse_json_lines_file(self):
        """
        Parse jsonl format, which every line is a json string.

        Returns:
            bool (bool): True means success. False means failed.

        Raises:
            errors.DataSetFileCannotOpenError (OSError): Cannot open the file.
            errors.DataSetParseError (json.decoder.JSONDecodeError): Cannot open the file using json parser.
        """
        try:
            with open(self.file_path) as fp:
                for line in fp:
                    self.data.append(self.add_dict_row(json.loads(line)))
        except Exception as e:
            print("Fail to load file : %s" % str(e))
            return False
        return True

    def parse(self):
        """
        Parse the dataset files.
        """
        if self.doc_formatting == "json":
            self.parse_json_file()
        elif self.doc_formatting == "jsonl":
            self.parse_json_lines_file()
        elif self.doc_formatting == "auto":
            funcs = {"json": self.parse_json_file, "jsonl": self.parse_json_lines_file}
            for func_name in funcs:
                if self.doc_formatting != "auto":
                    break
                try:
                    func = funcs[func_name]
                    if func():
                        self.doc_formatting = func_name
                except Exception:
                    continue

    def run(self):
        """
        Parse the dataset from file.
        """
        self.parse()


class HFBaseParser(BaseDatasetParser):
    """Hugging Face Base Dataset parser class."""

    def __init__(self, repo_id, config_map, process_fn=None, shuffle_file=False):
        """Init a HFBaseParser from one dataset in data_info.json"""
        self.repo_id = repo_id
        self.download_path = os.path.join(parse_config.DATASET_DOWNLOAD_ROOT, repo_id)
        self.file_name = config_map.get("file_name", "")
        self.file_path = os.path.join(self.download_path, self.file_name)

        self.formatting = config_map.get("formatting", "alpaca")
        self.doc_formatting = config_map.get("doc_formatting", parse_config.DEFAULT_DOC_FORMATTING)
        train_type = config_map.get("train_type", "")
        if self.formatting == "alpaca" and train_type == "sft":
            self.columns = config_map.get("columns", parse_config.DEFAULT_ALPACA_SFT_COLUMNS_MAPPING)
        elif self.formatting == "alpaca" and train_type == "dpo":
            self.columns = config_map.get("columns", parse_config.DEFAULT_ALPACA_DPO_COLUMNS_MAPPING)
        elif self.formatting == "sharegpt" and train_type == "sft":
            self.columns = config_map.get("columns", parse_config.DEFAULT_SHAREGPT_SFT_COLUMNS_MAPPING)
        else:
            self.columns = config_map.get("columns", parse_config.DEFAULT_SHAREGPT_DPO_COLUMNS_MAPPING)
        self.update_columns(self.columns)

        super().__init__(
            self.file_path,
            self.formatting,
            self.doc_formatting,
            self.columns,
            process_fn,
            shuffle_file,
        )

        self.train_type = train_type

    def download(self):
        """
        Download dataset function.
        """
        hf_download_proxy = os.getenv("https_proxy")
        if hf_download_proxy is None:
            hf_download_proxy = os.getenv("HTTPS_PROXY")
        snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            proxies={"http": hf_download_proxy},
            resume_download=True,
            max_workers=8,
            local_dir=self.download_path,
        )

    def run(self):
        """
        Download and parse the dataset.
        """
        self.download()
        self.parse()


def load_data_info():
    """
    Load the data_info.json to get all defined dataset info.
    """
    with open(parse_config.DATA_INFO_FILE) as fp:
        data_info = json.load(fp)
        return data_info


hf_repo_config_map = load_data_info()


def is_hf_dataset(repo_id):
    """
    Check if the data_info configuration of the repo-id.
    """
    global hf_repo_config_map
    return hf_repo_config_map.get(repo_id, None) is not None


def create_hf_dataset(repo_id, process_fn=None, shuffle_file=True):
    """
    Create a hugging-face repo dataset.
    """
    global hf_repo_config_map
    config_map = hf_repo_config_map.get(repo_id, None)
    parser = HFBaseParser(repo_id, config_map, process_fn, shuffle_file)
    return parser


def create_dataset_from_file(
    file_path,
    formatting="alpaca",
    doc_formatting="json",
    process_fn=None,
    shuffle_file=True,
):
    """
    Create dataset from file function.

    Args:
        file_path (str): the file path of dataset.
        formatting (str): formatting of the dataset, e.g. alpaca, sharegpt.
        doc_formatting (str): document formatting of the dataset, e.g. json, jsonl.

    Returns:
        parser (IterableDataset): The iterable dataset object.

    """
    if formatting not in parse_config.DEFAULT_DATASET_COLUMNS_MAPPING:
        msg = (
            f"{formatting} is not supported."
            f"Please use one of [{', '.join(list(parse_config.DEFAULT_DATASET_COLUMNS_MAPPING.keys()))}]"
        )
        raise errors.DataSetFormattingNotSupportedError(f"{msg}")
    columns = parse_config.DEFAULT_DATASET_COLUMNS_MAPPING[formatting]
    parser = BaseDatasetParser(file_path, formatting, doc_formatting, columns, process_fn, shuffle_file)
    return parser
