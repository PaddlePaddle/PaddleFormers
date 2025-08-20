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
import glob
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
        self.failed_row = 0

        self.process_fn = process_fn
        self.shuffle_file = shuffle_file

        self.output_file_name = self.file_name + ".ernie.json"
        self.output_file_path = os.path.join(parse_config.DATASET_OUTPUT_ROOT, self.output_file_name)
        self.output_json_indent = parse_config.DEFAULT_OUTPUT_JSON_INDENT

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
        return output

    def _alpaca_to_erine(self, item):
        """
        "If train_type is defined as either 'sft' or 'dpo', parse accordingly based on train_type.
        """
        if self.train_type == "dpo":
            return self._alpaca_dpo_to_erine(item)
        return self._alpaca_sft_to_erine(item)

    def __iter__(self):
        """Iterator function for dataset."""
        self.run()
        if self.shuffle_file:
            random.shuffle(self.data)
        for item in self.data:
            ex = self._alpaca_to_erine(item)
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

    def scan_dataset_file(self):
        """
        Scan files under dataset folder and return the first one filename.
        """
        files = glob.glob(os.path.join(self.download_path, "*"))
        filenames_under_workspace = sorted([filepath.split(os.sep)[-1] for filepath in files])
        filenames = []
        for filename in filenames_under_workspace:
            if filename.lower() == "readme.md":
                continue
            filenames.append(filename)
        if len(filenames) == 0:
            msg = f"{self.repo_id} cannot find dataset files after scan, please check or define in dataset_config.py"
            raise errors.DataSetFileNotFoundError(msg)
        elif len(filenames) > 1:
            msg = (
                f"{self.repo_id} cannot find more than one file in dataset files after scan. "
                f"please check and define in dataset_config.py. Scanned files: {filenames}"
            )
        return filenames[0]

    def check_and_fill_row_alpaca(self, row):
        """
        For alpaca formatting, check and fill default value for the essential field like:
            'SFT': prompt/instruction or query/input, response/output.
            'DPO': prompt/instruction or query/input, chosen, rejected
        """
        has_data = False
        for key in row:
            if row[key] is not None and len(row[key]) > 0:
                has_data = True

        value_mapping = parse_config.DEFAULT_COLUMN_VALUE_MAPPING
        if self.train_type == "dpo":
            value_mapping = parse_config.DEFAULT_ALPACA_DPO_COLUMNS_VALUE_MAPPING
        for key in value_mapping:
            if key not in row:
                row[key] = ""
        return has_data

    def check_row(self, row):
        """
        Check if the data meets the format requirements.
        """
        if self.formatting == "alpaca":
            return self.check_and_fill_row_alpaca(row)
        return True

    def append_data(self, row):
        """
        Append the correct row into data.
        """
        if not isinstance(row, dict):
            return
        if self.check_row(row):
            self.data.append(row)
        else:
            self.failed_row += 1

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
                self.update_columns(parse_config.DEFAULT_ALPACA_COLUMNS_MAPPING)

        default_values_mapping = parse_config.DEFAULT_COLUMN_VALUE_MAPPING
        if self.formatting == "alpaca" and self.train_type == "dpo":
            default_values_mapping = parse_config.DEFAULT_ALPACA_DPO_COLUMNS_VALUE_MAPPING

        row = {}
        for input_key, output_key in self.r_columns.items():
            value = dict_row.get(input_key, None)
            if value is not None:
                row[output_key] = value
                continue
            default_value = default_values_mapping.get(output_key, None)
            if default_value is not None:
                row[output_key] = default_value
        return row

    def add_str_row(self, str_row):
        """
        Mapping the raw json string into the columns.
        """
        line = str_row.strip()
        if len(line) == 0:
            return None
        try:
            input = json.loads(str_row)
            return self.add_dict_row(input)
        except json.decoder.JSONDecodeError:
            msg = f"Unformatted json-line: {str_row}, stop"
            raise errors.DataSetParseError(msg)
        except Exception as e:
            print("line error:%s" % str(e))

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
                        self.append_data(self.add_dict_row(item))
                elif isinstance(json_data, dict):
                    self.append_data(self.add_dict_row(json_data))
                else:
                    return False
        except OSError:
            msg = f"Cannot open dataset file: {self.file_path}"
            raise errors.DataSetFileCannotOpenError(msg)
        except json.decoder.JSONDecodeError:
            msg = f"Unformatted json file: {self.file_path}, stop"
            raise errors.DataSetParseError(msg)
        except Exception as e:
            print("Fail to load file:%s" % str(e))
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
        line = ""
        try:
            with open(self.file_path) as fp:
                for line in fp:
                    self.append_data(self.add_str_row(line))
        except OSError:
            msg = f"Cannot open dataset file: {self.file_path}"
            raise errors.DataSetFileCannotOpenError(msg)
        except json.decoder.JSONDecodeError as ee:
            print(f"bad line:{line}, {ee}")
            msg = f"Unformatted json file: {self.file_path}, stop"
            raise errors.DataSetParseError(msg)
        return True

    def output_json(self):
        """
        Output data into file which is json format.
        """
        if not parse_config.DEBUG_DATASET_OUTPUT_FORMATTED_FILE:
            return
        if not os.path.exists(parse_config.DATASET_OUTPUT_ROOT):
            os.makedirs(parse_config.DATASET_OUTPUT_ROOT)
        with open(self.output_file_path, "w") as ofp:
            ofp.write(json.dumps(self.data, ensure_ascii=False, indent=2))
        print(f"[DEBUG]Output parsed result as ernie-formatted json at {self.output_file_path}")

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
        print(
            f"{self.file_name} read {len(self.data)} items successfully and "
            f"{self.failed_row} failed from {self.file_path}, doc formatting:{self.doc_formatting}"
        )

    def check_dataset_filename(self):
        """
        Check if file exists.
        """
        if self.file_path == "":
            msg = "file_path should not be empty"
            raise errors.DataSetFileNotFoundError(msg)
        if not os.path.isfile(os.path.join(self.file_path)):
            print(f"Checking file_path:{self.file_path}")
            msg = f"cannot find dataset file:{self.file_path}"
            raise errors.DataSetFileNotFoundError(msg)

    def run(self):
        """
        Parse the dataset from file.
        """
        self.check_dataset_filename()
        self.parse()
        # self.output_json()


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
        self.columns = config_map.get("columns", parse_config.DEFAULT_ALPACA_COLUMNS_MAPPING)
        train_type = config_map.get("train_type", "")
        if train_type == "dpo":
            self.columns = config_map.get("columns", parse_config.DEFAULT_ALPACA_DPO_COLUMNS_MAPPING)
        super().__init__(
            self.file_path,
            self.formatting,
            self.doc_formatting,
            self.columns,
            process_fn,
            shuffle_file,
        )
        self.train_type = train_type
        self.update_columns(self.columns)

        self.output_file_name = repo_id.replace("/", ".") + ".json"
        self.output_file_path = os.path.join(parse_config.DATASET_OUTPUT_ROOT, self.output_file_name)
        self.output_json_indent = parse_config.DEFAULT_OUTPUT_JSON_INDENT

    def _base_download(self):
        """
        Download dataset from hugging-face.
        """
        snapshot_download(repo_id=self.repo_id, repo_type="dataset", local_dir=self.download_path)

    def download(self):
        """
        Download dataset function.
        """
        self._base_download()

    def check_dataset_filename(self):
        """
        Check if file exists.
        """
        if self.file_name == "":
            msg = f"file_name should be defined for {self.repo_id} in data_info.json"
            raise errors.DataSetFileNotFoundError(msg)
        if not os.path.isfile(os.path.join(self.file_path)):
            print(f"Checking file_path:{self.file_path}")
            msg = (
                f"{self.repo_id} cannot find dataset file:{self.file_name} "
                f'under path: "{self.download_path}". Please check data_info.json.'
            )
            raise errors.DataSetFileNotFoundError(msg)

    def run(self):
        """
        Download and parse the dataset.
        """
        self.download()
        self.check_dataset_filename()
        self.parse()
        # self.output_json()


class HFScanParser(HFBaseParser):
    """Dataset parser which scan the dataset files without definition in data_info.json"""

    def __init__(self, repo_id, process_fn=None, shuffle_file=False):
        """
        Init a HFScanParser to parse the dataset which is not defined in data_info.json.

        Args:
            repo_id (str): repo id for hugging-face hub.
        """
        super().__init__(repo_id, {}, process_fn, shuffle_file)

    def parse(self):
        """
        Parse the dataset file which is not defined in data_info.json.
        Firstly try to parse as a json file, then jsonl file.

        Raises:
            errors.DataSetFileCannotOpenError (OSError): Cannot open the file.
            errors.DataSetParseError (json.decoder.JSONDecodeError): Cannot open the file using json parser.
        """
        try:
            self.parse_json_file()
        except errors.DataSetParseError:
            self.parse_json_lines_file()
        except errors.DataSetFileCannotOpenError as e:
            raise e
        print(
            f"{self.repo_id} read {len(self.data)} items successfully and "
            f"{self.failed_row} failed from {self.file_path}"
        )

    def run(self):
        """
        Download and parse the dataset. Some parameters are defined after dataset has been downloaded.
        """
        self.download()
        self.file_name = self.scan_dataset_file()
        print(f'Find {self.file_name} under {self.download_path} when scanning "*.json".')
        self.file_path = os.path.join(self.download_path, self.file_name)
        self.check_dataset_filename()
        self.parse()
        self.output_json()


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
    if config_map:
        parser = HFBaseParser(repo_id, config_map, process_fn, shuffle_file)
    else:
        parser = HFScanParser(repo_id, process_fn, shuffle_file)
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
