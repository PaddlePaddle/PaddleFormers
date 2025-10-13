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

import os
import json
import collections
from paddle.io import IterableDataset

from .convertor import erniekit_convertor
from .io import load_csv, load_json, load_jsonl, load_parquet, load_txt
from .download_manager import HuggingFaceDownload

DATA_INFO_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data_info.json")
DATASET_WORKROOT = os.getenv("DATASET_WORKROOT", "/root/.cache/paddleformers")
DATASET_DOWNLOAD_ROOT = os.path.join(DATASET_WORKROOT, "download")


class BaseReader(IterableDataset):
    """ Basic data reader implement. """
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


class FileReader(BaseReader):
    def __init__(self, file_path, file_type, shuffle_file=True):
        super().__init__(file_path=file_path, file_type=file_type, shuffle_file=shuffle_file)

    def __iter__(self):
        ext = self._get_extension()

        print('run here')

        # load file
        if ext not in self.loader_map:
            raise ValueError(f"Unsupported file extension: {ext}")
        res = self.loader_map[ext](self._file_path)

        # data preprocess
        if self._file_type not in self.convertor_map:
            raise ValueError(f"Unsupported file type: {self._file_type}")
        for item in self.convertor_map[self._file_type](res):
            # ignore invalid example
            if item is None:
                continue
            elif isinstance(item, list) or isinstance(item, collections.abc.Generator):
                yield from item
            else:
                yield item

    def _get_extension(self):
        _, ext = os.path.splitext(self._file_path)
        return ext.lower()


class FileListReader(BaseReader):
    def __init__(self, file_path, file_type, shuffle_file=True):
        if not os.path.isdir(file_path):
            raise ValueError(f"Directory not found: {file_path}")
        super().__init__(file_path=file_path, file_type=file_type, shuffle_file=shuffle_file)

    def __iter__(self):
        for file_path in self._get_files():
            reader = FileReader(file_path, self._file_type, self._shuffle_file)
            yield reader

    def _get_files(self):
        files = []
        for filename in os.listdir(self._file_path):
            file_path = os.path.join(self._file_path, filename)
            if os.path.isfile(file_path):
                files.append(file_path)
        return files


class MultiSourceDataset(IterableDataset):
    """Dataset that combines multiple data sources with probability sampling."""

    def __init__(
        self,
        task_dataset_path,
        task_dataset_prob,
        sub_dataset_type=["erniekit"],
        random_seed=11,
        process_fn=None,
        process_fn_fc=None,
        shuffle_file=False,
        shuffle_files=False,
    ):
        """Initialize the multi-source dataset.

        Args:
            task_dataset_path (list): List contains path of data sources.
            task_dataset_prob (list): List contains probabilities of data sources.
            sub_dataset_type (list): List of type of sub-dataset ('erniekit', 'filelist', 'glob', or 'alpaca').
            random_seed (int): Seed for reproducible sampling.
            process_fn (callable, optional): Function to preprocess each example.
            shuffle_file (bool): Shuffle lines within each file.
            shuffle_files (bool): Shuffle order of files during iteration.
        """
        tasks = []
        for i in range(len(task_dataset_path)):
            tasks.append(
                {"prob": task_dataset_prob[i], "filepath": task_dataset_path[i]}
            )
        # filter zero probability task
        tasks = [task for task in tasks if task["prob"] > 0]
        self._task_group = tasks
        supported_type = ["erniekit", "alpaca", "sharegpt", "openai", "query-response"]
        for idx, task in enumerate(self._task_group):
            each_sub_dataset_type = sub_dataset_type[idx]
            if get_hf_dataset_config(task["filepath"]) is not None:
                task["dataset"] = HuggingFaceReader(
                    file_path=task["filepath"],
                    file_type=each_sub_dataset_type,
                    shuffle_file=shuffle_file,
                )
            if os.path.isdir(task["filepath"]):
                task["dataset"] = FileListReader(
                    file_path=task["filepath"],
                    file_type=each_sub_dataset_type,
                    shuffle_file=shuffle_file,
                )
            elif each_sub_dataset_type in supported_type:
                task["dataset"] = FileReader(
                    file_path=task["filepath"],
                    file_type=each_sub_dataset_type,
                    shuffle_file=shuffle_file,
                )
            else:
                raise NotImplementedError(
                    f"Cannot support {each_sub_dataset_type} now."
                )
        sum_prob = sum([task["prob"] for task in self._task_group])
        for task in self._task_group:
            task["prob_origin"] = task["prob"]
            task["prob"] = task["prob"] / sum_prob

        self.random_seed = random_seed

    def __iter__(self):
        """Iterate through examples from multiple sources with probability sampling.

        Yields:
            dict: Processed examples from randomly selected data sources.
        """
        rng = random.Random(self.random_seed)
        probs = [task["prob"] for task in self._task_group]
        # Initialize task iterator
        for task in self._task_group:
            task["iterator"] = iter(task["dataset"])
        while True:
            task = rng.choices(self._task_group, weights=probs)[0]
            try:
                yield from task["iterator"]
            except StopIteration:
                task["iterator"] = iter(task["dataset"])
                yield from task["iterator"]


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
