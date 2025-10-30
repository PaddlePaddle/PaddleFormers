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
""" Basic datasets implement. """

import collections
import glob
import gzip
import json
import random
from contextlib import contextmanager
from functools import partial

import numpy as np
from paddle.io import IterableDataset

from ..utils.log import logger
from .hf import hf_parser


@contextmanager
def open_file(filename):
    """Construct a file handler that can read normal or gzip-compressed files.

    The handler automatically detects compression based on file extension.

    Args:
        filename (str): Path to the target file, which may end with .gz for gzip compression.

    Returns:
        Generator[TextIO]: A file object generator that yields lines from the file.
    """
    if filename.endswith(".gz"):
        fp = gzip.open(filename, "rt")
    else:
        fp = open(filename)
    yield fp
    fp.close()


class FileDataset(IterableDataset):
    """Single file dataset that supports line processing and optional shuffling."""

    def __init__(self, filename, process_fn=None, shuffle_file=False):
        """Initialize the single file dataset.

        Args:
            filename (str): Path to the input data file.
            process_fn (callable, optional): Function to preprocess each line.
            shuffle_file (bool): Whether to shuffle lines before iteration.
        """
        self._filename = filename
        self._process_fn = process_fn
        self._shuffle_file = shuffle_file

    def __iter__(self):
        """Iterate through the dataset with optional shuffling and processing.

        Yields:
            dict: Processed examples from the file, skipping invalid entries.
        """
        with open_file(self._filename) as fin:
            if self._shuffle_file:
                lines = fin.readlines()
                np.random.shuffle(lines)
            else:
                lines = fin
            for lineno, line in enumerate(lines):
                try:
                    ex = json.loads(line)
                except Exception as e:
                    logger.warning(f"Skip loading error data at line {lineno} of {self._filename}. Error message: {e}")
                    continue
                if self._process_fn is not None:
                    try:
                        ex = self._process_fn(ex, self._filename)
                    except Exception as e:
                        logger.warning(
                            f"Skip parsing error data at line {lineno} of {self._filename}. Error message: {e}"
                        )
                        continue
                # ignore invalid example
                if ex is None:
                    continue
                elif isinstance(ex, list) or isinstance(ex, collections.abc.Generator):
                    yield from ex
                else:
                    yield ex


class FileListDataset(IterableDataset):
    """Multiple files dataset supporting file list and glob patterns."""

    def __init__(
        self,
        filename,
        file_format="filelist",
        process_fn=None,
        shuffle_file=False,
        shuffle_files=False,
    ):
        """Initialize the file list dataset.

        Args:
            filename (str): Path to file containing file list or glob pattern.
            file_format (str): 'filelist' for list file or 'glob' for pattern matching.
            process_fn (callable, optional): Function to preprocess each line.
            shuffle_file (bool): Shuffle lines within each file.
            shuffle_files (bool): Shuffle order of files during iteration.
        """
        if file_format == "filelist":
            self._filenames = []
            with open(filename) as fin:
                for line in fin:
                    cols = line.strip().split("\t")
                    self._filenames.append(cols[0])
        elif file_format == "glob":
            self._filenames = sorted(glob.glob(filename))
        else:
            raise ValueError(f"Unsupported file_format: {file_format}")

        self._sub_datasets = []
        for fname in self._filenames:
            self._sub_datasets.append(FileDataset(fname, process_fn=process_fn, shuffle_file=shuffle_file))

        self._shuffle_files = shuffle_files

    def __iter__(self):
        """Iterate through multiple files with optional shuffling.

        Yields:
            dict: Processed examples from all files in specified order.
        """
        if self._shuffle_files:
            # NOTE(hehuang) stateful shuffle
            sub_datasets = self._sub_datasets
            np.random.shuffle(self._sub_datasets)
        else:
            sub_datasets = self._sub_datasets
        for ds in sub_datasets:
            yield from ds


class MultiSourceDataset(IterableDataset):
    """Dataset that combines multiple data sources with probability sampling."""

    def __init__(
        self,
        task_dataset_path,
        task_dataset_prob,
        sub_dataset_type=["erniekit"],
        random_seed=11,
        process_fn=None,
        process_fn_pt=None,
        process_fn_fc=None,
        shuffle_file=False,
        shuffle_files=False,
        is_pretraining=False,
    ):
        """Initialize the multi-source dataset.

        Args:
            task_dataset_path (list): List contains path of data sources.
            task_dataset_prob (list): List contains probabilities of data sources.
            sub_dataset_type (list): List of type of sub-dataset ('erniekit', 'filelist', 'glob', or 'alpaca').
            random_seed (int): Seed for reproducible sampling.
            process_fn (callable, optional): Function to preprocess each example.
            process_fn_pt (callable, optional): Function to preprocess each pretraining example.
            process_fn_fc (callable, optional): Function to preprocess each example for function call.
            shuffle_file (bool): Shuffle lines within each file.
            shuffle_files (bool): Shuffle order of files during iteration.
        """
        tasks = []
        for i in range(len(task_dataset_path)):
            tasks.append({"prob": task_dataset_prob[i], "filepath": task_dataset_path[i]})
        # filter zero probability task
        tasks = [task for task in tasks if task["prob"] > 0]
        self._task_group = tasks
        for idx, task in enumerate(self._task_group):
            each_sub_dataset_type = sub_dataset_type[idx]
            if hf_parser.is_hf_dataset(task["filepath"]):
                task["dataset"] = hf_parser.create_hf_dataset(
                    repo_id=task["filepath"],
                    process_fn=(
                        partial(process_fn, task_name=task["task_name"]) if "task_name" in task else process_fn
                    ),
                    shuffle_file=shuffle_file,
                )
                continue

            if is_pretraining and each_sub_dataset_type == "erniekit":
                task["dataset"] = FileDataset(
                    task["filepath"],
                    process_fn=(
                        partial(process_fn_pt, task_name=task["task_name"]) if "task_name" in task else process_fn_pt
                    ),
                )
                continue

            if each_sub_dataset_type == "erniekit":
                task["dataset"] = FileDataset(
                    task["filepath"],
                    process_fn=(
                        partial(process_fn, task_name=task["task_name"]) if "task_name" in task else process_fn
                    ),
                    shuffle_file=shuffle_file,
                )
            elif each_sub_dataset_type in ["filelist", "glob"]:
                task["dataset"] = FileListDataset(
                    task["train_filelist"],
                    file_format=each_sub_dataset_type,
                    process_fn=(
                        partial(process_fn, task_name=task["task_name"]) if "task_name" in task else process_fn
                    ),
                    shuffle_file=shuffle_file,
                    shuffle_files=shuffle_files,
                )
            elif each_sub_dataset_type in ["alpaca", "sharegpt"]:
                task["dataset"] = hf_parser.create_dataset_from_file(
                    file_path=task["filepath"],
                    formatting=each_sub_dataset_type,
                    doc_formatting="auto",
                    process_fn=(
                        partial(process_fn, task_name=task["task_name"]) if "task_name" in task else process_fn
                    ),
                    shuffle_file=shuffle_file,
                )
            elif each_sub_dataset_type == "chatml":
                # only support for function call dataset
                task["dataset"] = FileDataset(
                    task["filepath"],
                    process_fn=(
                        partial(process_fn_fc, task_name=task["task_name"]) if "task_name" in task else process_fn_fc
                    ),
                    shuffle_file=shuffle_file,
                )
            else:
                raise NotImplementedError(f"Cannot support {each_sub_dataset_type} now.")
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


class InfiniteDataset(IterableDataset):
    """Infinite iterable dataset with shuffle support.

    This dataset supports continuous iteration and optional random shuffling.
    """

    def __init__(self, dataset, rng=None, random_shuffle=True):
        """Initialize InfiniteDataset.

        Args:
            dataset (Iterable): The original dataset to wrap.
            rng (Random, optional): Random number generator for shuffling.
            random_shuffle (bool): Whether to enable random shuffling.
        """
        self.data = list(iter(dataset))
        self.indices = list(range(len(self.data)))
        if rng is None:
            rng = random.Random()
        self.rng = rng
        self.random_shuffle = random_shuffle

    def __iter__(self):
        """Infinite iterator with optional shuffling.

        Yields:
            object: The next data sample from the dataset.
        """
        while True:
            if self.random_shuffle:
                self.rng.shuffle(self.indices)
            for i in self.indices:
                yield self.data[i]
