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
import random

from paddle.io import IterableDataset

from .file_reader import (
    FileListReader,
    FileReader,
    HuggingFaceReader,
    get_hf_dataset_config,
)


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


class MultiSourceDataset(IterableDataset):
    """Dataset that combines multiple data sources with probability sampling."""

    def __init__(self, **dataset_config):
        """Initialize the multi-source dataset.

        Args:
            dataset_config (dict): dataset configurations.
        """

        # arguments process
        task_dataset_path = [
            path for path in str(dataset_config["task_group"]).replace(" ", "").split(",") if path != ""
        ]
        task_dataset_prob = [
            float(prob) for prob in str(dataset_config["task_group_prob"]).replace(" ", "").split(",") if prob != ""
        ]
        sub_dataset_type = [
            type_ for type_ in str(dataset_config["sub_dataset_type"]).replace(" ", "").split(",") if type_ != ""
        ]

        if not (len(task_dataset_path) == len(task_dataset_prob) == len(sub_dataset_type)):
            raise ValueError("The len of dataset path, prob, type are inconsistent, please check the configuration.")

        if len(task_dataset_path) == 0:
            raise ValueError("The len of dataset path is zero, please check the configuration.")

        tasks = []
        for i in range(len(task_dataset_path)):
            tasks.append({"prob": task_dataset_prob[i], "filepath": task_dataset_path[i]})
        # filter zero probability task
        tasks = [task for task in tasks if task["prob"] > 0]
        self._task_group = tasks
        supported_type = ["erniekit", "messages"]
        for idx, task in enumerate(self._task_group):
            each_sub_dataset_type = sub_dataset_type[idx]
            if get_hf_dataset_config(task["filepath"]) is not None:
                task["dataset"] = HuggingFaceReader(
                    file_path=task["filepath"],
                    file_type=each_sub_dataset_type,
                    shuffle_file=dataset_config["random_shuffle"],
                    split_multi_turn=dataset_config.get("split_multi_turn", False),
                    template_backend=dataset_config.get("template_backend", "jinja"),
                )
            if os.path.isdir(task["filepath"]):
                task["dataset"] = FileListReader(
                    file_path=task["filepath"],
                    file_type=each_sub_dataset_type,
                    shuffle_file=dataset_config["random_shuffle"],
                    split_multi_turn=dataset_config.get("split_multi_turn", False),
                    template_backend=dataset_config.get("template_backend", "jinja"),
                )
            elif each_sub_dataset_type in supported_type:
                task["dataset"] = FileReader(
                    file_path=task["filepath"],
                    file_type=each_sub_dataset_type,
                    shuffle_file=dataset_config["random_shuffle"],
                    split_multi_turn=dataset_config.get("split_multi_turn", False),
                    template_backend=dataset_config.get("template_backend", "jinja"),
                )
            else:
                raise NotImplementedError(f"Cannot support {each_sub_dataset_type} now.")
        sum_prob = sum([task["prob"] for task in self._task_group])
        for task in self._task_group:
            task["prob_origin"] = task["prob"]
            task["prob"] = task["prob"] / sum_prob

        self.random_seed = dataset_config["random_seed"]

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
