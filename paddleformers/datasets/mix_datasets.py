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

import random
from abc import abstractmethod

import numpy as np
from paddle.io import IterableDataset, get_worker_info

from .base import InfiniteDataset


class BaseMixDataset(IterableDataset):
    """
    A dataset randomly samples from multiple datasets with specified probabilities.
    """

    def __init__(
        self,
        datasets_list,
        datasets_prob,
        mode: str = "upsampling",
        seed=42,
        random_shuffle=True,
        num_samples_each_epoch=6000000,
        reverse=False,
    ):
        """
        Initialize the RandomDataset.

        Args:
            datasets_list: List of datasets to sample from
            datasets_prob: List of probabilities corresponding to each dataset
            rng: Random number generator (default creates new one if None)
            random_shuffle: Whether to shuffle samples within each dataset (default True)
            num_samples_each_epoch: Total number of samples to generate per epoch
        """
        self.datasets_list = datasets_list
        self.datasets_prob = datasets_prob
        self.mode = mode
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(self.seed)
        self.epoch_index = 0
        self.epoch_np_rng = np.random.RandomState(self.epoch_index)
        self.random_shuffle = random_shuffle
        self.num_samples_each_epoch = num_samples_each_epoch
        self.reverse = reverse

    @abstractmethod
    def __iter__(self):
        """
        Returns an iterator that can loop over the dataset indefinitely.
        """
        pass

    @abstractmethod
    def __len__(self):
        """Returns the total size of the dataset."""
        pass


class RandomDataset(BaseMixDataset):
    """
    A dataset randomly samples from multiple datasets with specified probabilities.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the RandomDataset.

        Args:
            datasets_list: List of datasets to sample from
            datasets_prob: List of probabilities corresponding to each dataset
            rng: Random number generator (default creates new one if None)
            random_shuffle: Whether to shuffle samples within each dataset (default True)
            num_samples_each_epoch: Total number of samples to generate per epoch
        """
        super().__init__(*args, **kwargs)

        self.tasks = [
            {"iterator": iter(InfiniteDataset(dataset, self.rng, self.random_shuffle))}
            for dataset in self.datasets_list
        ]

    def __iter__(self):
        """
        Define the iterator behavior for the dataset.
        This will be called when iterating over the dataset.
        """
        worker_info = get_worker_info()

        while True:
            examples_all = []
            target_nums = [int(prob * self.num_samples_each_epoch) for prob in self.datasets_prob]

            for i, task in enumerate(self.tasks):
                examples = [next(task["iterator"]) for _ in range(target_nums[i])]
                if self.random_shuffle:
                    self.epoch_np_rng.shuffle(examples)
                if worker_info is not None:
                    examples = examples[worker_info.id :: worker_info.num_workers]
                examples_all.extend(examples)

            if self.random_shuffle:
                self.epoch_np_rng.shuffle(examples_all)

            if self.reverse:
                examples_all = examples_all[::-1]

            for example in examples_all:
                yield example

            self.epoch_index += 1
            self.epoch_np_rng = np.random.RandomState(self.epoch_index)

    def __len__(self):
        return self.num_samples_each_epoch


class ConcatDataset(BaseMixDataset):
    """
    A dataset that concatenates multiple datasets into a single one.

    This class loads all items from the provided datasets into a single list in memory.
    It can then be iterated over indefinitely, with an option to shuffle the data
    at the beginning of each pass (epoch).
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the ConcatDataset.

        Args:
            datasets_list (list): A list of dataset objects to concatenate.
            datasets_prob (any): A parameter for dataset probabilities.
            rng (random.Random, optional): A random number generator for shuffling. Defaults to a new instance.
            random_shuffle (bool, optional): If True, the dataset will be shuffled at the start of each epoch. Defaults to True.
        """
        super().__init__(*args, **kwargs)

        self.data = []
        for dataset in self.datasets_list:
            self.data.extend(list(iter(dataset)))
        self.indices = list(range(len(self.data)))

    def __iter__(self):
        """
        Returns an iterator that can loop over the dataset indefinitely.
        """
        while True:
            if self.random_shuffle:
                self.epoch_np_rng.shuffle(self.indices)

            for i in self.indices:
                yield self.data[i]

            self.epoch_index += 1
            self.epoch_np_rng = np.random.RandomState(self.epoch_index)

    def __len__(self):
        """Returns the total size of the dataset."""
        return len(self.data)


class InterLeaveDataset(BaseMixDataset):
    """
    Creates a new dataset by interleaving multiple source datasets according to specified probabilities.

    This class supports two sampling strategies:
    - 'upsampling' (first_exhausted): Stops as soon as any dataset is fully exhausted
    - 'oversampling' (all_exhausted): Stops only when all datasets have been fully exhausted at least once
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the InterLeaveDataset and builds the complete dataset.
        """
        super().__init__(*args, **kwargs)

        if self.mode not in ["upsampling", "oversampling"]:
            raise ValueError(f"Unknown mode '{self.mode}'. Mode must be 'upsampling' or 'oversampling'.")
        self.datasets_prob = np.array(self.datasets_prob)

        self.datasets_data = [list(iter(ds)) for ds in self.datasets_list]
        self.lengths = np.array([len(ds_list) for ds_list in self.datasets_data])

        # construct interleave dataset
        self.data = []
        self._build_dataset()

        self.indices = list(range(len(self.data)))

    def _build_dataset(self):
        """
        Builds the final dataset using the interleaving sampling strategy.
        """
        is_exhausted = np.full(len(self.lengths), False)

        oversampling = self.mode == "oversampling"
        bool_strategy_func = np.all if oversampling else np.any

        print(f"Building dataset in {self.mode} mode...")
        print(f"Dataset lengths: {self.lengths.tolist()}")
        print(f"Probabilities: {self.datasets_prob.tolist()}")

        def iter_random_indices():
            """Get an infinite iterator that randomly samples the index of the source to pick examples from."""
            while True:
                yield from (
                    int(i) for i in self.np_rng.choice(len(self.datasets_data), size=1000, p=self.datasets_prob)
                )

        current_index = [0] * len(self.datasets_data)
        samples_taken = [0] * len(self.datasets_data)

        for source_idx in iter_random_indices():
            if bool_strategy_func(is_exhausted):
                break

            current_dataset = self.datasets_data[source_idx]
            sample = current_dataset[current_index[source_idx]]
            self.data.append(sample)

            current_index[source_idx] += 1
            samples_taken[source_idx] += 1

            if current_index[source_idx] >= self.lengths[source_idx]:
                is_exhausted[source_idx] = True
                current_index[source_idx] = 0

        print(f"Dataset construction complete: {len(self.data)} total samples")

        for i, (taken, original_size) in enumerate(zip(samples_taken, self.lengths)):
            actual_prob = taken / len(self.data) if len(self.data) > 0 else 0
            resampling_ratio = taken / original_size if original_size > 0 else 0
            print(f"Dataset {i}: {taken} samples taken from {original_size} available")
            print(f"  Target prob: {self.datasets_prob[i]:.3f}, Actual prob: {actual_prob:.3f}")
            print(f"  Resampling ratio: {resampling_ratio:.2f}x")

            if resampling_ratio >= 1.0:
                print(f"All {original_size} original samples were used at least once")
            else:
                unused = original_size - taken
                print(f"{unused} samples were not used from this dataset")

    def __iter__(self):
        """
        Returns an iterator over the pre-built dataset.
        """
        while True:
            if self.random_shuffle:
                self.epoch_np_rng.shuffle(self.indices)

            for i in self.indices:
                yield self.data[i]

            self.epoch_index += 1
            self.epoch_np_rng = np.random.RandomState(self.epoch_index)

    def __len__(self):
        """Returns the exact size of the pre-built dataset."""
        return len(self.data)


CLASS_MAPPING = {
    "concat": ConcatDataset,
    "interleave_under": InterLeaveDataset,
    "interleave_over": InterLeaveDataset,
    "random": RandomDataset,
}


def create_dataset_instance(class_name, *args, **kwargs):
    target_class = CLASS_MAPPING.get(class_name)

    if target_class:
        return target_class(*args, **kwargs)
    else:
        print(f"Error: cannot find class named '{class_name}'.")
        return None
