# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import atexit
import inspect
import os
import time
import warnings
from collections import namedtuple
from itertools import islice

# Add this for extremely slow connection to hf sever even for local dataset.
os.environ["HF_UPDATE_DOWNLOAD_COUNTS"] = "False"

import datasets
from multiprocess import Pool, RLock

import paddleformers

try:
    import paddle.distributed as dist
except Exception:
    warnings.warn("paddle.distributed is not contains in you paddle!")

import importlib
from functools import partial

from paddle.io import Dataset, IterableDataset
from paddle.utils.download import _get_unique_endpoints

from ..utils.env import DATA_HOME

__all__ = ["MapDataset", "DatasetBuilder", "IterDataset", "load_dataset"]

DATASETS_MODULE_PATH = "paddleformers.datasets."

# Patch for intranet
from datasets import load_dataset as origin_load_dataset  # noqa: E402


def load_from_ppnlp(path, *args, **kwargs):
    ppnlp_path = paddleformers.datasets.__path__[0]
    new_path = os.path.split(path)[-1]
    new_path = os.path.join(ppnlp_path, "hf_datasets", new_path + ".py")
    if os.path.exists(new_path):
        return origin_load_dataset(new_path, trust_remote_code=True, *args, **kwargs)
    else:
        return origin_load_dataset(path, trust_remote_code=True, *args, **kwargs)


datasets.load_dataset = load_from_ppnlp


class DatasetTuple:
    def __init__(self, splits):
        self.identifier_map, identifiers = self._gen_identifier_map(splits)
        self.tuple_cls = namedtuple("datasets", identifiers)
        self.tuple = self.tuple_cls(*[None for _ in splits])

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return self.tuple[key]
        if isinstance(key, str):
            return getattr(self.tuple, self.identifier_map[key])

    def __setitem__(self, key, value):
        self.tuple = self.tuple._replace(**{self.identifier_map[key]: value})

    def _gen_identifier_map(self, splits):
        identifier_map = {}
        identifiers = []
        for i in range(len(splits)):
            identifiers.append("splits_" + str(i))
            identifier_map[splits[i]] = "splits_" + str(i)
        return identifier_map, identifiers

    def __len__(self):
        return len(self.tuple)


def import_main_class(module_path):
    """
    Import a module at module_path and return its DatasetBuilder class.

    """
    module_path = DATASETS_MODULE_PATH + module_path
    module = importlib.import_module(module_path)
    main_cls_type = DatasetBuilder

    # Find the main class in our imported module
    module_main_cls = None
    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, main_cls_type):
            if name == "DatasetBuilder":
                continue
            module_main_cls = obj
            break

    return module_main_cls


def load_from_hf(path, name=None, splits=None, **kwargs):
    from datasets import DatasetDict, IterableDatasetDict
    from datasets import load_dataset as load_hf_dataset
    from datasets.features import ClassLabel

    try:
        if "split" in kwargs:
            hf_datasets = load_hf_dataset(path, name=name, **kwargs)
        else:
            hf_datasets = load_hf_dataset(path, name=name, split=splits, **kwargs)

    except FileNotFoundError:
        raise FileNotFoundError("Couldn't find the dataset script for '" + path + "' on PaddleFormers or HuggingFace")
    else:
        label_list = []
        if isinstance(hf_datasets, DatasetDict):
            datasets = DatasetTuple(list(hf_datasets.keys()))
            for split, ds in hf_datasets.items():
                for feature in ds.features.values():
                    if isinstance(feature, ClassLabel):
                        label_list = feature.names
                datasets[split] = MapDataset(ds, label_list=label_list)
        elif isinstance(hf_datasets, IterableDatasetDict):
            datasets = DatasetTuple(list(hf_datasets.keys()))
            for split, ds in hf_datasets.items():
                datasets[split] = IterDataset(ds)
        elif isinstance(hf_datasets, list):
            datasets = DatasetTuple(splits)
            for i, split in enumerate(splits):
                for feature in hf_datasets[i].features.values():
                    if isinstance(feature, ClassLabel):
                        label_list = feature.names
                datasets[split] = MapDataset(hf_datasets[i], label_list=label_list)
        else:
            for feature in hf_datasets.features.values():
                if isinstance(feature, ClassLabel):
                    label_list = feature.names
            datasets = MapDataset(hf_datasets, label_list=label_list)
    return datasets


def load_dataset(path_or_read_func, name=None, data_files=None, splits=None, lazy=None, **kwargs):
    """
    This method will load a dataset, either form PaddleFormers library or from a
    self-defined data loading script, by calling functions in `DatasetBuilder`.


    Either `splits` or `data_files` must be specified.

    Args:
        path_or_read_func (str|callable): Name of the dataset processing script
            in custom data reading function.
        name (str, optional): Additional name to select a more specific dataset.
            Defaults to None.
        data_files (str|list|tuple|dict, optional): Defining the path of dataset
            files. If None. `splits` must be specified. Defaults to None.
        splits (str|list|tuple, optional): Which split of the data to load. If None.
            `data_files` must be specified. Defaults to None.
        lazy (bool, optional): Weather to return `MapDataset` or an `IterDataset`.
            True for `IterDataset`. False for `MapDataset`. If None, return the
            default type of this dataset. Defaults to None.
        kwargs (dict): Other keyword arguments to be passed to the `DatasetBuilder`.

    Returns:
        A `MapDataset` or `IterDataset` or a tuple of those.

    """
    if inspect.isfunction(path_or_read_func):
        assert lazy is not None, "lazy can not be None in custom mode."
        kwargs["name"] = name
        kwargs["data_files"] = data_files
        kwargs["splits"] = splits
        custom_kwargs = {}
        for name in inspect.signature(path_or_read_func).parameters.keys():
            if name in kwargs.keys():
                custom_kwargs[name] = kwargs[name]

        reader_instance = SimpleBuilder(lazy=lazy, read_func=path_or_read_func)
        return reader_instance.read(**custom_kwargs)
    else:
        try:
            reader_cls = import_main_class(path_or_read_func)
        except ModuleNotFoundError:
            datasets = load_from_hf(
                path_or_read_func, name=name, splits=splits, data_files=data_files, streaming=lazy, **kwargs
            )
        else:
            reader_instance = reader_cls(lazy=lazy, name=name, **kwargs)

            # Check if selected name and split is valid in this DatasetBuilder
            if hasattr(reader_instance, "BUILDER_CONFIGS"):
                if name in reader_cls.BUILDER_CONFIGS.keys():
                    split_names = reader_cls.BUILDER_CONFIGS[name]["splits"].keys()
                else:
                    raise ValueError(
                        'Invalid name "{}". Should be one of {}.'.format(name, list(reader_cls.BUILDER_CONFIGS.keys()))
                    )
            elif hasattr(reader_instance, "SPLITS"):
                split_names = reader_instance.SPLITS.keys()
            else:
                raise AttributeError("Either 'SPLITS' or 'BUILDER_CONFIGS' must be implemented for DatasetBuilder.")

            selected_splits = []
            if isinstance(splits, list) or isinstance(splits, tuple):
                selected_splits.extend(splits)
            else:
                selected_splits += [splits]

            for split_name in selected_splits:
                if split_name not in split_names and split_name is not None:
                    raise ValueError('Invalid split "{}". Should be one of {}.'.format(split_name, list(split_names)))

            datasets = reader_instance.read_datasets(data_files=data_files, splits=splits)
        return datasets


class MapDataset(Dataset):
    """
    Wraps a map-style dataset-like object as an instance of `MapDataset`, and equips it
    with `map` and other utility methods. All non-magic methods of the raw object
    are also accessible.

    Args:
        data (list|Dataset): An object with `__getitem__` and `__len__` methods. It could
            be a list or a subclass of `paddle.io.Dataset`.
        kwargs (dict, optional): Other information to be passed to the dataset.

    """

    def __init__(self, data, **kwargs):
        self.data = data
        self._transform_pipline = []
        self.new_data = self.data
        self.info = kwargs
        self.label_list = self.info.pop("label_list", None)
        self.vocab_info = self.info.pop("vocab_info", None)

    def _transform(self, data):
        for fn in self._transform_pipline:
            data = fn(data)
        return data

    def __getitem__(self, idx):
        """
        Basic function of `MapDataset` to get sample from dataset with a given
        index.
        """
        return self._transform(self.new_data[idx]) if self._transform_pipline else self.new_data[idx]

    def __len__(self):
        """
        Returns the number of samples in dataset.
        """
        return len(self.new_data)

    def filter(self, fn, num_workers=0):
        """
        Filters samples by the filter function and uses the filtered data to
        update this dataset.

        Args:
            fn (callable): A filter function that takes a sample as input and
                returns a boolean. Samples that return False would be discarded.
            num_workers(int, optional): Number of processes for multiprocessing. If
                set to 0, it doesn't use multiprocessing. Defaults to `0`.
        """
        assert num_workers >= 0, "num_workers should be a non-negative value"
        if num_workers > 1:
            shards = [
                self._shard(num_shards=num_workers, index=index, contiguous=True) for index in range(num_workers)
            ]
            kwds_per_shard = [dict(self=shards[rank], fn=fn) for rank in range(num_workers)]
            pool = Pool(num_workers, initargs=(RLock(),))

            results = [pool.apply_async(self.__class__._filter, kwds=kwds) for kwds in kwds_per_shard]
            transformed_shards = [r.get() for r in results]

            pool.close()
            pool.join()
            self.new_data = []
            for i in range(num_workers):
                self.new_data += transformed_shards[i].new_data
            return self
        else:
            return self._filter(fn)

    def _filter(self, fn):
        self.new_data = [self.new_data[idx] for idx in range(len(self.new_data)) if fn(self.new_data[idx])]
        return self

    def shard(self, num_shards=None, index=None, contiguous=False):
        self.new_data = self._shard(num_shards=num_shards, index=index, contiguous=contiguous).data
        return self

    def _shard(self, num_shards=None, index=None, contiguous=False):
        """
        Split the dataset into `num_shards` pieces. Note that the size of each
        shard might be different because the original dataset may not be evenly
        divisible.

        Args:
            num_shards (int, optional): An integer representing the number of
                data shards. If None, `num_shards` would be number of trainers.
                Defaults to `None`.
            index (int, optional): An integer representing the index of the
                current shard. If None, `index` would be the current trainer rank
                id. Defaults to `None`.
            contiguous: (bool, optional): If true, contiguous chunks of data
                will be select for sharding. And total number of examples will
                be the same. Otherwise each shard will contain all examples of
                dataset whose index mod `num_shards` = `index`. Defaults to `False`.
        """
        if num_shards is None:
            num_shards = dist.get_world_size()
        if index is None:
            index = dist.get_rank()

        if contiguous:
            div = len(self) // num_shards
            mod = len(self) % num_shards
            start = div * index + min(index, mod)
            end = start + div + (1 if index < mod else 0)
            new_data = [self.new_data[idx] for idx in range(start, end)]
        else:
            new_data = [self.new_data[idx] for idx in range(len(self.new_data)) if idx % num_shards == index]

        return MapDataset(new_data)

    def map(self, fn, lazy=True, batched=False, num_workers=0):
        """
        Performs specific function on the dataset to transform and update every sample.

        Args:
            fn (callable): Transformations to be performed. It receives single
                sample as argument if batched is False. Else it receives all examples.
            lazy (bool, optional): If True, transformations would be delayed and
                performed on demand. Otherwise, transforms all samples at once. Note that
                if `fn` is stochastic, `lazy` should be True or you will get the same
                result on all epochs. Defaults to False.
            batched(bool, optional): If True, transformations would take all examples as
                input and return a collection of transformed examples. Note that if set
                True, `lazy` option would be ignored. Defaults to False.
            num_workers(int, optional): Number of processes for multiprocessing. If
                set to 0, it doesn't use multiprocessing. Note that if set to positive
                value, `lazy` option would be ignored. Defaults to 0.
        """

        assert num_workers >= 0, "num_workers should be a non-negative value"
        if num_workers > 1:
            shards = [
                self._shard(num_shards=num_workers, index=index, contiguous=True) for index in range(num_workers)
            ]
            kwds_per_shard = [
                dict(self=shards[rank], fn=fn, lazy=False, batched=batched) for rank in range(num_workers)
            ]
            pool = Pool(num_workers, initargs=(RLock(),))
            results = [pool.apply_async(self.__class__._map, kwds=kwds) for kwds in kwds_per_shard]
            transformed_shards = [r.get() for r in results]
            pool.close()
            pool.join()
            self.new_data = []
            for i in range(num_workers):
                self.new_data += transformed_shards[i].new_data
            return self
        else:
            return self._map(fn, lazy=lazy, batched=batched)

    def _map(self, fn, lazy=True, batched=False):
        if batched:
            self.new_data = fn(self.new_data)
        elif lazy:
            self._transform_pipline.append(fn)
        else:
            self.new_data = [fn(self.new_data[idx]) for idx in range(len(self.new_data))]
        return self


class IterDataset(IterableDataset):
    """
    Wraps a dataset-like object as an instance of `IterDataset`, and equips it with
    `map` and other utility methods. All non-magic methods of the raw object
    also accessible.

    Args:
        data (Iterable): An object with `__iter__` function. It can be a Iterable or a
            subclass of `paddle.io.IterableDataset`.
        kwargs (dict, optional): Other information to be passed to the dataset.

    """

    def __init__(self, data, **kwargs):
        self.data = data
        self._transform_pipline = []
        self._filter_pipline = []

        self.label_list = kwargs.pop("label_list", None)
        self.vocab_info = kwargs.pop("vocab_info", None)

    def _transform(self, data):
        for fn in self._transform_pipline:
            data = fn(data)
        return data

    def _shard_filter(self, num_samples):
        return True

    def _filter(self, data):
        for fn in self._filter_pipline:
            if not fn(data):
                return False
        return True

    def __iter__(self):
        """
        yields sample sequentially.
        """
        num_samples = 0
        if inspect.isfunction(self.data):
            for example in self.data():
                if (not self._filter_pipline or self._filter(self._filter_pipline)) and self._shard_filter(
                    num_samples=num_samples
                ):
                    yield self._transform(example) if self._transform_pipline else example
                num_samples += 1
        else:
            if inspect.isgenerator(self.data):
                warnings.warn("Receiving generator as data source, data can only be iterated once")
            for example in self.data:
                if (not self._filter_pipline or self._filter(self._filter_pipline)) and self._shard_filter(
                    num_samples=num_samples
                ):
                    yield self._transform(example) if self._transform_pipline else example
                num_samples += 1

    def skip(self, n):
        if inspect.isfunction(self.data):
            raise NotImplementedError("Function-based IterDataset does not support `.skip()`")
        self.data = islice(self.data, n, None)
        return self

    def filter(self, fn):
        """
        Filters samples by the filter function and uses the filtered data to
        update this dataset.

        Args:
            fn (callable): A filter function that takes a sample as input and
                returns a boolean. Samples that return False are discarded.
        """

        self._filter_pipline.append(fn)

        return self

    def shard(self, num_shards=None, index=None):
        """
        Split the dataset into `num_shards` pieces.

        Args:
            num_shards (int, optional): An integer representing the number of
                data shards. If None, `num_shards` would be number of trainers.
                Defaults to None.
            index (int, optional): An integer representing the index of the
                current shard. If None, `index` would be the current trainer rank
                id. Defaults to None.
        """
        if num_shards is None:
            num_shards = dist.get_world_size()
        if index is None:
            index = dist.get_rank()

        def sharder(num_shards, index, num_samples):
            if num_samples % num_shards == index:
                return True
            else:
                return False

        fn = partial(sharder, num_shards=num_shards, index=index)
        self._shard_filter = fn
        return self

    def map(self, fn):
        """
        Performs specific function on the dataset to transform and update every sample.

        Args:
            fn (callable): Transformations to be performed. It receives single
                sample as argument.
        """

        self._transform_pipline.append(fn)

        return self


class DatasetBuilder:
    """
    A base class for all DatasetBuilder. It provides a `read()` function to turn
    a data file into a MapDataset or IterDataset.

    `_get_data()` function and `_read()` function should be implemented to download
    data file and read data file into a `Iterable` of the examples.
    """

    lazy = False

    def __init__(self, lazy=None, name=None, **config):
        if lazy is not None:
            self.lazy = lazy
        self.name = name
        self.config = config

    def read_datasets(self, splits=None, data_files=None):
        def remove_if_exit(filepath):
            if isinstance(filepath, (list, tuple)):
                for file in filepath:
                    try:
                        os.remove(file)
                    except OSError:
                        pass
            else:
                try:
                    os.remove(filepath)
                except OSError:
                    pass

        if data_files is None:
            if splits is None:
                splits = (
                    list(self.BUILDER_CONFIGS[self.name]["splits"].keys())
                    if hasattr(self, "BUILDER_CONFIGS")
                    else list(self.SPLITS.keys())
                )

            assert (
                isinstance(splits, str)
                or (isinstance(splits, list) and isinstance(splits[0], str))
                or (isinstance(splits, tuple) and isinstance(splits[0], str))
            ), "`splits` should be a string or list of string or a tuple of string."

            if isinstance(splits, str):
                splits = [splits]
            datasets = DatasetTuple(splits)
            parallel_env = dist.ParallelEnv()
            unique_endpoints = _get_unique_endpoints(parallel_env.trainer_endpoints[:])
            # move register hook to first and register together
            lock_files = []
            for split in splits:
                lock_file = os.path.join(DATA_HOME, self.__class__.__name__)
                if self.name is not None:
                    lock_file = lock_file + "." + self.name
                lock_file += "." + split + ".done" + "." + str(os.getppid())
                lock_files.append(lock_file)
            # Must register to all procs to make the lock file can be removed
            # when any proc breaks. Otherwise, the single registered proc may
            # not receive proper signal send by the parent proc to exit.
            atexit.register(lambda: remove_if_exit(lock_files))
            for split in splits:
                filename = self._get_data(split)
                lock_file = os.path.join(DATA_HOME, self.__class__.__name__)
                if self.name is not None:
                    lock_file = lock_file + "." + self.name
                lock_file += "." + split + ".done" + "." + str(os.getppid())
                # `lock_file` indicates the finished status of`_get_data`.
                # `_get_data` only works in the `unique_endpoints` specified
                # proc since `get_path_from_url` only work for it. The other
                # procs wait `_get_data` to be finished.
                if parallel_env.current_endpoint in unique_endpoints:
                    f = open(lock_file, "w")
                    f.close()
                else:
                    while not os.path.exists(lock_file):
                        time.sleep(1)
                datasets[split] = self.read(filename=filename, split=split)
        else:
            assert (
                isinstance(data_files, str) or isinstance(data_files, tuple) or isinstance(data_files, list)
            ), "`data_files` should be a string or tuple or list of strings."
            if isinstance(data_files, str):
                data_files = [data_files]
            default_split = "train"
            if splits:
                if isinstance(splits, str):
                    splits = [splits]
                datasets = DatasetTuple(splits)
                assert len(splits) == len(
                    data_files
                ), "Number of `splits` and number of `data_files` should be the same if you want to specify the split of local data file."
                for i in range(len(data_files)):
                    datasets[splits[i]] = self.read(filename=data_files[i], split=splits[i])
            else:
                datasets = DatasetTuple(["split" + str(i) for i in range(len(data_files))])
                for i in range(len(data_files)):
                    datasets["split" + str(i)] = self.read(filename=data_files[i], split=default_split)

        return datasets if len(datasets) > 1 else datasets[0]

    def read(self, filename, split="train"):
        """
        Returns a dataset containing all the examples that can be read from the file path.

        If `self.lazy` is False, this eagerly reads all instances from `self._read()`
        and returns a `MapDataset`.

        If `self.lazy` is True, this returns an `IterDataset`, which internally
        relies on the generator created from `self._read()` to lazily produce examples.
        In this case your implementation of `_read()` must also be lazy
        (that is, not load all examples into memory at once).

        Args:
            filename (str): Path of data file to read, usually provided by `_get_data`
                function.
            split (str, optional): The split name of selected dataset. This only makes
                a different when data files of different splits have different structures.

        Returns:
            A `MapDataset|IterDataset`.
        """

        label_list = self.get_labels()
        vocab_info = self.get_vocab()

        def _create_dict(labels):
            # For multiple labels in the form of list.
            if isinstance(labels[0], list) or isinstance(labels[0], tuple):
                label_dict = []
                for sub_labels in labels:
                    sub_dict = {}
                    for i, label in enumerate(sub_labels):
                        sub_dict[label] = i
                    label_dict.append(sub_dict)
            else:
                label_dict = {}
                for i, label in enumerate(labels):
                    label_dict[label] = i
            return label_dict

        def _convert_label_to_id(labels, label_dict):
            if isinstance(labels, list) or isinstance(labels, tuple):
                for label_idx in range(len(labels)):
                    labels[label_idx] = label_dict[labels[label_idx]]
            else:
                labels = label_dict[labels]
            return labels

        if self.lazy:

            def generate_examples():
                generator = (
                    self._read(filename, split) if self._read.__code__.co_argcount > 2 else self._read(filename)
                )
                for example in generator:
                    # We need to check if the example contains label column and confirm its name.
                    # For now we only allow `label` or `labels` to be the name of label column.
                    if "labels" in example.keys():
                        label_col = "labels"
                    elif "label" in example.keys():
                        label_col = "label"
                    else:
                        label_col = None

                    # Convert class label to label ids.
                    if label_list is not None and example.get(label_col, None):
                        label_dict = _create_dict(label_list)
                        # For multiple labels in the form of list.
                        if isinstance(label_dict, list):
                            for idx, sub_dict in enumerate(label_dict):
                                example[label_col][idx] = _convert_label_to_id(example[label_col][idx], sub_dict)
                        else:
                            example[label_col] = _convert_label_to_id(example[label_col], label_dict)

                        yield example
                    else:
                        yield example

            return IterDataset(generate_examples(), label_list=label_list, vocab_info=vocab_info)
        else:
            examples = self._read(filename, split) if self._read.__code__.co_argcount > 2 else self._read(filename)

            # Then some validation.
            if not isinstance(examples, list):
                examples = list(examples)

            if not examples:
                raise ValueError(
                    "No instances were read from the given filepath {}. " "Is the path correct?".format(filename)
                )

            # We need to check if the example contains label column and confirm its name.
            # For now we only allow `label` or `labels` to be the name of label column.
            if "labels" in examples[0].keys():
                label_col = "labels"
            elif "label" in examples[0].keys():
                label_col = "label"
            else:
                label_col = None

            # Convert class label to label ids.
            if label_list is not None and examples[0].get(label_col, None):
                label_dict = _create_dict(label_list)
                for idx in range(len(examples)):
                    # For multiple labels in the form of list.
                    if isinstance(label_dict, list):
                        for i, sub_dict in enumerate(label_dict):
                            examples[idx][label_col][i] = _convert_label_to_id(examples[idx][label_col][i], sub_dict)
                    else:
                        examples[idx][label_col] = _convert_label_to_id(examples[idx][label_col], label_dict)

            return MapDataset(examples, label_list=label_list, vocab_info=vocab_info)

    def _read(self, filename: str, *args):
        """
        Reads examples from the given file_path and returns them as an
        `Iterable` (which could be a list or a generator).

        This method must be implemented in self-defined `DatasetBuilder`.
        """
        raise NotImplementedError

    def _get_data(self, mode: str):
        """
        Downloads examples from the given URL and customized split
        informations and returns a filepath.

        This method must be implemented in self-defined `DatasetBuilder`.
        """
        raise NotImplementedError

    def get_labels(self):
        """
        Returns list of class labels of the dataset if specified.
        """
        return None

    def get_vocab(self):
        """
        Returns vocab file path of the dataset if specified.
        """
        return None


class SimpleBuilder(DatasetBuilder):
    def __init__(self, lazy, read_func):
        self._read = read_func
        self.lazy = lazy

    def read(self, **kwargs):
        if self.lazy:

            def generate_examples():
                generator = self._read(**kwargs)
                for example in generator:
                    yield example

            return IterDataset(generate_examples)
        else:
            examples = self._read(**kwargs)
            if hasattr(examples, "__len__") and hasattr(examples, "__getitem__"):
                return MapDataset(examples)
            else:
                return MapDataset(list(examples))
