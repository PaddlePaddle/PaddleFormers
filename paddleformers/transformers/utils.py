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
from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib
import inspect
import os
import re
import shutil
import sys
import warnings
from contextlib import ExitStack
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, ContextManager, List, Optional, Type, Union

from filelock import FileLock

from paddleformers import __version__

from ..utils.downloader import (
    COMMUNITY_MODEL_PREFIX,
    download_check,
    get_path_from_url_with_filelock,
    is_url,
    url_file_exists,
)

if TYPE_CHECKING:
    from paddleformers.transformers import PretrainedModel

import numpy as np
import paddle
import tqdm
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from huggingface_hub.utils import EntryNotFoundError
from paddle.common_ops_import import convert_dtype
from paddle.nn import Layer
from requests.exceptions import HTTPError

from ..utils.download import resolve_file_path
from ..utils.env import HF_CACHE_HOME, MODEL_HOME
from ..utils.import_utils import import_module
from ..utils.log import logger
from .aistudio_utils import aistudio_download

HUGGINGFACE_CO_RESOLVE_ENDPOINT = "https://huggingface.co"


def convert_ndarray_dtype(np_array: np.ndarray, target_dtype: str) -> np.ndarray:
    """convert ndarray

    Args:
        np_array (np.ndarray): numpy ndarray instance
        target_dtype (str): the target dtype

    Returns:
        np.ndarray: converted numpy ndarray instance
    """
    source_dtype = convert_dtype(np_array.dtype)
    if source_dtype == "uint16" or target_dtype == "bfloat16":
        tensor = paddle.to_tensor(np_array)
        tensor = paddle.cast(tensor, target_dtype)
        return tensor.cpu().numpy()

        # TODO(wj-Mcat): device_guard will slow the converting
        # with device_guard("cpu"):
        #     tensor = paddle.to_tensor(np_array)
        #     tensor = paddle.cast(tensor, target_dtype)
        # return tensor.cpu().numpy()

    if target_dtype == "bfloat16":
        target_dtype = "uint16"

    return np_array.astype(target_dtype)


def convert_to_dict_message(conversation: List[List[str]]):
    """Convert the list of chat messages to a role dictionary chat messages."""
    conversations = []
    for index, item in enumerate(conversation):
        assert 1 <= len(item) <= 2, "Each Rounds in conversation should have 1 or 2 elements."
        if isinstance(item[0], str):
            conversations.append({"role": "user", "content": item[0]})
            if len(item) == 2 and isinstance(item[1], str):
                conversations.append({"role": "assistant", "content": item[1]})
            else:
                # If there is only one element in item, it must be the last round.
                # If it is not the last round, it must be an error.
                if index != len(conversation) - 1:
                    raise ValueError(f"Round {index} has error round")
        else:
            raise ValueError("Each round in list should be string")
    return conversations


def get_scale_by_dtype(dtype: str = None, return_positive: bool = True) -> float:
    """get scale value by dtype

    Args:
        dtype (str): the string dtype value

    Returns:
        float: the scale value
    """
    if dtype is None:
        dtype = paddle.get_default_dtype()

    dtype = convert_dtype(dtype)
    scale_value = 1e6

    # TODO(wj-Mcaf): support int8, int4 dtypes later
    if dtype == "float16":
        scale_value = 1e4

    if return_positive:
        return scale_value
    return -1 * scale_value


def fn_args_to_dict(func, *args, **kwargs):
    """
    Inspect function `func` and its arguments for running, and extract a
    dict mapping between argument names and keys.
    """
    if hasattr(inspect, "getfullargspec"):
        (spec_args, spec_varargs, spec_varkw, spec_defaults, _, _, _) = inspect.getfullargspec(func)
    else:
        (spec_args, spec_varargs, spec_varkw, spec_defaults) = inspect.getargspec(func)
    # add positional argument values
    init_dict = dict(zip(spec_args, args))
    # add default argument values
    kwargs_dict = dict(zip(spec_args[-len(spec_defaults) :], spec_defaults)) if spec_defaults else {}
    for k in list(kwargs_dict.keys()):
        if k in init_dict:
            kwargs_dict.pop(k)
    kwargs_dict.update(kwargs)
    init_dict.update(kwargs_dict)
    return init_dict


def adapt_stale_fwd_patch(self, name, value):
    """
    Since there are some monkey patches for forward of PretrainedModel, such as
    model compression, we make these patches compatible with the latest forward
    method.
    """
    if name == "forward":
        # NOTE(guosheng): In dygraph to static, `layer.forward` would be patched
        # by an instance of `StaticFunction`. And use string compare to avoid to
        # import fluid.
        if type(value).__name__.endswith("StaticFunction") or self.forward.__class__.__name__.endswith(
            "StaticFunction"
        ):
            return value
        if type(value).__name__.endswith("WeakMethod") or self.forward.__class__.__name__.endswith("WeakMethod"):
            return value

        # NOTE(changwenbin & zhoukangkang):
        # When use model = paddle.incubate.jit.inference(model), it reportes errors, we fix it here.
        # is_inference_mode API is only available in PaddlePaddle develop，so we add a try except.
        try:
            from paddle.incubate.jit import is_inference_mode

            if is_inference_mode(value):
                return value
        except:
            pass

        if hasattr(inspect, "getfullargspec"):
            (
                patch_spec_args,
                patch_spec_varargs,
                patch_spec_varkw,
                patch_spec_defaults,
                _,
                _,
                _,
            ) = inspect.getfullargspec(value)
            (spec_args, spec_varargs, spec_varkw, spec_defaults, _, _, _) = inspect.getfullargspec(self.forward)
        else:
            (patch_spec_args, patch_spec_varargs, patch_spec_varkw, patch_spec_defaults) = inspect.getargspec(value)
            (spec_args, spec_varargs, spec_varkw, spec_defaults) = inspect.getargspec(self.forward)
        new_args = [
            arg
            for arg in ("output_hidden_states", "output_attentions", "return_dict")
            if arg not in patch_spec_args and arg in spec_args
        ]

        if new_args:
            if self.__module__.startswith("paddleformers"):
                warnings.warn(
                    f"The `forward` method of {self.__class__ if isinstance(self, Layer) else self} is patched and the patch "
                    "might be based on an old version which missing some "
                    f"arguments compared with the latest, such as {new_args}. "
                    "We automatically add compatibility on the patch for "
                    "these arguments, and maybe the patch should be updated."
                )
            else:
                warnings.warn(
                    f"The `forward` method of {self.__class__ if isinstance(self, Layer) else self} "
                    "is patched and the patch might be conflict with patches made "
                    f"by paddleformers which seems have more arguments such as {new_args}. "
                    "We automatically add compatibility on the patch for "
                    "these arguments, and maybe the patch should be updated."
                )
            if isinstance(self, Layer) and inspect.isfunction(value):

                @functools.wraps(value)
                def wrap_fwd(*args, **kwargs):
                    for arg in new_args:
                        kwargs.pop(arg, None)
                    return value(self, *args, **kwargs)

            else:

                @functools.wraps(value)
                def wrap_fwd(*args, **kwargs):
                    for arg in new_args:
                        kwargs.pop(arg, None)
                    return value(*args, **kwargs)

            return wrap_fwd
    return value


class InitTrackerMeta(type(Layer)):
    """
    This metaclass wraps the `__init__` method of a class to add `init_config`
    attribute for instances of that class, and `init_config` use a dict to track
    the initial configuration. If the class has `_pre_init` or `_post_init`
    method, it would be hooked before or after `__init__` and called as
    `_pre_init(self, init_fn, init_args)` or `_post_init(self, init_fn, init_args)`.
    Since InitTrackerMeta would be used as metaclass for pretrained model classes,
    which always are Layer and `type(Layer)` is not `type`, thus use `type(Layer)`
    rather than `type` as base class for it to avoid inheritance metaclass
    conflicts.
    """

    def __init__(cls, name, bases, attrs):
        init_func = cls.__init__
        # If attrs has `__init__`, wrap it using accessible `_pre_init, _post_init`.
        # Otherwise, no need to wrap again since the super cls has been wrapped.
        # TODO: remove reduplicated tracker if using super cls `__init__`
        pre_init_func = getattr(cls, "_pre_init", None) if "__init__" in attrs else None
        post_init_func = getattr(cls, "_post_init", None) if "__init__" in attrs else None
        cls.__init__ = InitTrackerMeta.init_and_track_conf(init_func, pre_init_func, post_init_func)
        super(InitTrackerMeta, cls).__init__(name, bases, attrs)

    @staticmethod
    def init_and_track_conf(init_func, pre_init_func=None, post_init_func=None):
        """
        wraps `init_func` which is `__init__` method of a class to add `init_config`
        attribute for instances of that class.
        Args:
            init_func (callable): It should be the `__init__` method of a class.
                warning: `self` always is the class type of down-stream model, eg: BertForTokenClassification
            pre_init_func (callable, optional): If provided, it would be hooked after
                `init_func` and called as `pre_init_func(self, init_func, *init_args, **init_args)`.
                Default None.
            post_init_func (callable, optional): If provided, it would be hooked after
                `init_func` and called as `post_init_func(self, init_func, *init_args, **init_args)`.
                Default None.

        Returns:
            function: the wrapped function
        """

        @functools.wraps(init_func)
        def __impl__(self, *args, **kwargs):
            # registered helper by `pre_init_func`
            if pre_init_func:
                pre_init_func(self, init_func, *args, **kwargs)
            # keep full configuration
            init_func(self, *args, **kwargs)
            # registered helper by `post_init_func`
            if post_init_func:
                post_init_func(self, init_func, *args, **kwargs)
            self.init_config = kwargs
            if args:
                kwargs["init_args"] = args
            kwargs["init_class"] = self.__class__.__name__

        return __impl__

    def __setattr__(self, name, value):
        value = adapt_stale_fwd_patch(self, name, value)
        return super(InitTrackerMeta, self).__setattr__(name, value)


def param_in_func(func, param_field: str) -> bool:
    """check if the param_field is in `func` method, eg: if the `bert` param is in `__init__` method

    Args:
        cls (type): the class of PretrainedModel
        param_field (str): the name of field

    Returns:
        bool: the result of existence
    """

    if hasattr(inspect, "getfullargspec"):
        result = inspect.getfullargspec(func)
    else:
        result = inspect.getargspec(func)

    return param_field in result[0]


def resolve_cache_dir(
    from_hf_hub: bool, from_aistudio: bool, from_modelscope: bool, cache_dir: Optional[str] = None
) -> str:
    """resolve cache dir for PretrainedModel and PretrainedConfig

    Args:
        from_hf_hub (bool): if load from huggingface hub
        from_aistudio (bool): if load from aistudio
        from_modelscope (bool): if load from modelscope
        cache_dir (str): cache_dir for models
    """
    if cache_dir is not None:
        return cache_dir
    if from_modelscope:
        return None
    if from_aistudio:
        return None
    if from_hf_hub:
        return HF_CACHE_HOME
    return MODEL_HOME


def find_transformer_model_type(model_class: Type) -> str:
    """get the model type from module name,
        eg:
            BertModel -> bert,
            RobertaForTokenClassification -> roberta

    Args:
        model_class (Type): the class of model

    Returns:
        str: the type string
    """
    from paddleformers.transformers import PretrainedModel

    default_model_type = ""

    if not issubclass(model_class, PretrainedModel):
        return default_model_type

    module_name: str = model_class.__module__
    if not module_name.startswith("paddleformers.transformers."):
        return default_model_type

    tokens = module_name.split(".")
    if len(tokens) < 3:
        return default_model_type

    return tokens[2]


def find_transformer_model_class_by_name(model_name: str) -> Optional[Type[PretrainedModel]]:
    """find transformer model_class by name

    Args:
        model_name (str): the string of class name

    Returns:
        Optional[Type[PretrainedModel]]: optional pretrained-model class
    """
    transformer_module = import_module("paddleformers.transformers")

    for obj_name in dir(transformer_module):
        if obj_name.startswith("_"):
            continue
        obj = getattr(transformer_module, obj_name, None)
        if obj is None:
            continue

        name = getattr(obj, "__name__", None)
        if name is None:
            continue

        if name == model_name:
            return obj
    logger.debug(f"can not find model_class<{model_name}>")
    return None


def convert_file_size_to_int(size: Union[int, str]):
    """
    Converts a size expressed as a string with digits an unit (like `"5MB"`) to an integer (in bytes).
    Args:
        size (`int` or `str`): The size to convert. Will be directly returned if an `int`.
    Example:
    ```py
    >>> convert_file_size_to_int("1MiB")
    1048576
    ```
    """
    if isinstance(size, int):
        return size
    if size.upper().endswith("GIB"):
        return int(size[:-3]) * (2**30)
    if size.upper().endswith("MIB"):
        return int(size[:-3]) * (2**20)
    if size.upper().endswith("KIB"):
        return int(size[:-3]) * (2**10)
    if size.upper().endswith("GB"):
        int_size = int(size[:-2]) * (10**9)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("MB"):
        int_size = int(size[:-2]) * (10**6)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("KB"):
        int_size = int(size[:-2]) * (10**3)
        return int_size // 8 if size.endswith("b") else int_size
    raise ValueError("`size` is not in a valid format. Use an integer followed by the unit, e.g., '5GB'.")


def paddleformers_hub_download(
    repo_id: str,
    filename: str,
    *,
    subfolder: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    pretrained_model_name_or_path: str = None,
) -> str:
    if subfolder is None:
        subfolder = ""
    if pretrained_model_name_or_path is not None and is_url(repo_id):
        cache_dir = os.path.join(cache_dir, pretrained_model_name_or_path, subfolder)
    else:
        cache_dir = os.path.join(cache_dir, repo_id, subfolder)

    # check in cache_dir
    weight_file_path = os.path.join(cache_dir, filename)

    if os.path.exists(weight_file_path):
        logger.info(f"Already cached {weight_file_path}")
        return weight_file_path

    # Download from custom model url
    if is_url(repo_id):
        # check whether the target file exist in the community bos server
        if url_file_exists(repo_id):
            logger.info(f"Downloading {repo_id}")
            weight_file_path = get_path_from_url_with_filelock(repo_id, cache_dir)
            # # check the downloaded weight file and registered weight file name
            download_check(repo_id, "paddleformers_hub_download")

            # make sure that model states names: model_states.pdparams
            new_weight_file_path = os.path.join(os.path.split(weight_file_path)[0], filename)

            if weight_file_path != new_weight_file_path:
                # create lock file, which is empty, under the `LOCK_FILE_HOME` directory.
                lock_file_name = hashlib.md5((repo_id + cache_dir).encode("utf-8")).hexdigest()
                # create `.lock` private directory in the cache dir
                lock_file_path = os.path.join(cache_dir, ".lock", lock_file_name)

                with FileLock(lock_file_path):
                    if not os.path.exists(new_weight_file_path):
                        shutil.move(weight_file_path, new_weight_file_path)

                weight_file_path = new_weight_file_path

            return weight_file_path

        return None

    # find in community repo
    url_list = [COMMUNITY_MODEL_PREFIX, repo_id, filename]
    if subfolder != "":
        url_list.insert(2, subfolder)
    community_model_file_path = "/".join(url_list)
    assert is_url(community_model_file_path)

    # check whether the target file exist in the community bos server
    if url_file_exists(community_model_file_path):
        logger.info(f"Downloading {community_model_file_path}")
        weight_file_path = get_path_from_url_with_filelock(community_model_file_path, cache_dir)
        # # check the downloaded weight file and registered weight file name
        download_check(community_model_file_path, "paddleformers_hub_download")
        return weight_file_path

    return None


# Return value when trying to load a file from cache but the file does not exist in the distant repo.
_CACHED_NO_EXIST = object()


def cached_file(
    path_or_repo_id: Union[str, os.PathLike],
    filename: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    subfolder: str = "",
    from_aistudio: bool = False,
    from_modelscope: bool = False,
    _raise_exceptions_for_missing_entries: bool = True,
    _raise_exceptions_for_connection_errors: bool = True,
    pretrained_model_name_or_path=None,
) -> str:
    """
    Tries to locate a file in a local folder and repo, downloads and cache it if necessary.
    Args:
        path_or_repo_id (`str` or `os.PathLike`):
            This can be either:
            - a string, the *model id* of a model repo on huggingface.co.
            - a path to a *directory* potentially containing the file.
        filename (`str`):
            The name of the file to locate in `path_or_repo`.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.

    Returns:
        `Optional[str]`: Returns the resolved file (to the cache folder if downloaded from a repo).
    Examples:
    ```python
    # Download a model weight from the Hub and cache it.
    model_weights_file = cached_file("bert-base-uncased", "pytorch_model.bin")
    ```
    """

    if subfolder is None:
        subfolder = ""

    path_or_repo_id = str(path_or_repo_id)
    full_filename = os.path.join(subfolder, filename)
    if os.path.isdir(path_or_repo_id):
        resolved_file = os.path.join(os.path.join(path_or_repo_id, subfolder), filename)
        if not os.path.isfile(resolved_file):
            if _raise_exceptions_for_missing_entries:
                raise EnvironmentError(
                    f"{path_or_repo_id} does not appear to have a file named {full_filename}. Checkout "
                    f"'https://huggingface.co/{path_or_repo_id}/' for available files."
                )
            else:
                return None
        return resolved_file

    if cache_dir is not None and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if from_aistudio:
        try:
            resolved_file = aistudio_download(
                repo_id=path_or_repo_id, filename=filename, subfolder=subfolder, cache_dir=cache_dir
            )
        except:
            resolved_file = None
    else:
        # if cache_dir is None:
        #     cache_dir = os.path.join(MODEL_HOME, ".cache")
        try:
            # Load from URL or cache if already cached
            resolved_file = paddleformers_hub_download(
                path_or_repo_id,
                filename,
                subfolder=None if len(subfolder) == 0 else subfolder,
                # revision=revision,
                cache_dir=cache_dir,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
            )
        except HTTPError as err:
            # First we try to see if we have a cached version (not up to date):
            resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir)
            if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
                return resolved_file
            if not _raise_exceptions_for_connection_errors:
                return None

            raise EnvironmentError(
                f"There was a specific connection error when trying to load {path_or_repo_id}:\n{err}"
            )

    return resolved_file


def cached_file_for_hf_hub(
    path_or_repo_id: Union[str, os.PathLike],
    filename: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    subfolder: str = "",
    _raise_exceptions_for_missing_entries: bool = True,
):

    if subfolder is None:
        subfolder = ""

    path_or_repo_id = str(path_or_repo_id)
    full_filename = os.path.join(subfolder, filename)
    if os.path.isdir(path_or_repo_id):
        resolved_file = os.path.join(os.path.join(path_or_repo_id, subfolder), filename)
        if not os.path.isfile(resolved_file):
            if _raise_exceptions_for_missing_entries:
                raise EnvironmentError(
                    f"{path_or_repo_id} does not appear to have a file named {full_filename}. Checkout "
                    f"'https://huggingface.co/{path_or_repo_id}' for available files."
                )
            else:
                return None
        return resolved_file

    if cache_dir is None:
        cache_dir = os.path.join(MODEL_HOME, ".cache")
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    try:
        # Load from URL or cache if already cached
        download_check(path_or_repo_id, full_filename, addition="from_hf_hub")
        resolved_file = hf_hub_download(
            repo_id=path_or_repo_id,
            filename=filename,
            cache_dir=cache_dir,
            subfolder=subfolder,
            library_name="PaddleNLP",
            library_version=__version__,
        )
        return resolved_file
    except Exception as e:
        print(e)
        msg = f"""
            {path_or_repo_id} is not a local folder and is not a valid model identifier "
            "listed on 'https://huggingface.co/models' If this is a private repository, make sure to "
            "pass a token having permission to this repo with `use_auth_token` or log in with "
            "`huggingface-cli login` and pass `use_auth_token=True`.
            """
        if _raise_exceptions_for_missing_entries:
            raise EnvironmentError(msg)
        else:
            logger.info(msg)
            return None


def get_checkpoint_shard_files(
    pretrained_model_name_or_path,
    index_filename,
    cache_dir=None,
    subfolder="",
    from_aistudio=False,
    from_modelscope=False,
    from_hf_hub=False,
):
    """
    For a given model:
    - download and cache all the shards of a sharded checkpoint if `pretrained_model_name_or_path` is a model ID on the
      Hub
    - returns the list of paths to all the shards, as well as some metadata.
    For the description of each arg, see [`PretrainedModel.from_pretrained`]. `index_filename` is the full path to the
    index (downloaded and cached if `pretrained_model_name_or_path` is a model ID on the Hub).
    """

    import json

    if not os.path.isfile(index_filename):
        raise ValueError(f"Can't find a checkpoint index ({index_filename}) in {pretrained_model_name_or_path}.")

    with open(index_filename, "r") as f:
        index = json.loads(f.read())

    shard_filenames = sorted(set(index["weight_map"].values()))
    sharded_metadata = index["metadata"]
    sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
    sharded_metadata["weight_map"] = index["weight_map"].copy()

    file_map = {file: set() for file in shard_filenames}
    for weight, file in index["weight_map"].items():
        file_map[file].add(weight)

    sharded_metadata["file_map"] = file_map

    # First, let's deal with local folder.
    if os.path.isdir(pretrained_model_name_or_path):
        shard_filenames = [os.path.join(pretrained_model_name_or_path, subfolder, f) for f in shard_filenames]
        return shard_filenames, sharded_metadata

    # At this stage pretrained_model_name_or_path is a model identifier on the Hub
    cached_filenames = []
    # Check if the model is already cached or not. We only try the last checkpoint, this should cover most cases of
    # downloaded (if interrupted).
    last_shard = try_to_load_from_cache(
        pretrained_model_name_or_path,
        shard_filenames[-1],
        cache_dir=cache_dir,
    )

    show_progress_bar = last_shard is None
    for shard_filename in tqdm.tqdm(shard_filenames, desc="Downloading shards", disable=not show_progress_bar):
        try:
            cached_filename = resolve_file_path(
                pretrained_model_name_or_path,
                [shard_filename],
                subfolder,
                cache_dir=cache_dir,
                from_hf_hub=from_hf_hub,
                from_aistudio=from_aistudio,
                from_modelscope=from_modelscope,
            )
            assert (
                cached_filename is not None
            ), f"please make sure {shard_filename} under {pretrained_model_name_or_path}"
        # We have already dealt with RepositoryNotFoundError and RevisionNotFoundError when getting the index, so
        # we don't have to catch them here.
        except EntryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} does not appear to have a file named {shard_filename} which is "
                "required according to the checkpoint index."
            )
        except HTTPError:
            raise EnvironmentError(
                f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load {shard_filename}. You should try"
                " again after checking your internet connection."
            )

        cached_filenames.append(cached_filename)

    return cached_filenames, sharded_metadata


def is_safetensors_available():
    return importlib.util.find_spec("safetensors") is not None


@contextlib.contextmanager
def device_guard(device="cpu", dev_id=0):
    origin_device = paddle.device.get_device()
    if device == "cpu":
        paddle.set_device(device)
    elif device in ["gpu", "xpu", "npu"]:
        paddle.set_device("{}:{}".format(device, dev_id))
    try:
        yield
    finally:
        paddle.set_device(origin_device)


def paddleformers_load(path, map_location="cpu"):
    assert map_location in ["cpu", "gpu", "xpu", "npu", "numpy", "np"]
    if map_location in ["numpy", "np"]:
        return paddle.load(path, return_numpy=True)
    else:
        with device_guard(map_location):
            return paddle.load(path)
            # TODO(zhonghui03): the following code has problems when hot start optimizer checkpoint.
            if map_location == "cpu":
                from paddle.framework.io import (
                    _parse_every_object,
                    _to_LodTensor,
                    _transformed_from_lodtensor,
                )

                def _ndarray_to_tensor(obj, return_numpy=False):
                    if return_numpy:
                        return obj
                    if paddle.in_dynamic_mode():
                        return paddle.Tensor(obj, zero_copy=True)
                    else:
                        return _to_LodTensor(obj)

                state_dict = paddle.load(path, return_numpy=True)
                # Hack for zero copy for saving loading time. for paddle.load there need copy to create paddle.Tensor
                return _parse_every_object(state_dict, _transformed_from_lodtensor, _ndarray_to_tensor)

            else:
                return paddle.load(path)


def is_paddle_support_lazy_init():
    return hasattr(paddle, "LazyGuard")


class ContextManagers:
    """
    Wrapper for `contextlib.ExitStack` which enters a collection of context managers. Adaptation of `ContextManagers`
    in the `fastcore` library.
    """

    def __init__(self, context_managers: List[ContextManager]):
        self.context_managers = context_managers
        self.stack = ExitStack()

    def __enter__(self):
        for context_manager in self.context_managers:
            self.stack.enter_context(context_manager)

    def __exit__(self, *args, **kwargs):
        self.stack.__exit__(*args, **kwargs)


def use_hybrid_parallel():
    try:
        from paddle.distributed import fleet

        hcg = fleet.get_hybrid_communicate_group()
        return hcg
    except:
        return None


def optimizer_name_suffix():
    hcg = use_hybrid_parallel()
    if hcg is not None:
        name = []
        if hcg.get_model_parallel_world_size() > 1:
            name.append(f"tp{hcg.get_model_parallel_rank():0>2d}")
        if hcg.get_pipe_parallel_world_size() > 1:
            name.append(f"pp{hcg.get_stage_id():0>2d}")
        if hcg.get_sharding_parallel_world_size() > 1:
            name.append(f"shard{hcg.get_sharding_parallel_rank():0>2d}")

        return "_".join(name)
    else:
        return None


def weight_name_suffix():
    hcg = use_hybrid_parallel()
    if hcg is not None:
        name = []
        if hcg.get_model_parallel_world_size() > 1:
            name.append(f"tp{hcg.get_model_parallel_rank():0>2d}")
        if hcg.get_pipe_parallel_world_size() > 1:
            name.append(f"pp{hcg.get_stage_id():0>2d}")
        return "_".join(name)
    else:
        return None


def dtype_byte_size(dtype):
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`.

    Example:

    ```py
    >>> dtype_byte_size(paddle.float32)
    4
    ```
    """
    if dtype == paddle.bool:
        return 1 / 8
    if dtype == paddle.float8_e4m3fn or dtype == paddle.float8_e5m2:
        return 1
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


def apply_print_resets(buf):
    return re.sub(r"^.*\r", "", buf, 0, re.M)


class CaptureStd:
    """
    Context manager to capture:

        - stdout: replay it, clean it up and make it available via `obj.out`
        - stderr: replay it and make it available via `obj.err`

    Args:
        out (`bool`, *optional*, defaults to `True`): Whether to capture stdout or not.
        err (`bool`, *optional*, defaults to `True`): Whether to capture stderr or not.
        replay (`bool`, *optional*, defaults to `True`): Whether to replay or not.
            By default each captured stream gets replayed back on context's exit, so that one can see what the test was
            doing. If this is a not wanted behavior and the captured data shouldn't be replayed, pass `replay=False` to
            disable this feature.

    Examples:

    ```python
    # to capture stdout only with auto-replay
    with CaptureStdout() as cs:
        print("Secret message")
    assert "message" in cs.out

    # to capture stderr only with auto-replay
    import sys

    with CaptureStderr() as cs:
        print("Warning: ", file=sys.stderr)
    assert "Warning" in cs.err

    # to capture both streams with auto-replay
    with CaptureStd() as cs:
        print("Secret message")
        print("Warning: ", file=sys.stderr)
    assert "message" in cs.out
    assert "Warning" in cs.err

    # to capture just one of the streams, and not the other, with auto-replay
    with CaptureStd(err=False) as cs:
        print("Secret message")
    assert "message" in cs.out
    # but best use the stream-specific subclasses

    # to capture without auto-replay
    with CaptureStd(replay=False) as cs:
        print("Secret message")
    assert "message" in cs.out
    ```"""

    def __init__(self, out=True, err=True, replay=True):
        self.replay = replay

        if out:
            self.out_buf = StringIO()
            self.out = "error: CaptureStd context is unfinished yet, called too early"
        else:
            self.out_buf = None
            self.out = "not capturing stdout"

        if err:
            self.err_buf = StringIO()
            self.err = "error: CaptureStd context is unfinished yet, called too early"
        else:
            self.err_buf = None
            self.err = "not capturing stderr"

    def __enter__(self):
        if self.out_buf:
            self.out_old = sys.stdout
            sys.stdout = self.out_buf

        if self.err_buf:
            self.err_old = sys.stderr
            sys.stderr = self.err_buf

        return self

    def __exit__(self, *exc):
        if self.out_buf:
            sys.stdout = self.out_old
            captured = self.out_buf.getvalue()
            if self.replay:
                sys.stdout.write(captured)
            self.out = apply_print_resets(captured)

        if self.err_buf:
            sys.stderr = self.err_old
            captured = self.err_buf.getvalue()
            if self.replay:
                sys.stderr.write(captured)
            self.err = captured

    def __repr__(self):
        msg = ""
        if self.out_buf:
            msg += f"stdout: {self.out}\n"
        if self.err_buf:
            msg += f"stderr: {self.err}\n"
        return msg


def caculate_llm_per_token_flops(
    hidden_size,
    intermediate_size,
    layer_num,
    vocab_size,
    seq_length=None,
    recompute=False,
    recompute_granularity=None,
):

    # TFLOPs formula (from Equation 3 in Section 5.1 of https://arxiv.org/pdf/2104.04473.pdf).
    flops_per_transformer = 0
    flops_recompute_transformer = 0

    # qkvo matmul
    flops_qkvo_matmul = seq_length * hidden_size**2 * 4

    # [b,s,h] [b,h,s] bs^2h
    # [b,s,s] [b,s,h] bs^2h
    # q_states * k_states + attn_weight * v_states
    flops_core_attn = seq_length**2 * hidden_size * 2

    # swiglu, matmul + dot
    flops_ffn = seq_length * hidden_size * intermediate_size * 3 + seq_length * intermediate_size

    flops_per_transformer = flops_qkvo_matmul + flops_core_attn + flops_ffn
    if recompute:
        if recompute_granularity == "full":
            flops_recompute_transformer = flops_per_transformer
        if recompute_granularity == "full_attn":
            flops_recompute_transformer = flops_qkvo_matmul + flops_core_attn
        if recompute_granularity == "core_attn":
            flops_recompute_transformer = flops_core_attn

    # final loggits
    flops_loggits = seq_length * hidden_size * vocab_size

    # 2 for mul + add in matmul
    # 1 for forward, 2 for backwards since we caluate gradients for input_x and input_y
    return 2 * (layer_num * (flops_per_transformer * 3 + flops_recompute_transformer) + 3 * flops_loggits) / seq_length
