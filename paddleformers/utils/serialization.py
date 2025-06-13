# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import io
import os
import pickle
from functools import lru_cache
from typing import Union
from zipfile import ZipFile

import numpy as np
import paddle
from _io import BufferedReader
from safetensors import deserialize

from .env import PYTORCH_WEIGHTS_NAME, SAFE_WEIGHTS_NAME

MZ_ZIP_LOCAL_DIR_HEADER_SIZE = 30


_TYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "U64": np.uint64,
    "I32": np.int32,
    "U32": np.uint32,
    "I16": np.int16,
    "U16": np.uint16,
    "BF16": np.uint16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": bool,
}


class SerializationError(Exception):
    """Exception for serialization"""

    pass


def seek_by_string(file_handler: BufferedReader, string: str, file_size: int) -> int:
    """seek the index of file-handler with target words
    Args:
        file_handler (BufferedReader): file handler
        string (str): the specific string in the file
        file_size (int): size of file
    Returns:
        int: end index of target string
    """
    word_index = 0
    word_bytes = string.encode("latin")
    empty_byte = "".encode("latin")

    while word_index < len(string) and file_handler.tell() < file_size:
        content = file_handler.read(1)
        if content == empty_byte:
            break

        if word_bytes[word_index] == content[0]:
            word_index += 1
        else:
            word_index = 0

    if file_handler.tell() >= file_size - 1:
        raise SerializationError(f"can't find the find the target string<{string}> in the file")
    return file_handler.tell()


def read_prefix_key(path):
    file_size = os.stat(path).st_size
    with open(path, "rb") as file_handler:
        end_index = seek_by_string(file_handler, "data.pkl", file_size)
        file_handler.seek(MZ_ZIP_LOCAL_DIR_HEADER_SIZE)
        prefix_key = file_handler.read(end_index - MZ_ZIP_LOCAL_DIR_HEADER_SIZE - len("/data.pkl"))
    return prefix_key.decode("latin")


def _maybe_decode_ascii(bytes_str: Union[bytes, str]) -> str:
    if isinstance(bytes_str, bytes):
        return bytes_str.decode("ascii")
    return bytes_str


@lru_cache(maxsize=None)
def _storage_type_to_dtype_to_map():
    """convert storage type to numpy dtype"""
    return {
        "DoubleStorage": np.double,
        "FloatStorage": np.float32,
        "HalfStorage": np.half,
        "LongStorage": np.int64,
        "IntStorage": np.int32,
        "ShortStorage": np.int16,
        "CharStorage": np.int8,
        "ByteStorage": np.uint8,
        "BoolStorage": np.bool_,
        "ComplexDoubleStorage": np.cdouble,
        "ComplexFloatStorage": np.cfloat,
        "BFloat16Storage": np.uint16,  # support bf16
    }


class StorageType:
    """Temp Class for Storage Type"""

    def __init__(self, name):
        self.dtype = _storage_type_to_dtype_to_map()[name]

    def __str__(self):
        return f"StorageType(dtype={self.dtype})"


def _element_size(dtype: str) -> int:
    """
    Returns the element size for a dtype, in bytes
    """
    if dtype in [np.float16, np.float32, np.float64]:
        return np.finfo(dtype).bits >> 3
    elif dtype == np.bool_:
        return 1
    else:
        return np.iinfo(dtype).bits >> 3


class UnpicklerWrapperStage(pickle.Unpickler):
    def find_class(self, mod_name, name):
        if type(name) is str and "Storage" in name:
            try:
                return StorageType(name)
            except KeyError:
                pass

        if mod_name == "torch._utils":
            # rebuild torch.nn.Papameter
            if name == "_rebuild_parameter":
                return _rebuild_parameter
            # rebuild torch.nn.Papameter with state
            if name == "_rebuild_parameter_with_state":
                return _rebuild_parameter_with_state
            # rebuild torch.Tensor
            return _rebuild_tensor_stage

        # pytorch_lightning tensor builder
        if "pytorch_lightning" in mod_name:
            return dumpy
        return super().find_class(mod_name, name)


class SafeUnpickler(pickle.Unpickler):
    """
    A safe unpickler that only allows loading of built-in basic data types.
    """

    def find_class(self, module, name):
        """
        Overrides the find_class method to only allow loading of built-in basic data types.

        :param module: The module name.
        :param name: The class name.
        :return: The class if allowed, otherwise raises UnpicklingError.
        """
        if module == "builtins" and name in {"int", "float", "str", "tuple", "list", "dict", "set"}:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Unsafe object loading is prohibited: {module}.{name}")


def _rebuild_tensor_stage(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    # if a tensor has shape [M, N] and stride is [1, N], it's column-wise / fortran-style
    # if a tensor has shape [M, N] and stride is [M, 1], it's row-wise / C-style
    # defaults to C-style
    if stride is not None and len(stride) > 1 and stride[0] == 1 and stride[1] > 1:
        order = "F"
    else:
        order = "C"

    # fix bug when load https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    numel = int(np.prod(size))
    return storage[storage_offset : storage_offset + numel].reshape(size, order=order)


def _rebuild_parameter(data, requires_grad, backward_hooks):
    return data


def _rebuild_parameter_with_state(data, requires_grad, backward_hooks, state):
    return data


def dumpy(*args, **kwarsg):
    return None


def load_torch(path: str, **pickle_load_args):
    from ..transformers.utils import device_guard

    if path.endswith(PYTORCH_WEIGHTS_NAME) or os.path.split(path)[-1].startswith("pytorch_model-"):
        import torch

        state_dict = torch.load(path, map_location="cpu", weights_only=False)

        for key in list(state_dict.keys()):
            if isinstance(state_dict[key], torch.Tensor):
                t = state_dict.pop(key)
                capsule = torch.utils.dlpack.to_dlpack(t)
                t = paddle.utils.dlpack.from_dlpack(capsule)
                state_dict[key] = t
        return state_dict
    elif path.endswith(SAFE_WEIGHTS_NAME) or os.path.split(path)[-1].startswith("model-"):
        # torch safetensors -> numpy -> paddle.Tensor
        with open(path, "rb") as f:
            data = f.read()

        flat = deserialize(data)
        state_dict = {}
        for k, v in flat:
            dtype = _TYPES[v["dtype"]]
            with device_guard("cpu"):
                if v["dtype"] == "BF16":
                    arr = paddle.Tensor(
                        np.frombuffer(v["data"], dtype=dtype).reshape(v["shape"]), dtype="bfloat16", zero_copy=True
                    )
                else:
                    arr = paddle.Tensor(np.frombuffer(v["data"], dtype=dtype).reshape(v["shape"]), zero_copy=True)

            state_dict[k] = arr

    return state_dict


def load_torch_inner(path: str, **pickle_load_args):
    """
    load torch weight file with the following steps:
    1. load the structure of pytorch weight file
    2. read the tensor data and re-construct the state-dict
    Args:
        path: the path of pytorch weight file
        **pickle_load_args: args of pickle module
    Returns:
    """

    if path.endswith(PYTORCH_WEIGHTS_NAME) or os.path.split(path)[-1].startswith("pytorch_model-"):
        pickle_load_args.update({"encoding": "utf-8"})

        prefix_key = read_prefix_key(path)

        torch_zip = ZipFile(path, "r")
        loaded_storages = {}

        def load_tensor(dtype, numel, key, location):
            name = f"{prefix_key}/data/{key}"
            typed_storage = np.frombuffer(torch_zip.open(name).read()[:numel], dtype=dtype)
            return typed_storage

        def persistent_load(saved_id):
            assert isinstance(saved_id, tuple)
            typename = _maybe_decode_ascii(saved_id[0])
            data = saved_id[1:]

            assert (
                typename == "storage"
            ), f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
            storage_type, key, location, numel = data
            dtype = storage_type.dtype

            if key in loaded_storages:
                typed_storage = loaded_storages[key]
            else:
                nbytes = numel * _element_size(dtype)
                typed_storage = load_tensor(dtype, nbytes, key, _maybe_decode_ascii(location))
                loaded_storages[key] = typed_storage

            return typed_storage

        data_iostream = torch_zip.open(f"{prefix_key}/data.pkl").read()
        unpickler_stage = UnpicklerWrapperStage(io.BytesIO(data_iostream), **pickle_load_args)
        unpickler_stage.persistent_load = persistent_load
        state_dict = unpickler_stage.load()
        torch_zip.close()
    elif path.endswith(SAFE_WEIGHTS_NAME) or os.path.split(path)[-1].startswith("model-"):
        # torch safetensors -> numpy -> paddle.Tensor
        with open(path, "rb") as f:
            data = f.read()

        flat = deserialize(data)
        state_dict = {}
        for k, v in flat:
            dtype = _TYPES[v["dtype"]]
            if v["dtype"] == "BF16":
                arr = paddle.to_tensor(np.frombuffer(v["data"], dtype=dtype).reshape(v["shape"]), dtype="bfloat16")
            else:
                arr = paddle.to_tensor(np.frombuffer(v["data"], dtype=dtype).reshape(v["shape"]))
            state_dict[k] = arr

    return state_dict
