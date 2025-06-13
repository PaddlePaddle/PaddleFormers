# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
"""Shared Memory Utils"""

from dataclasses import dataclass
from typing import List, Mapping, Tuple

import numpy as np
import paddle

from ...transformers.utils import device_guard


@dataclass
class TensorMeta:
    shape: Tuple[int] = None  # type: ignore
    dtype: paddle.dtype = None  # type: ignore
    element_size: int = 0
    numel: int = 0
    offset: int = 0


dtype_mapping = {
    paddle.float32: np.float32,
    paddle.float64: np.float64,
    paddle.int32: np.int32,
    paddle.int64: np.int64,
    paddle.uint8: np.uint8,
    paddle.int8: np.int8,
    paddle.bool: np.bool_,
    paddle.float16: np.float16,
    paddle.bfloat16: np.uint16,
    paddle.complex64: np.complex64,
    paddle.complex128: np.complex128,
}


def _write_shared_memory(value: paddle.Tensor, meta: TensorMeta, buffer):
    """
    Write a CPU tensor into the shared memory.
    """
    if value.numel() == 0:
        return
    shm_numpy = np.frombuffer(
        buffer, dtype=dtype_mapping[value.dtype], count=int(value.numel()), offset=int(meta.offset)
    )
    with device_guard("cpu"):
        shm_tensor = paddle.Tensor(shm_numpy, zero_copy=True).reshape(value.shape)
    shm_tensor.copy_(value, False)


def _traverse_copy_to_shm(value, meta, buffer):
    if isinstance(value, Mapping):
        for k, v in value.items():
            if isinstance(v, (Mapping, List)):
                m = meta[k]
                _traverse_copy_to_shm(v, m, buffer)
            elif paddle.is_tensor(v):
                m = meta[k]
                _write_shared_memory(v, m, buffer)
            else:
                meta[k] = v
    elif isinstance(value, List):
        for i, v in enumerate(value):
            if isinstance(v, (Mapping, List)):
                m = meta[i]
                _traverse_copy_to_shm(v, m, buffer)
            elif paddle.is_tensor(v):
                m = meta[i]
                _write_shared_memory(v, m, buffer)
            else:
                meta[i] = v


def _read_ndarray_from_buf(value, shm_tensor_buffer):
    """
    Read a numpy array from the buffer of shared memory.
    """
    if isinstance(value, TensorMeta):
        if value.numel == 0:
            return np.array([], dtype=dtype_mapping[value.dtype])
        else:
            shm_numpy = np.frombuffer(
                buffer=shm_tensor_buffer.buf,
                dtype=dtype_mapping[value.dtype],
                offset=value.offset,
                count=value.numel,
            ).reshape(value.shape)
            return shm_numpy
    else:
        return value


def _read_state_dict_from_shm(meta_dict, tensor_shm):
    state_dict = _traverse_state_dict(
        meta_dict,
        lambda x: _read_ndarray_from_buf(x, tensor_shm),
    )
    return state_dict


def _traverse_state_dict(value, visitor):
    """
    Invoke ``visitor`` for each value recursively in ``state_dict``.
    """
    if isinstance(value, Mapping):
        temp_dict = {}
        for k, v in value.items():
            temp_dict[k] = _traverse_state_dict(v, visitor)
        return temp_dict
    elif isinstance(value, List):
        temp_list = []
        for _, v in enumerate(value):
            temp_list.append(_traverse_state_dict(v, visitor))
        return temp_list
    else:
        return visitor(value)


def create_meta_dict(state_dict):
    buffer_size = 0

    def _create_tensor_meta(value: paddle.Tensor):
        nonlocal buffer_size
        if not paddle.is_tensor(value):
            return value
        meta = TensorMeta(
            shape=tuple(value.shape),  # type: ignore
            dtype=value.dtype,
            element_size=value.element_size(),
            numel=int(value.numel()),
            offset=int(buffer_size),
        )
        buffer_size += value.numel() * value.element_size()
        return meta

    meta_dict = _traverse_state_dict(state_dict, _create_tensor_meta)
    return meta_dict, buffer_size
