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

import ml_dtypes
import numpy as np
import paddle
import paddle.nn as nn

np.bfloat16 = ml_dtypes.bfloat16
np.float8_e5m2 = ml_dtypes.float8_e5m2
np.float8_e4m3fn = ml_dtypes.float8_e4m3fn


origin_repr = nn.Embedding.extra_repr


def new_repr(self):
    return origin_repr(self) + f", dtype={self.weight.dtype}"


nn.Embedding.extra_repr = new_repr

origin_tensort_init = paddle.Tensor.__call__
origin_to_tensor = paddle.to_tensor
origin_numpy = paddle.Tensor.numpy
origin_numel = paddle.Tensor.numel
origin_set_value = paddle.core.eager.Tensor.set_value


paddle_numpy_mapping = {
    paddle.float8_e5m2: (paddle.int8, np.float8_e5m2),
    paddle.float8_e4m3fn: (paddle.int8, np.float8_e4m3fn),
    # paddle.bfloat16: (paddle.int16, np.bfloat16),
}

numpy_paddle_mapping = {
    np.dtype(np.float8_e5m2): (np.int8, paddle.float8_e5m2),
    np.dtype(np.float8_e4m3fn): (np.int8, paddle.float8_e4m3fn),
    np.dtype(np.bfloat16): (np.uint16, paddle.bfloat16),
}

paddle_numel_mapping = {
    paddle.float8_e5m2: (paddle.int8, None),
    paddle.float8_e4m3fn: (paddle.int8, None),
}

paddle_set_value_mapping = {
    paddle.float8_e5m2: (paddle.int8, None),
    paddle.float8_e4m3fn: (paddle.int8, None),
    # paddle.bfloat16: (paddle.int16, None),
    np.dtype(np.float8_e5m2): (np.int8, paddle.float8_e5m2),
    np.dtype(np.float8_e4m3fn): (np.int8, paddle.float8_e4m3fn),
}


def enhance_init(*args, **kwargs):
    if len(args) > 0 and isinstance(args[0], np.ndarray) and args[0].dtype in numpy_paddle_mapping:
        inter_dtype, tgt_dtype = numpy_paddle_mapping[args[0].dtype]
        tensor = args[0].view(inter_dtype)
        new_args = (tensor, *args[1:])
        tensor = origin_tensort_init(*new_args, **kwargs)
        return tensor.view(tgt_dtype)
    return origin_tensort_init(*args, **kwargs)


def enhance_to_tensor(*args, **kwargs):
    # Fix with kwargs input
    if len(args) > 0:
        tensor = args[0]
    else:
        tensor = kwargs.get("data", None)

    if isinstance(tensor, np.ndarray) and tensor.dtype in numpy_paddle_mapping:
        inter_dtype, tgt_dtype = numpy_paddle_mapping[tensor.dtype]
        tensor = tensor.view(inter_dtype)
        if "data" in kwargs:
            new_args = args
            kwargs["data"] = tensor
        else:
            new_args = (tensor, *args[1:])
        tensor = origin_to_tensor(*new_args, **kwargs)
        return tensor.view(tgt_dtype)
    return origin_to_tensor(*args, **kwargs)


def enhance_set_value(self, *args, **kwargs):
    # Fix with kwargs input
    if len(args) > 0:
        tensor = args[0]
    else:
        tensor = kwargs.get("value", None)

    if isinstance(tensor, np.ndarray) and tensor.dtype in paddle_set_value_mapping:
        inter_dtype, tgt_dtype = paddle_set_value_mapping[tensor.dtype]
        tensor = tensor.view(inter_dtype)
        if "value" in kwargs:
            new_args = args
            kwargs["value"] = tensor
        else:
            new_args = (tensor, *args[1:])
        return origin_set_value(self, *new_args, **kwargs)

    if isinstance(tensor, paddle.Tensor) and tensor.dtype in paddle_set_value_mapping:
        inter_dtype, _ = paddle_set_value_mapping[tensor.dtype]
        tensor = tensor.view(inter_dtype)
        if "value" in kwargs:
            new_args = args
            kwargs["value"] = tensor
        else:
            new_args = (tensor, *args[1:])
        new_self = self.view(inter_dtype)
        return origin_set_value(new_self, *new_args, **kwargs)

    return origin_set_value(self, *args, **kwargs)


def _numpy(self, *args, **kwargs):
    if self.dtype in paddle_numpy_mapping:
        inter_pd_dtype, np_dtype = paddle_numpy_mapping[self.dtype]
        tensor = origin_numpy(self.view(inter_pd_dtype), *args, **kwargs)
        return tensor.view(np_dtype)
    return origin_numpy(self, *args, **kwargs)


def _numel(self, *args, **kwargs):
    if self.dtype in paddle_numel_mapping:
        inter_pd_dtype, _ = paddle_numel_mapping[self.dtype]
        ret = origin_numel(self.view(inter_pd_dtype), *args, **kwargs)
        return ret
    return origin_numel(self, *args, **kwargs)


paddle.Tensor.numpy = _numpy
paddle.Tensor.__call__ = enhance_init
paddle.to_tensor = enhance_to_tensor
paddle.core.eager.Tensor.set_value = enhance_set_value
paddle.Tensor.numel = _numel
