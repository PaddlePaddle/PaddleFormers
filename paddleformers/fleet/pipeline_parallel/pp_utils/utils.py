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

import paddle
from paddle.distributed.fleet.utils.log_util import get_sync_logger

PADDLE_TO_NUMBER = {
    paddle.float16: 0,
    paddle.float32: 1,
    paddle.float64: 2,
    paddle.int32: 3,
    paddle.int64: 4,
    paddle.bfloat16: 5,
    paddle.bool: 6,
}

NUMBER_TO_DTYPE = {
    0: "float16",
    1: "float32",
    2: "float64",
    3: "int32",
    4: "int64",
    5: "bfloat16",
    6: "bool",
}


def paddle_2_number(dtype):
    assert dtype in PADDLE_TO_NUMBER.keys()
    return PADDLE_TO_NUMBER[dtype]


def number_2_dtype(number):
    assert number in NUMBER_TO_DTYPE.keys()
    return NUMBER_TO_DTYPE[number]


def profile_pipeline_details(msg):
    GB = 1024.0 * 1024.0 * 1024.0
    if paddle.base.core.is_compiled_with_cuda():
        memory_allocated_size = paddle.device.cuda.memory_allocated() / GB
        memory_reserved_size = paddle.device.cuda.memory_reserved() / GB
    else:
        memory_allocated_size, memory_reserved_size = 0, 0
    get_sync_logger().info(
        f"{msg}: memory_allocated_size={memory_allocated_size:.2f}, memory_reserved_size={memory_reserved_size:.2f}"
    )


def tuple_to_dict_helper(input_tensor):
    # recv tuple -> fwd input dict
    use_dict = False
    if isinstance(input_tensor, tuple):
        use_dict = hasattr(input_tensor[0], "key")
    else:  # single tensor
        use_dict = hasattr(input_tensor, "key")
    if use_dict:
        input_tensor = convert_tensor_tuple_to_dict(input_tensor)
    return input_tensor, use_dict


def dict_to_tuple_helper(output_tensor):
    if isinstance(output_tensor, dict):
        output_tensor_tuple = convert_tensor_dict_to_tuple(
            output_tensor_dict=output_tensor
        )
    else:  # single tensor or tensor tuple
        output_tensor_tuple = output_tensor
    return output_tensor_tuple


def convert_tensor_dict_to_tuple(output_tensor_dict):
    output_tensor = []
    for key, tensor in output_tensor_dict.items():
        if isinstance(tensor, (list, tuple)):
            for idx, t in enumerate(tensor):
                t.key = key + " " + str(idx)
                output_tensor.append(t)
        else:  # single tensor
            tensor.key = key
            output_tensor.append(tensor)

    return tuple(output_tensor)


def convert_tensor_tuple_to_dict(input_tensor_tuple):
    input_tensor_dict = {}
    for tensor in input_tensor_tuple:
        key = tensor.key
        if " " in key:
            real_key, _ = key.split(" ")
            if real_key in input_tensor_dict.keys():
                input_tensor_dict[real_key].append(tensor)
            else:
                input_tensor_dict[real_key] = [tensor]
        else:
            input_tensor_dict[key] = tensor
        delattr(tensor, "key")
    return input_tensor_dict
