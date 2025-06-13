# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


def bloom_postprocess_past_key_value(past_key_values):
    # (layer_num, bs, head_num/tensor_parallel_degree, prefixlen, head_dim)*2
    keys, values = paddle.transpose(past_key_values, perm=[2, 0, 1, 3, 4]).split(2)
    # keys: [layer_num, bs, head_num/tensor_parallel_degree, head_dim, prefixlen]
    # value: [layer_num, bs, head_num/tensor_parallel_degree, prefixlen, head_dim]
    # keys, values = past_key_values[0].transpose([0, 1, 2, 4, 3]), past_key_values[1]
    return tuple(zip(keys, values))


def chatglm_postprocess_past_key_value(past_key_values):
    # (layer_num, prefixlen, bs, head_num/tensor_parallel_degree, head_dim)*2
    keys, values = paddle.transpose(past_key_values, perm=[2, 1, 0, 3, 4]).split(2)

    return tuple(zip(keys, values))


def llama_postprocess_past_key_value(past_key_values):
    # (layer_num, bs, prefixlen, head_num/tensor_parallel_degree, head_dim)*2
    keys, values = paddle.transpose(past_key_values, perm=[2, 0, 1, 3, 4]).split(2)

    return tuple(zip(keys, values))


def mistral_postprocess_past_key_value(past_key_values):
    # (layer_num, bs, head_num/tensor_parallel_degree, prefixlen, head_dim)*2
    keys, values = paddle.transpose(past_key_values, perm=[2, 0, 3, 1, 4]).split(2)

    return tuple(zip(keys, values))


def qwen_postprocess_past_key_value(past_key_values):
    # (layer_num, bs, prefixlen, head_num/tensor_parallel_degree, head_dim)*2
    keys, values = paddle.transpose(past_key_values, perm=[2, 0, 1, 3, 4]).split(2)

    return tuple(zip(keys, values))
