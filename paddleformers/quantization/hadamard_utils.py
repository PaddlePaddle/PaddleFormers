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

from ..utils import infohub


def matmul_hadU(X):

    input = X.clone().reshape((-1, X.shape[-1], 1))
    output = input.clone()
    while input.shape[1] > 1:
        input = input.reshape((input.shape[0], input.shape[1] // 2, 2, input.shape[2]))
        output = output.reshape(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.reshape((input.shape[0], input.shape[1], -1))
        (input, output) = (output, input)
    del output

    return input.reshape(X.shape)


def create_hadamard_matrix(block_size, dtype):
    Q = paddle.diag(paddle.ones((block_size), dtype=dtype))
    block = matmul_hadU(Q)
    return block


def hadamard_matmul(input, side, hadamard_matrix, block_size):
    # left -> H.T@input right -> input@H
    origin_shape = input.shape
    input = input.reshape([-1, origin_shape[-1]])
    if side == "left":
        # H.T@input -> (input.T@H).T
        input = input.transpose([1, 0])
    block_num = input.shape[-1] // block_size
    output = input.reshape([-1, block_num, block_size]) @ hadamard_matrix
    output = output.reshape([-1, block_num * block_size])
    if side == "left":
        output = output.transpose([1, 0])
    output = output.reshape(origin_shape)

    return output


def apply_hadamard_matmul(x, side, block_size):
    if getattr(infohub, "hadamard") is None:
        setattr(infohub, "hadamard", {})

    if block_size in infohub.hadamard:
        hadamard_matrix = infohub.hadamard[block_size]
    else:
        hadamard_matrix = create_hadamard_matrix(block_size, x.dtype)
        infohub.hadamard[block_size] = hadamard_matrix
    target_x = hadamard_matmul(x, side, hadamard_matrix, block_size)
    return target_x
