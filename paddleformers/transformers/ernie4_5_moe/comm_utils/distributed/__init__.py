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

"""
Distributed utils.
"""
import paddle

# Import / Override modules for specific devices
if paddle.is_compiled_with_xpu():
    from .common_dist_utils import (
        AllGatherVarlenOp,
        RRColumnSequenceParallelLinear,
        RRRowSequenceParallelLinear,
        get_hcg,
        mark_as_sequence_parallel_parameter,
        sequence_parallel_sparse_mask_labels,
    )
    from .xpu_dist_utils import (
        ColumnParallelLinear,
        ColumnSequenceParallelLinear,
        GatherOp,
        RowParallelLinear,
        RowSequenceParallelLinear,
        ScatterOp,
    )
else:
    from .common_dist_utils import (
        AllGatherVarlenOp,
        ColumnParallelLinear,
        ColumnSequenceParallelLinear,
        GatherOp,
        RowParallelLinear,
        RowSequenceParallelLinear,
        RRColumnSequenceParallelLinear,
        RRRowSequenceParallelLinear,
        ScatterOp,
        get_hcg,
        mark_as_sequence_parallel_parameter,
        sequence_parallel_sparse_mask_labels,
    )

__all__ = [
    "ColumnParallelLinear",
    "ColumnSequenceParallelLinear",
    "RowParallelLinear",
    "RowSequenceParallelLinear",
    "GatherOp",
    "ScatterOp",
    "mark_as_sequence_parallel_parameter",
    "ParallelCrossEntropy",
    "get_rng_state_tracker",
    "parallel_matmul",
    "RRColumnSequenceParallelLinear",
    "RRRowSequenceParallelLinear",
    "AllGatherVarlenOp",
    "sequence_parallel_sparse_mask_labels",
    "get_hcg",
]


def parallel_matmul(
    x,
    y,
    bias=None,
    transpose_y=False,
    tensor_parallel_degree=1,
    tensor_parallel_output=True,
    fuse_linear=False,
    training=None,
):
    """
    Parallel matmul wrapper.

    Args:
        x (Tensor): Input tensor.
        y (Tensor): Weight tensor.
        bias (Tensor, optional): Bias tensor. Default is None.
        transpose_y (bool, optional): Whether to transpose y. Default is False.
        tensor_parallel_degree (int, optional): Tensor parallel degree. Default is 1.
        tensor_parallel_output (bool, optional): Whether to output tensor parallel. Default is True.
        fuse_linear (bool, optional): Whether to fuse linear. Default is False.
        training (bool, optional): Training state. Default is None.
    Returns:
        Tensor: Output tensor.
    """
    if paddle.is_compiled_with_xpu():
        from .common_dist_utils import _parallel_matmul as default_parallel_matmul
        from .xpu_dist_utils import parallel_matmul as xpu_parallel_matmul

        if xpu_parallel_matmul is not None:
            return xpu_parallel_matmul()(
                x,
                y,
                bias=bias,
                transpose_y=transpose_y,
                tensor_parallel_degree=tensor_parallel_degree,
                tensor_parallel_output=tensor_parallel_output,
                fused_linear=fuse_linear,
            )
        else:
            return default_parallel_matmul(
                x,
                y,
                bias=bias,
                transpose_y=transpose_y,
                tensor_parallel_degree=tensor_parallel_degree,
                tensor_parallel_output=tensor_parallel_output,
                fuse_linear=fuse_linear,
            )
    else:
        from .common_dist_utils import _parallel_matmul

    return _parallel_matmul(
        x,
        y,
        bias=bias,
        transpose_y=transpose_y,
        tensor_parallel_degree=tensor_parallel_degree,
        tensor_parallel_output=tensor_parallel_output,
        fuse_linear=fuse_linear,
    )
