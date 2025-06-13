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

import os

import paddle

try:
    import paddle_custom_device
except ImportError:
    pass

from paddle import distributed as dist
from paddle.autograd import PyLayer

try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        ColumnSequenceParallelLinear,
        RowSequenceParallelLinear,
    )
except:
    pass
from ..utils.tools import get_env_device

__all_gather_recomputation__ = False
if int(os.getenv("FLAGS_NPU_MC2_Recompute", 0)):
    __all_gather_recomputation__ = True


def is_mc2_valid():
    current_device = get_env_device()
    if current_device == "npu":
        return int(os.getenv("FLAGS_NPU_MC2", 0))
    return 0


if is_mc2_valid():

    class MC2ColumnParallelCoreLinear(PyLayer):
        @staticmethod
        def forward(ctx, input_, weight, group):
            ctx.save_for_backward(input_, weight)
            ctx.group = group
            input_mp = input_
            result_mp = paddle.matmul(input_mp, weight)
            return result_mp

        @staticmethod
        def backward(ctx, dy):
            input_, weight = ctx.saved_tensor()
            sub_grad = dy.reshape([-1, dy.shape[-1]])
            rank = paddle.distributed.get_rank()
            hcom_name = ctx.group.process_group.get_comm_name(rank)

            d_weight = (
                paddle.matmul(input_.reshape([-1, input_.shape[-1]]), sub_grad, transpose_x=True)
                if not weight.stop_gradient
                else None
            )
            d_input = paddle_custom_device.npu.fused_mm_allreduce(
                sub_grad, weight.t(), bias=None, hcom=hcom_name, reduce_op="sum", comm_turn=0
            )

            if d_weight is not None:
                return d_input.reshape(input_.shape), d_weight
            else:
                return d_input.reshape(input_.shape), None

    class MC2RowParallelCoreLinear(PyLayer):
        @staticmethod
        def forward(ctx, input_, weight, group):
            ctx.save_for_backward(input_, weight)
            rank = paddle.distributed.get_rank()
            hcom_name = group.process_group.get_comm_name(rank)
            x = input_.reshape([-1, input_.shape[-1]])
            out = paddle_custom_device.npu.fused_mm_allreduce(
                x, weight, bias=None, hcom=hcom_name, reduce_op="sum", comm_turn=0
            )
            output = out.reshape([input_.shape[0], input_.shape[1], weight.shape[1]])
            ctx.ring_id = group.id
            return output

        @staticmethod
        def backward(ctx, dy):
            input_, weight = ctx.saved_tensor()
            out_grad = dy
            sub_grad = out_grad.reshape([-1, out_grad.shape[-1]])
            input_grad = paddle.matmul(sub_grad, weight, transpose_y=True)
            if weight.stop_gradient:
                return input_grad.reshape(input_.shape), None
            else:
                input_reshape = input_.reshape([-1, input_.shape[-1]])
                weight_grad = paddle.matmul(input_reshape, sub_grad, transpose_x=True)
                return input_grad.reshape(input_.shape), weight_grad

    class MC2ColumnSeqParallelCoreLinear(PyLayer):
        @staticmethod
        def forward(ctx, input_, weight, group):
            ctx.weight_stop_gradient = weight.stop_gradient
            ctx.save_for_backward(input_, weight)

            rank = dist.get_rank()
            hcomm_info = group.process_group.get_comm_name(rank)

            world_size = group.nranks
            output, gather_out = paddle_custom_device.npu.fused_allgather_mm(
                input_,
                weight,
                bias=None,
                hcom=hcomm_info,
                world_size=world_size,
                gather_index=0,
                gather_output=(not __all_gather_recomputation__),
                comm_turn=0,
            )

            ctx.all_gather_output = gather_out
            ctx.world_size = world_size
            ctx.group = group
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input_, weight = ctx.saved_tensor()

            if __all_gather_recomputation__:
                dim_size = input_.shape
                dim_size[0] = dim_size[0] * ctx.world_size
                all_gather_output = paddle.empty(dim_size, dtype=input_.dtype)
                all_gather_output.stop_gradient = True
                all_gather_work = dist.stream.all_gather(all_gather_output, input_, group=ctx.group, sync_op=False)
            else:
                all_gather_output = ctx.all_gather_output

            grad_input = paddle.matmul(grad_output, weight, transpose_y=True)
            sub_grad_input = paddle.empty(input_.shape, dtype=input_.dtype)
            reduce_scatter_work = dist.stream.reduce_scatter(
                sub_grad_input, grad_input, group=ctx.group, sync_op=False
            )

            if __all_gather_recomputation__:
                all_gather_work.wait()

            grad_weight = (
                paddle.matmul(all_gather_output, grad_output, transpose_x=True)
                if not ctx.weight_stop_gradient
                else None
            )
            reduce_scatter_work.wait()

            return sub_grad_input, grad_weight

    class MC2RowSeqParallelCoreLinear(PyLayer):
        @staticmethod
        def forward(ctx, input_, weight, group):
            ctx.weight_stop_gradient = weight.stop_gradient
            ctx.save_for_backward(input_, weight)

            rank = dist.get_rank()
            hcomm_info = group.process_group.get_comm_name(rank)
            world_size = group.nranks

            output = paddle_custom_device.npu.fused_mm_reduce_scatter(
                input_,
                weight,
                bias=None,
                hcom=hcomm_info,
                world_size=world_size,
                reduce_op="sum",
                comm_turn=0,
            )

            ctx.hcomm_info = hcomm_info
            ctx.world_size = world_size
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input_, weight = ctx.saved_tensor()
            hcomm_info = ctx.hcomm_info
            world_size = ctx.world_size

            grad_input, all_gather_grad_output = paddle_custom_device.npu.fused_allgather_mm(
                grad_output,
                weight.t(),
                bias=None,
                hcom=hcomm_info,
                world_size=world_size,
                gather_index=0,
                gather_output=True,
                comm_turn=0,
            )
            grad_weight = (
                paddle.matmul(input_, all_gather_grad_output, transpose_x=True)
                if not ctx.weight_stop_gradient
                else None
            )

            return grad_input, grad_weight

    class MC2ColumnSeqParallelLinear(ColumnSequenceParallelLinear):
        def forward(self, x):
            output = MC2ColumnSeqParallelCoreLinear.apply(x, self.weight, self.model_parallel_group)
            output = output + self.bias if self.bias is not None else output
            return output

    class MC2RowSeqParallelLinear(RowSequenceParallelLinear):
        def forward(self, x):
            output = MC2RowSeqParallelCoreLinear.apply(x, self.weight, self.model_parallel_group)
            output = output + self.bias if self.bias is not None else output
            return output

else:
    MC2ColumnSeqParallelCoreLinear = None
    MC2RowSeqParallelCoreLinear = None
    MC2ColumnSeqParallelLinear = None
    MC2RowSeqParallelLinear = None
    MC2ColumnParallelCoreLinear = None
    MC2RowParallelCoreLinear = None
