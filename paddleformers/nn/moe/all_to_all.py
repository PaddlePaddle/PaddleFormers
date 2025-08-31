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
import paddle.distributed as dist
from paddle.autograd import PyLayer
from paddle.distributed.communication import stream

from .utils import manual_backward


class AlltoAll(PyLayer):
    """
    Custom PyLayer for All-to-All communication with backward pass.
    """

    @staticmethod
    def forward(ctx, x, group, sync_op=True):
        """
        Perform All-to-All communication in the group.

        Args:
            x: Input tensor
            group: Communication group
            sync_op: Whether to perform synchronous operation

        Returns:
            Tensor: Output tensor
        """
        ctx.group = group
        if dist.get_world_size(group) <= 1:
            return x
        output = paddle.empty_like(x)
        output.stop_gradient = False
        task = stream.alltoall_single(output, x, None, None, group, sync_op=sync_op, use_calc_stream=sync_op)
        if not sync_op:
            return output, task
        else:
            return output

    @staticmethod
    def backward(ctx, *dx):
        """
        Backward pass for All-to-All communication.

        Args:
            dx: Gradient tensor

        Returns:
            Tensor: Gradient after backward All-to-All
        """
        return AlltoAll.apply(*dx, group=ctx.group)


class AlltoAllAsync(PyLayer):
    """
    Custom PyLayer for asynchronous All-to-All communication.
    """

    @staticmethod
    def forward(ctx, x, *fn_args, group=None, fn=None, is_first_fwd=False):
        """
        Asynchronous All-to-All communication with function execution.

        Args:
            x: Input tensor
            fn_args: Arguments for the function
            group: Communication group
            fn: Function to execute
            is_first_fwd: Whether this is the first forward pass

        Returns:
            tuple: (output tensor, function outputs)
        """
        assert fn is not None, "use AlltoAll no async"
        ctx.group = group
        if dist.get_world_size(group) <= 1:
            ctx.bwf, fn_out = manual_backward(fn, is_first_fwd, *fn_args)
            return (x,) + fn_out
        x_out = paddle.empty_like(x)
        x_out.stop_gradient = False
        task = stream.alltoall_single(
            x_out,
            x,
            None,
            None,
            group,
            sync_op=False,
        )
        ctx.bwf, fn_out = manual_backward(fn, is_first_fwd, *fn_args)
        task.wait()
        return (x_out,) + fn_out

    @staticmethod
    def backward(ctx, dx_out, *fn_out_grads):
        """
        Backward pass for asynchronous All-to-All.

        Args:
            dx_out: Gradient of output
            fn_out_grads: Gradients of function outputs

        Returns:
            tuple: (gradient tensor, function argument gradients)
        """
        if dist.get_world_size(ctx.group) <= 1:
            fn_args_grads = ctx.bwf(*fn_out_grads)
            return (dx_out,) + fn_args_grads

        dx = paddle.empty_like(dx_out)
        dx.stop_gradient = False
        task = stream.alltoall_single(
            dx,
            dx_out,
            None,
            None,
            ctx.group,
            sync_op=False,
        )
        fn_args_grads = ctx.bwf(*fn_out_grads)
        task.wait()
        return (dx,) + fn_args_grads
