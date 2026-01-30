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

import logging

import numpy as np
import paddle
from paddle import distributed as dist
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    fused_allreduce_gradients_with_group,
)
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
    build_sharded_state_dict,
)
from paddle.incubate.tensor.manipulation import create_async_load
from paddle.nn import functional as F
from paddle.nn.layer.layers import Layer

from paddleformers.cli.train.ernie_pretrain.models.comm_utils import (
    all_gather,
    reduce_scatter,
    scatter,
)

try:
    from paddle.nn.functional import all_gather_gemm, flux, gemm_reduce_scatter
except ImportError:
    gemm_reduce_scatter = None
    all_gather_gemm = None
    flux = None

logger = logging.getLogger(__name__)


def get_hcg():
    return fleet.get_hybrid_communicate_group()


async_loader = None


def get_async_loader():
    global async_loader
    if not hasattr(fleet.fleet, "_hcg"):
        if async_loader is None:
            async_loader = create_async_load()
        return async_loader

    hcg = get_hcg()
    if not hasattr(hcg, "async_loader"):
        hcg.async_loader = create_async_load()
    return hcg.async_loader


def hack_offload_wait(task):
    task.cpu_wait()


def hack_reload_wait(task):
    task.cuda_wait()


class ScatterOp(PyLayer):
    @staticmethod
    def forward(ctx, input, axis=0, group=None):
        ctx.axis = axis
        ctx.group = group
        return scatter(input, axis=axis, group=ctx.group)

    @staticmethod
    def backward(ctx, grad):
        return all_gather(grad, axis=ctx.axis, group=ctx.group)


class GatherOp(PyLayer):
    @staticmethod
    def forward(ctx, input, axis=0, group=None):
        ctx.axis = axis
        ctx.group = group
        return all_gather(input, axis=axis, group=group)

    @staticmethod
    def backward(ctx, grad):
        return scatter(grad, axis=ctx.axis, group=ctx.group)


class AllGatherOp(PyLayer):
    @staticmethod
    def forward(ctx, input, group=None):
        ctx.group = group
        return all_gather(input, group=group)

    @staticmethod
    def backward(ctx, grad):
        return reduce_scatter(grad, group=ctx.group)


class ReduceScatterOp(PyLayer):
    @staticmethod
    def forward(ctx, input, group=None):

        ctx.group = group
        return reduce_scatter(input, group=group)

    @staticmethod
    def backward(ctx, grad):
        return all_gather(grad, group=ctx.group)


class AllGatherVarlenOp(PyLayer):
    @staticmethod
    def forward(ctx, input, group=None):
        hcg = fleet.get_hybrid_communicate_group()
        if group is None:
            group = hcg.get_model_parallel_group()

        shape0 = paddle.to_tensor([input.shape[0]])
        shape0_all = paddle.empty(shape=[group.nranks], dtype=shape0.dtype)
        dist.stream.all_gather(shape0_all, shape0, group=group, use_calc_stream=True)
        shape0_all = shape0_all.numpy()
        max_shape0 = shape0_all.max()

        indices = []
        for idx, s in enumerate(shape0_all):
            offset = idx * max_shape0
            indices.append(list(range(offset, offset + s)))
        indices = np.concatenate(indices, axis=0)
        indices = indices.reshape([-1] + [1] * (len(input.shape) - 1))
        indices = paddle.to_tensor(indices, dtype=paddle.int32)

        padding = max_shape0 - input.shape[0]

        ctx.shape0 = input.shape[0]
        ctx.max_shape0 = max_shape0
        ctx.shape0_all = shape0_all
        ctx.padding = padding
        ctx.indices = indices
        ctx.group = group

        if padding > 0:
            input_shape = input.shape
            input_shape[0] = padding
            padding_tensor = paddle.empty(shape=input_shape, dtype=input.dtype)
            input = paddle.concat([input, padding_tensor], axis=0)
        output = all_gather(input, group)
        output = paddle.take_along_axis(output, indices, axis=0)

        return output

    @staticmethod
    def backward(ctx, grad):
        input_shape = grad.shape
        input_shape[0] = ctx.max_shape0 * ctx.shape0_all.shape[0]
        output = paddle.zeros(shape=input_shape, dtype=grad.dtype)

        grad = paddle.scatter(output, ctx.indices, grad)

        grad = scatter(grad, ctx.group)

        if ctx.padding > 0:
            grad = grad[: ctx.shape0]
        return grad


class GemmReduceScatterOp(PyLayer):
    @staticmethod
    def forward(ctx, input, weight, group):
        ctx.save_for_backward(input, weight)
        ctx.group = group
        output = gemm_reduce_scatter(input, weight, group)
        return output

    @staticmethod
    def backward(ctx, grad):
        input, weight = ctx.saved_tensor()
        group = ctx.group
        if input.stop_gradient and weight.stop_gradient:
            return None, None

        if input.stop_gradient:
            input_grad = None
            grad_parallel = None
        else:
            input_grad, grad_parallel = all_gather_gemm(grad, weight, group, deepcopy_input_parallel=False)

        if weight.stop_gradient:
            weight_grad = None
        else:
            if grad_parallel is None:
                grad_parallel = all_gather(grad)
            weight_grad = paddle.matmul(input, grad_parallel, transpose_x=True)
        return input_grad, weight_grad


class AllGatherGemmOp(PyLayer):
    @staticmethod
    def forward(ctx, input, weight, group):
        output, input_parallel = all_gather_gemm(input, weight, group, deepcopy_input_parallel=True)
        ctx.save_for_backward(input_parallel, weight)
        ctx.group = group
        ctx.input_stop_gradient = input.stop_gradient
        return output

    @staticmethod
    def backward(ctx, grad):
        input_parallel, weight = ctx.saved_tensor()
        group = ctx.group
        if ctx.input_stop_gradient and weight.stop_gradient:
            return None, None
        if ctx.input_stop_gradient:
            input_grad = None
        else:
            input_grad = gemm_reduce_scatter(grad, weight, group)
        if weight.stop_gradient:
            weight_grad = None
        else:
            weight_grad = paddle.matmul(input_parallel, grad, transpose_x=True)

        return input_grad, weight_grad


def sequence_parallel_sparse_mask_labels(labels, ignore_label=-100):
    hcg = fleet.get_hybrid_communicate_group()
    group = hcg.get_model_parallel_group()
    labels = labels.flatten()
    labels_local = paddle.split(labels, group.nranks)[group.rank]

    tgt_index = paddle.nonzero(labels_local != ignore_label).squeeze()
    if tgt_index.numel() == 0:
        tgt_index = paddle.to_tensor([0])

    tgt_index = tgt_index.reshape([-1]).astype(paddle.int32)
    labels_local_gather = paddle.take_along_axis(labels_local, tgt_index, axis=0)
    labels_all_gather = AllGatherVarlenOp.apply(labels_local_gather)
    return labels_all_gather, tgt_index.reshape([-1, 1])


def mark_as_sequence_parallel_parameter(parameter):
    parameter.sequence_parallel = True


def is_sequence_parallel_parameter(parameter):
    return getattr(parameter, "sequence_parallel", False)


def create_fused_allreduce_gradient_hook(parameter_list, accumulation_steps):
    hcg = get_hcg()
    group = hcg.get_model_parallel_group()

    step = [0]
    accumulation_steps *= len(parameter_list)

    def __impl__(grad):
        step[0] += 1
        if step[0] == accumulation_steps:
            step[0] = 0
            fused_allreduce_gradients_with_group(parameter_list, group=group, scale=1.0)
        return grad

    return __impl__


def create_non_fused_allreduce_gradient_hook(param, model, verbose=False):
    hcg = get_hcg()
    pg = hcg.get_model_parallel_group().process_group
    step = [0]

    @paddle.autograd.no_grad()
    def __impl__():
        step[0] += 1
        accumulation_steps = model.accumulate_steps
        if verbose:
            logger.info(
                f'hook called: acc-step={step[0]}/{accumulation_steps}, use_main_grad={hasattr(param, "main_grad")}'
            )
        if (step[0] % accumulation_steps) == 0:
            step[0] = 0
            if hasattr(param, "main_grad"):
                pg.allreduce(param.main_grad).wait()
            else:
                pg.allreduce(param.grad).wait()

    return __impl__


def register_sequence_parallel_allreduce_hooks(model, fuse_sequence_parallel_allreduce=False):
    logger.warning("DO NOT use sphook unless your PyLayer does not trigger param backward hook")
    mp_group = get_hcg().get_model_parallel_group()
    if mp_group.nranks <= 1:
        return

    params = []
    for n, p in model._layers.named_parameters():
        if is_sequence_parallel_parameter(p):
            logger.info(f"register bw hook for:{n}")
            params.append(p)
    logger.info(f"#-sp-sync param:{len(params)}")

    if fuse_sequence_parallel_allreduce:
        raise NotImplementedError
    else:
        for i, p in enumerate(params):
            if p.stop_gradient:
                continue
            hook = create_non_fused_allreduce_gradient_hook(p, model, verbose=False)
            p._register_backward_hook(hook)


def is_fused_matmul_bias_supported():
    if paddle.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm():
        try:
            from paddle.base import core
        except ModuleNotFoundError:
            logger.warning("Unable to import paddle.base, are you using paddle latest build?")
            try:
                from paddle.fluid import core
            except ModuleNotFoundError:
                logger.warning("Unable to import paddle.fluid, are you using paddle latest build?")
                return False
        return hasattr(core.eager.ops.legacy, "fused_gemm_epilogue")
    else:
        return False


class ColumnSequenceParallelLinear(Layer):
    def __init__(
        self,
        in_features,
        out_features,
        weight_attr=None,
        has_bias=None,
        gather_output=True,
        fuse_matmul_bias=False,
        mp_group=None,
        use_rr=False,
        name=None,
        use_comm=True,
        use_tpsp_comm_overlap=False,
    ):
        super(ColumnSequenceParallelLinear, self).__init__()

        hcg = get_hcg()
        self.model_parallel_group = hcg.get_model_parallel_group() if mp_group is None else mp_group
        self.world_size = hcg.get_model_parallel_group().nranks if mp_group is None else mp_group.nranks
        self._name = name
        self.is_mp = self.world_size > 1
        self.use_comm = use_comm
        if not self.use_comm:
            assert not use_rr, "The moe allgather not compatibale with rr for now."

        self.use_tpsp_comm_overlap = use_tpsp_comm_overlap
        if self.use_tpsp_comm_overlap:
            assert all_gather_gemm is not None
            assert flux is not None

        assert (
            gather_output is False
        ), "If sequence_parallel is True, \
                                        gather_output is False"

        self.gather_output = gather_output
        assert out_features % self.world_size == 0, (
            f"Number of column of the weight for linear ({out_features}) must be"
            f" divisible by model parallel size ({self.world_size})"
        )
        self.output_size_per_partition = out_features // self.world_size

        self._weight_attr = weight_attr
        self._dtype = self._helper.get_default_dtype()

        if self.is_mp and paddle.in_dynamic_mode():
            with get_rng_state_tracker().rng_state():
                self.weight = self.create_parameter(
                    shape=[in_features, self.output_size_per_partition],
                    attr=self._weight_attr,
                    dtype=self._dtype,
                    is_bias=False,
                )
        else:
            self.weight = self.create_parameter(
                shape=[in_features, self.output_size_per_partition],
                attr=self._weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )

        self.weight.is_distributed = True if self.is_mp else False
        if self.weight.is_distributed:
            self.weight.split_axis = 1

        if has_bias:
            self.bias = self.create_parameter(
                shape=[self.output_size_per_partition],
                attr=paddle.nn.initializer.Constant(value=0.0),
                dtype=self._dtype,
                is_bias=True,
            )
            self.bias.is_distributed = True if self.is_mp else False
            if self.bias.is_distributed:
                self.bias.split_axis = 0
        else:
            self.bias = None

        self.linear = F.linear

        if fuse_matmul_bias:
            if not is_fused_matmul_bias_supported():
                raise NotImplementedError(
                    "You set fuse_matmul_bias=True in ColumnSequenceParallelLinear, "
                    "however, the paddle you are using not support this operation. "
                    "Please set fuse_matmul_bias=False or use paddle compiled "
                    "with cuda 11.6 or higher."
                )
            from paddle.incubate.nn.functional import fused_linear

            self.linear = fused_linear

    def forward(self, x, use_comm=True):
        if (
            self.use_tpsp_comm_overlap
            and self.is_mp
            and (use_comm and self.use_comm)
            and flux.all_gather_gemm_can_implement(x, self.weight, self.model_parallel_group)
        ):
            output = AllGatherGemmOp.apply(x, self.weight, self.model_parallel_group)
            if self.bias is not None:
                output += self.bias
            return output
        else:
            if self.is_mp and (use_comm and self.use_comm):
                input_parallel = AllGatherOp.apply(x)
            else:
                input_parallel = x

            output = self.linear(input_parallel, self.weight, self.bias)
            return output

    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        state_dict = self.state_dict(structured_name_prefix="")
        return build_sharded_state_dict(state_dict, {"weight": 1}, structured_name_prefix)


class MPScale(PyLayer):
    @staticmethod
    def forward(ctx, x, mp_degree):
        out = paddle.scale(x, 1.0 / mp_degree)
        return out

    @staticmethod
    def backward(ctx, dout):
        return dout


class RowSequenceParallelLinear(Layer):
    def __init__(
        self,
        in_features,
        out_features,
        weight_attr=None,
        has_bias=True,
        input_is_parallel=False,
        fuse_matmul_bias=False,
        use_rr=False,
        mp_group=None,
        name=None,
        use_comm=True,
        use_tpsp_comm_overlap=False,
    ):
        super(RowSequenceParallelLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        assert (
            input_is_parallel is True
        ), "If sequence_parallel is True, \
                                           input_is_parallel should be true."

        self.input_is_parallel = input_is_parallel
        self._weight_attr = weight_attr
        self._dtype = self._helper.get_default_dtype()
        self._name = name
        self.use_comm = use_comm
        if not self.use_comm:
            assert not use_rr, "The moe allgather not compatibale with rr for now."

        self.use_tpsp_comm_overlap = use_tpsp_comm_overlap
        if self.use_tpsp_comm_overlap:
            assert gemm_reduce_scatter is not None
            assert flux is not None

        hcg = get_hcg()
        self.model_parallel_group = hcg.get_model_parallel_group() if mp_group is None else mp_group
        self.world_size = hcg.get_model_parallel_group().nranks if mp_group is None else mp_group.nranks
        self.rank = hcg.get_model_parallel_group().rank if mp_group is None else mp_group.rank

        self.is_mp = self.world_size > 1
        assert in_features % self.world_size == 0, (
            f"Number of row of the weight for linear ({in_features}) must be"
            f" divisible by model parallel size ({self.world_size})"
        )

        self.input_size_per_partition = in_features // self.world_size

        if self.is_mp and paddle.in_dynamic_mode():
            with get_rng_state_tracker().rng_state():
                self.weight = self.create_parameter(
                    shape=[self.input_size_per_partition, self.out_features],
                    attr=self._weight_attr,
                    dtype=self._dtype,
                    is_bias=False,
                )
        else:
            self.weight = self.create_parameter(
                shape=[self.input_size_per_partition, self.out_features],
                attr=self._weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )

        self.weight.is_distributed = True if self.is_mp else False
        if self.weight.is_distributed:
            self.weight.split_axis = 0

        if has_bias:
            self.bias = self.create_parameter(
                shape=[self.out_features],
                attr=paddle.nn.initializer.Constant(value=0.0),
                dtype=self._dtype,
                is_bias=True,
            )
            if self.is_mp:
                mark_as_sequence_parallel_parameter(self.bias)
        else:
            self.bias = None

        self.linear = F.linear
        self.mp_scale = None

        if fuse_matmul_bias:
            if not is_fused_matmul_bias_supported():
                raise NotImplementedError(
                    "You set fuse_matmul_bias=True in RowParallelLinear, "
                    "however, the paddle you are using not support this operation. "
                    "Please set fuse_matmul_bias=False or use paddle compiled "
                    "with cuda 11.6 or higher."
                )
            from paddle.incubate.nn.functional import fused_linear

            self.linear = fused_linear

    def forward(self, x):
        input_parallel = x
        if self.is_mp:
            if self.mp_scale is not None:
                bias = self.mp_scale(self.bias, self.world_size)
            else:
                bias = None

            if (
                self.use_tpsp_comm_overlap
                and self.use_comm
                and flux.gemm_reduce_scatter_can_implement(x, self.weight, self.model_parallel_group)
            ):
                output_ = GemmReduceScatterOp.apply(x, self.weight, self.model_parallel_group)
                if bias is not None:
                    output_ = output_ + bias
            else:
                output_parallel = self.linear(input_parallel, self.weight, bias)
                if self.use_comm:
                    output_ = ReduceScatterOp.apply(output_parallel)
                else:
                    output_ = output_parallel

            if bias is None and self.bias is not None and self.use_comm:
                output = output_ + self.bias
            else:
                output = output_
        else:
            output = self.linear(input_parallel, self.weight, self.bias)
        return output

    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        state_dict = self.state_dict(structured_name_prefix="")
        return build_sharded_state_dict(state_dict, {"weight": 0}, structured_name_prefix)
