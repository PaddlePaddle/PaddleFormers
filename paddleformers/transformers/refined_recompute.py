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

from __future__ import annotations

import contextlib
import copy
import inspect
import queue
import random
import uuid
import weakref

import numpy as np
import paddle
import paddle.autograd
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel.parallel_layers.random import (
    get_rng_state_tracker,
)
from paddle.distributed.fleet.recompute.recompute import check_recompute_necessary
from paddle.distributed.fleet.recompute.recompute import recompute as original_recompute
from paddle.distributed.fleet.recompute.recompute import switch_rng_state_tracker

try:
    from paddle.distributed.fleet.utils import sequence_parallel_utils
except ImportError:
    sequence_parallel_utils = None
from paddle.distributed.fleet.layers.mpu import mp_layers, mp_ops

from .linear_utils import (
    ColumnParallelLinear,
    ColumnSequenceParallelLinear,
    RowParallelLinear,
    RowSequenceParallelLinear,
)

try:
    from paddle.base import core, framework
except ImportError:
    from paddle.fluid import core, framework

__all__ = [
    "NoRecomputeContext",
    "no_recompute",
    "recompute",
    "get_global_rr_queue_dict",
    "get_skip_recompute_ops",
    "RRColumnSequenceParallelLinear",
    "RRRowSequenceParallelLinear",
    "RRColumnParallelLinear",
    "RRRowParallelLinear",
]
_in_no_recompute = False
global_rr_queue_dict = {}
recompute_suffix = "@recompute"
_recompute_id = -1

# https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_7th/%E3%80%90Hackathon%207th%E3%80%91FundableProject%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#%E4%B9%9Dpaddle-lod-%E9%80%80%E5%9C%BA%E6%B8%85%E7%90%86
if hasattr(core.VarDesc.VarType, "DENSE_TENSOR"):
    DENSE_TENSOR = core.VarDesc.VarType.DENSE_TENSOR
else:
    DENSE_TENSOR = core.VarDesc.VarType.LOD_TENSOR


def set_recompute_id(value=-1):
    """switch recompute id to the given value"""
    global _recompute_id
    _recompute_id = str(value)


def get_recompute_id():
    """get current recompute id"""
    global _recompute_id
    return str(_recompute_id)


@contextlib.contextmanager
def switch_recompute_id_ctx(value=-1):
    """switch recompute id to the given value within the context"""
    raw_recompute_id = get_recompute_id()
    set_recompute_id(value)
    yield
    set_recompute_id(raw_recompute_id)


def in_no_recompute_ctx():
    """check if in no recompute context"""
    global _in_no_recompute
    return _in_no_recompute


def set_no_recompute(value=True):
    """set whether in no recompute mode"""
    global _in_no_recompute
    _in_no_recompute = value


@contextlib.contextmanager
def switch_recompute_ctx(kwargs):
    """switch recompute context to the given value within the context"""
    for ts in kwargs.values():
        if paddle.is_tensor(ts) and not ts.name.endswith(recompute_suffix):
            # 1. add recompute suffix to the tensor name
            ts.name = ts.name + recompute_suffix
    # 2. set in no recompute mode
    set_no_recompute(True)
    yield
    for ts in kwargs.values():
        if paddle.is_tensor(ts) and ts.name.endswith(recompute_suffix):
            # 3. remove recompute suffix from the tensor name
            ts.name = ts.name[: -len(recompute_suffix)]
    # 4. reset in no recompute mode
    set_no_recompute(False)


def get_global_rr_queue_dict():
    """get global rr queue dict"""
    global global_rr_queue_dict
    return global_rr_queue_dict


# def print_global_rr_queue_info(name="pack"):
#     queue_dict = get_global_rr_queue_dict()
#     print("{:<10} {:<20} {:<10}".format("Action", "Queue Name", "Queue Size"))
#     print("-" * 50)
#     for k, v in queue_dict.items():
#         print("{:<10} {:<20} {:<10}".format(name, k, v.qsize()))
#     print("=" * 50)


def parse_to_kwargs(function, *args, **kwargs):
    """Parse the function arguments into a dictionary."""
    signature = inspect.signature(function)
    bound_arguments = signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    return bound_arguments.arguments


class _NoopSaveInputs(paddle.autograd.PyLayer):
    """
    This layer does nothing but save all input tensors.
    This is used to prevent the gradients of the inputs being computed.
    """

    @staticmethod
    def forward(ctx, *args):
        """This function does nothing but save all input tensors."""
        tensors = [o for o in args if isinstance(o, paddle.Tensor)]
        ctx.save_for_backward(*tensors)
        return paddle.empty((0,), dtype=tensors[0].dtype)

    @staticmethod
    def backward(ctx, *args):
        """Should not be called since we don't support backward on this graph."""
        raise AssertionError("Did not expect to backward on this graph")


def no_recompute(function, *args, **kwargs):
    """
    Within a recompute context, do not recompute intermediate activations.

    Parameters:
        function (paddle.nn.Layer): The layer or sequence of layers that describe a part of the model's
                                   forward pass, whose intermediate activations will not be released.
        *args (Tensor): Input tensors to the function.
        **kwargs (Dict): Keyword arguments to the function.

    Returns:
        The output of the function given the input tensors and keyword arguments.
    """
    recompute_id_with_suffix = get_recompute_id()
    # enable kwargs, in no recompute context, has grad
    enable = kwargs.pop("enable", True) and recompute_id_with_suffix != "-1" and framework._dygraph_tracer()._has_grad
    keys_ignore_to_save = kwargs.pop("keys_ignore_to_save", [])
    if not enable:
        return function(*args, **kwargs)

    if isinstance(function, paddle.nn.Layer):
        func = function.forward
        input_kwargs = parse_to_kwargs(func, *args, **kwargs)
    elif isinstance(function, paddle.autograd.PyLayer):
        func = function.apply
        input_kwargs = parse_to_kwargs(function.forward, *args, **kwargs)
    else:
        func = function
        input_kwargs = parse_to_kwargs(func, *args, **kwargs)

    is_first_fwd = recompute_id_with_suffix.endswith("@first")
    recompute_id = recompute_id_with_suffix.split("@")[0]

    if is_first_fwd:
        if recompute_id not in global_rr_queue_dict:
            global_rr_queue_dict[recompute_id] = queue.Queue()

        with switch_recompute_ctx(input_kwargs):
            result = func(*args, **kwargs)

        global_rr_queue_dict[recompute_id].put(result)
    else:
        tensor_list = []
        for key, val in input_kwargs.items():
            if key in keys_ignore_to_save:
                continue
            if val is not None and paddle.is_tensor(val):
                tensor_list.append(val)

        if len(tensor_list) > 0:
            _NoopSaveInputs.apply(*tensor_list)

        result = global_rr_queue_dict[recompute_id].get()

        if global_rr_queue_dict[recompute_id].empty():
            global_rr_queue_dict.pop(recompute_id)
    return result


class NoRecomputeContext:
    """
    A Context Manager class that do not recompute intermediate activations.
    """

    def __init__(self, enable=True, keys_ignore_to_save=[]):
        """initialize the RefinedRecomputeFunction object."""
        self._enable = enable
        self._keys_ignore_to_save = keys_ignore_to_save

    def __enter__(self):
        """enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """exit the context manager."""
        pass

    def __call__(self, function, *args, **kwargs):
        """
        Within a recompute context, do not recompute intermediate activations.

        Parameters:
            function (paddle.nn.Layer): The layer or sequence of layers that describe a part of the model's
                                    forward pass, whose intermediate activations will not be released.
            *args (Tensor): Input tensors to the function.
            **kwargs (Dict): Keyword arguments to the function.

        Returns:
            The output of the function given the input tensors and keyword arguments.
        """
        kwargs["enable"] = self._enable
        kwargs["keys_ignore_to_save"] = self._keys_ignore_to_save
        return no_recompute(function, *args, **kwargs)


def share_buffer_to_tensor_or_param(inner_x):
    """share buffer to tensor or param"""
    if hasattr(inner_x, "main_grad"):
        # do not deepcopy the `main_grad` to save memory
        state = copy.deepcopy({k: v for k, v in inner_x.__dict__.items() if k != "main_grad"})
        tmp_tensor = framework.EagerParamBase(
            shape=inner_x.shape, dtype=inner_x.dtype, name=inner_x.name + "cpy", **state
        )
        setattr(tmp_tensor, "main_grad", inner_x.main_grad)
        inner_x._unsafe_share_buffer_to(tmp_tensor)
    else:
        if inner_x.is_dist():
            # TODO(jeff41404): it seems better to use `tmp_tensor = core.eager.Tensor(inner_x)`,
            # but other errors will be triggered during the current period, and can be modified after resolution
            tmp_tensor = core.eager.Tensor(
                inner_x.dtype,
                inner_x.shape,
                inner_x.name + "cpy",
                DENSE_TENSOR,
                inner_x.persistable,
                inner_x.process_mesh,
                inner_x.placements,
            )
        else:
            tmp_tensor = core.eager.Tensor(
                inner_x.dtype,
                inner_x.shape,
                inner_x.name + "cpy",
                DENSE_TENSOR,
                inner_x.persistable,
            )
        inner_x._unsafe_share_buffer_to(tmp_tensor)
        tmp_tensor.stop_gradient = inner_x.stop_gradient
    return tmp_tensor


def _recompute_without_reentrant(function, preserve_rng_state=True, *args, **kwargs):
    """
    recompute without reentrant, that means use hook to implement the recompute function rather than re-entrant autograd.
    """

    if preserve_rng_state:
        cur_device = paddle.get_device()
        if "gpu:" in cur_device:
            fw_cuda_rng_state = paddle.get_cuda_rng_state()
        elif "cpu" in cur_device:
            fw_cuda_rng_state = paddle.get_rng_state()
        elif "xpu:" in cur_device:
            fw_cuda_rng_state = paddle.get_rng_state()
        elif cur_device.split(":")[0] in paddle.device.get_all_custom_device_type():
            fw_cuda_rng_state = paddle.get_rng_state(cur_device)
        else:
            raise RuntimeError(f"Recompute with RNG preserve is not support current device: {cur_device}.")
        fwd_cuda_rng_state_tracker = get_rng_state_tracker().get_states_tracker()
        fwd_numpy_state = np.random.get_state()
        fwd_random_state = random.getstate()
    tracer = framework._dygraph_tracer()
    is_fw_autocast = False if tracer._amp_level == core.AmpLevel.O0 else True
    if tracer._amp_level == core.AmpLevel.O2:
        amp_level = "O2"
    elif tracer._amp_level in (core.AmpLevel.O1, core.AmpLevel.O0):
        amp_level = "O1"

    if tracer._amp_dtype == "float16":
        amp_dtype = "float16"
    elif tracer._amp_dtype in ("bfloat16", "float32"):
        amp_dtype = "bfloat16"

    amp_white_list, amp_black_list = tracer._get_amp_op_list()

    class IntermediateHolder:
        pass

    storage = weakref.WeakKeyDictionary()
    holder_list = []
    # generate a unique id for the recompute context
    recompute_id = str(int(uuid.uuid4()))

    def pack(x):
        # [PACK] in no recompute context or input tensor no need recompute, return the input tensor directly
        if x is not None and x.persistable or (in_no_recompute_ctx() and not x.name.endswith(recompute_suffix)):
            return share_buffer_to_tensor_or_param(x)

        # remove the recompute suffix
        res = IntermediateHolder()
        holder_list.append(weakref.ref(res))
        return res

    def unpack(x):
        # [UNPACK] in no recompute context or input tensor no need recompute, return the input tensor directly
        if paddle.is_tensor(x):
            return x

        unpack_counter = 0
        if len(storage) == 0:

            def inner_pack(inner_x):
                if inner_x is not None and inner_x.persistable:
                    return

                nonlocal unpack_counter
                unpack_counter += 1

                if unpack_counter - 1 >= len(holder_list):
                    raise Exception(
                        "Not supported to retrieve a tensor saved by autograd multiple times that is no need to recompute."
                        "Please check your `keys_ignore_to_save`."
                    )

                if holder_list[unpack_counter - 1]() is None:
                    return
                if inner_x is None:
                    storage[holder_list[unpack_counter - 1]()] = None
                    return

                storage[holder_list[unpack_counter - 1]()] = share_buffer_to_tensor_or_param(inner_x)
                return

            def inner_unpack(inner_x):
                raise Exception("An unexpected backward called on a tensor!")

            rng_cxt_manager = (
                contextlib.nullcontext()
                if not preserve_rng_state
                else switch_rng_state_tracker(
                    fw_cuda_rng_state, fwd_cuda_rng_state_tracker, fwd_numpy_state, fwd_random_state
                )
            )
            with rng_cxt_manager:
                with paddle.set_grad_enabled(True):
                    with paddle.amp.auto_cast(
                        enable=is_fw_autocast,
                        custom_white_list=amp_white_list,
                        custom_black_list=amp_black_list,
                        level=amp_level,
                        dtype=amp_dtype,
                    ):
                        with switch_recompute_id_ctx(recompute_id + "@second"):
                            with paddle.autograd.saved_tensors_hooks(inner_pack, inner_unpack):
                                function(*args, **kwargs)

        if x not in storage:
            raise Exception(
                "Not supported to retrieve a tensor saved by autograd multiple times that is no need to recompute."
            )
        return storage.pop(x)

    with switch_recompute_id_ctx(recompute_id + "@first"):
        with paddle.autograd.saved_tensors_hooks(pack, unpack):
            outputs = function(*args, **kwargs)

    return outputs


def recompute(function, *args, **kwargs):
    """
    recompute intermediate activations to save then memory.

    Parameters:
        function(paddle.nn.Layer): layer of sequence of layers that describes part of forward pass of the model
              whose intermediate activations will be released to save memory in forward stage and will be recomputed
              in backward stage for gradient calculation.
        *args(Tensor): inputs to the function.
        **kwargs(Dict): Kwargs should only contain two kinds of key-value params, the one is part of function's key-value params,
                        and the other contains 'preserve_rng_state' and 'use_reentrant'. the key-value pair of preserve_rng_state,
                        which is used to indicate whether to save the forward rng. If it is True, then the last forward rng value
                        will be restored when the forward recalculation of backpropagation is performed, its default value is True.
                        the key-value pair of use_reentrant is used to indicate which implementation of recompute you will be used.
                        'use_reentrant=True' means to use the PyLayer implementation of recompute, 'use_reentrant=False' means to
                        use the Hook implementation of recompute, its default value is True.
    Returns:
        Output of function on args.
    """
    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop("preserve_rng_state", True)

    # whether to use reentrant method to implement recompute
    use_reentrant = kwargs.pop("use_reentrant", True)

    if not paddle.in_dynamic_mode():
        from paddle.distributed.auto_parallel.interface import (
            recompute as static_auto_recompute,
        )

        return static_auto_recompute(function)(*args, **kwargs)

    if not use_reentrant:
        _ = kwargs.pop("offload_indices", [])  # currently not support offload_indices
        if framework._dygraph_tracer()._has_grad:
            check_args = list(args)
            check_args.extend(list(kwargs.values()))
            check_recompute_necessary(check_args)
        return _recompute_without_reentrant(function, preserve, *args, **kwargs)
    else:
        kwargs["preserve_rng_state"] = preserve
        kwargs["use_reentrant"] = use_reentrant
        return original_recompute(function, *args, **kwargs)


def get_pp_vp_split_layers(layer_num, pp_size, vp_size, skip_recompute_num=-1):
    """
    Get the selected layers to skip recompute.

    Args:
    - skip_recompute_num (int, optional): The number of stages to skip recompute. If not provided or is negative
      one, it means that all layers should be skipped. Default: -1.

    Returns:
    - :obj:`set`: A set containing the selected layers to skip recompute.

    """

    assert pp_size > 1, (
        "Only support pipeline parallel, " f"pp_size must be greater than 1, but got pp_size: {pp_size}"
    )

    if skip_recompute_num == -1:
        # select all layers to skip recompute
        skip_recompute_num = vp_size

    no_recompute_layer_num = []
    if skip_recompute_num == 0:
        return set(no_recompute_layer_num)

    if vp_size == 1:
        # If vp_size == 1, we can not select model chunk for pp,
        # so if skip_recompute_num > 0, we select the all layers to skip recompute.
        if skip_recompute_num > 0:
            return set(range(layer_num))
        else:
            return set()

    assert layer_num % (pp_size * vp_size) == 0, (
        "layer_num must be divisible by pp_size * vp_size,"
        f" but got layer_num: {layer_num}, pp_size: {pp_size}, vp_size: {vp_size}"
    )

    chunk_size = layer_num // (pp_size * vp_size)
    chunk_list = [list(range(i * chunk_size, (i + 1) * chunk_size)) for i in range(pp_size * vp_size)]

    stage_chunk_list = [[] for _ in range(pp_size)]
    for i in range(pp_size * vp_size):
        stage_chunk_list[i % pp_size].append(chunk_list[i])

    for i in range(pp_size):
        no_recompute_layer_num.extend(stage_chunk_list[i][-skip_recompute_num:])

    # Convert to 1D list
    return set(sum(no_recompute_layer_num, []))


def get_skip_recompute_ops(config, layer_idx):
    """
    Creates a dictionary for skipping recomputation based on the configuration file,
    effective only at the specified layer index.

    Args:
        config (dict): The configuration file of the input model.
        layer_idx (int): The layer index used to check whether recomputation should be skipped.

    Returns:
        dict: Returns an updated configuration file containing the following key-value pairs:
            - skip_recompute_ops (dict): A dictionary with each model layer's each operation's name
                                         and a boolean indicating whether to skip recomputation, defaults to None.
            - If the refined_recompute key does not exist or recompute is set to False,
              the original configuration file is returned.

    """
    skip_recompute_ops = dict()
    if not config.recompute or not isinstance(config.refined_recompute, dict):
        return skip_recompute_ops

    try:
        hcg = fleet.get_hybrid_communicate_group()
        pp_size = max(hcg.get_pipe_parallel_world_size(), 1)
    except:
        pp_size = 1
    layer_num = config.num_layers if hasattr(config, "num_layers") else config.num_hidden_layers
    if hasattr(config, "add_tail_layer") and config.add_tail_layer:
        layer_num += 1

    for op_name, skip_num in config.refined_recompute.items():
        # is pp model
        if pp_size > 1:
            vp_size = max(config.virtual_pp_degree, 1)
            no_recompute_layers = get_pp_vp_split_layers(layer_num, pp_size, vp_size, skip_num)
            if layer_idx in no_recompute_layers:
                skip_recompute_ops[op_name] = True
            else:
                skip_recompute_ops[op_name] = False
        else:
            if skip_num == 0:  # 0 means all recompute
                skip_recompute_ops[op_name] = False
            elif skip_num < 0:  # < 0 means all skip recompute
                skip_recompute_ops[op_name] = True
            else:
                if layer_idx < skip_num:  # < the number of layers to skip recompute
                    skip_recompute_ops[op_name] = True
                else:
                    skip_recompute_ops[op_name] = False
    return skip_recompute_ops


class RRColumnParallelLinear(ColumnParallelLinear):
    def forward(self, x):
        # use inner api to process identity
        def _overlap_linear():
            return mp_layers.InnerOverlapLinear.apply(
                x,
                self.weight,
                self.bias,
                self.fuse_matmul_bias,
                self.mp_async_allreduce,
                self.mp_skip_c_identity,
                self.mp_fused_linear_param_grad_add,
                self.model_parallel_group,
            )

        if self.mp_async_allreduce:
            output_parallel = _overlap_linear()
        else:
            if self.is_mp:
                input_parallel = mp_ops._c_identity(
                    x,
                    group=self.model_parallel_group,
                    skip_c_identity_dynamic=self.mp_skip_c_identity,
                )
            else:
                input_parallel = x

            def fwd(input_parallel):
                return self.linear(input_parallel, self.weight, self.bias, name=self._name)

            output_parallel = no_recompute(fwd, input_parallel)

        if self.gather_output and self.is_mp:
            output = mp_ops._c_concat(output_parallel, group=self.model_parallel_group)
        else:
            output = output_parallel
        return output


class RRRowParallelLinear(RowParallelLinear):
    def forward(self, x):
        if self.input_is_parallel or (not self.is_mp):
            input_parallel = x
        else:
            # split last dim
            input_parallel = mp_ops._c_split(x, group=self.model_parallel_group)

        if self.is_mp:
            if self.fuse_matmul_bias:
                bias = mp_layers.MPScale.apply(self.bias, self.world_size)
            else:
                bias = None

            def fwd(input_parallel):
                output_parallel = self.linear(input_parallel, self.weight, bias, name=self._name)
                output_ = mp_ops._mp_allreduce(
                    output_parallel,
                    group=self.model_parallel_group,
                    use_calc_stream=True,
                    use_model_parallel=True,
                    skip_c_identity_dynamic=self.mp_skip_c_identity,
                )
                return output_

            output_ = no_recompute(fwd, input_parallel)

            if not self.fuse_matmul_bias and self.bias is not None:
                output = output_ + self.bias
            else:
                output = output_
        else:
            output = self.linear(input_parallel, self.weight, self.bias, name=self._name)

        return output


class RRColumnSequenceParallelLinear(ColumnSequenceParallelLinear):
    """RRColumnSequenceParallelLinear"""

    def forward(self, x):
        if self.mp_async_allreduce:
            output = sequence_parallel_utils.SPInnerOverlapLinear.apply(
                x,
                self.weight,
                self.bias,
                self.fuse_matmul_bias,
                self.recompute_allgather,
                self.mp_fused_linear_param_grad_add,
                self.model_parallel_group,
            )
        else:
            input_parallel = sequence_parallel_utils.AllGatherOp.apply(x) if self.is_mp else x

            def fwd(input_parallel):
                output = self.linear(input_parallel, self.weight, self.bias, name=self._name)
                return output

            # create a dummpy fwd function
            output = no_recompute(fwd, input_parallel)
        return output


class RRRowSequenceParallelLinear(RowSequenceParallelLinear):
    """RRRowSequenceParallelLinear"""

    def forward(self, x):
        input_parallel = x
        if self.is_mp:
            if self.mp_scale is not None:
                bias = self.mp_scale(self.bias, self.world_size)
            else:
                bias = None

            def fwd(input_parallel):
                output_parallel = self.linear(input_parallel, self.weight, bias, name=self._name)
                output_ = sequence_parallel_utils.ReduceScatterOp.apply(output_parallel)
                return output_

            # create a dummpy fwd function
            output_ = no_recompute(fwd, input_parallel)
            # register_hook to all_reduce self.bias
            if bias is None and self.bias is not None:
                output = output_ + self.bias
            else:
                output = output_
        else:
            output = self.linear(input_parallel, self.weight, self.bias, name=self._name)
        return output


# if __name__ == "__main__":
#     # test flashmask_attention
#     paddle.seed(2024)
#     from paddle.nn.functional.flash_attention import flashmask_attention

#     dtype = "float16"
#     paddle.set_default_dtype(dtype)

#     in_weight_shape = (32, 3 * 2 * 32)
#     linear1 = paddle.nn.Linear(
#         in_weight_shape[0],
#         in_weight_shape[-1],
#     )
#     paddle.seed(2024)
#     in_weight = paddle.create_parameter(shape=in_weight_shape, dtype=dtype, name="in_weight")
#     in_weight.set_value(paddle.normal(0, 0.02, in_weight_shape))
#     in_weight.main_grad = paddle.normal(0, 0.02, in_weight.shape).cast("float32")
#     linear1.weight.set_value(in_weight)
#     in_bias = paddle.create_parameter(shape=(in_weight.shape[-1],), dtype=dtype, name="in_bias", is_bias=True)
#     in_bias.main_grad = paddle.normal(0, 0.02, in_bias.shape).cast("float32")
#     linear1.bias.set_value(in_bias)
#     linear1.weight.main_grad = in_weight.main_grad
#     linear1.bias.main_grad = in_bias.main_grad

#     out_weight_shape = (2 * 32, 32)
#     out_weight = paddle.create_parameter(shape=out_weight_shape, dtype=dtype, name="out_weight")
#     out_weight.set_value(paddle.normal(0, 0.02, out_weight_shape))
#     out_weight.main_grad = paddle.normal(0, 0.02, out_weight.shape).cast("float32")

#     class cus_multiply(paddle.autograd.PyLayer):
#         @staticmethod
#         def forward(ctx, a, b):
#             y = paddle.multiply(a, b)
#             ctx.save_for_backward(a, b)
#             return y

#         @staticmethod
#         def backward(ctx, dy):
#             a, b = ctx.saved_tensor()
#             grad_a = dy * a
#             grad_b = dy * b
#             return grad_a, grad_b

#     multiply = cus_multiply.apply

#     def fwd(x, startend_row_indices, enable=True):
#         def fwd_linear(x):
#             weight = multiply(linear1.weight, linear1.weight * 0.1)
#             bias = multiply(linear1.bias, linear1.bias * 0.1)
#             qkv = paddle.nn.functional.silu(paddle.nn.functional.linear(x, weight, bias))
#             q, k, v = paddle.chunk(qkv, 3, axis=-1)
#             q = q.reshape([q.shape[0], q.shape[1], 2, q.shape[2] // 2])
#             k = k.reshape([k.shape[0], k.shape[1], 2, v.shape[2] // 2])
#             v = v.reshape([v.shape[0], k.shape[1], 2, v.shape[2] // 2])
#             return q, k, v

#         q, k, v = no_recompute(fwd_linear, x, enable=enable)

#         q, k, v = q * q, k * k, v * v
#         out = no_recompute(
#             flashmask_attention,
#             q,
#             k,
#             v,
#             startend_row_indices=startend_row_indices,
#             causal=True,
#             enable=enable,
#         )
#         out = out.flatten(-2, -1)
#         out = paddle.matmul(out, out_weight)
#         return out

#     x = paddle.normal(0, 0.02, (1, 128, 32))
#     x.stop_gradient = False
#     x_input = x
#     startend_row_indices = paddle.randint(0, 128, (1, 2, 128, 1), dtype="int32")

#     enable = True
#     # 第一层
#     o1 = recompute(
#         fwd,
#         x,
#         startend_row_indices,
#         enable=enable,
#     )
#     # 第二层
#     o2 = recompute(fwd, o1 + x, startend_row_indices, enable=enable)
#     # 第三层
#     o3 = recompute(fwd, o2 + x, startend_row_indices, enable=enable)

#     o3.sum().backward()
#     print(x_input.grad.mean())
#     print(linear1.weight.grad.mean())
#     print(out_weight.grad.mean())
