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
"""refined recompute"""

import inspect
import queue
from collections import defaultdict

import paddle
from paddle import framework
from paddle.base import core

__all__ = [
    "RefinedRcomputeQueue",
    "global_rr_queue_log",
    "RefinedRecomputeFunction",
    "create_skip_config_for_refined_recompute",
]


_is_second_fwd = False


def is_second_fwd():
    """
    Determine if it is the second forward propagation
    """
    global _is_second_fwd
    return _is_second_fwd


def set_second_fwd(value=True):
    """
    Set whether to perform the second forward propagation based on the value
    """
    global _is_second_fwd
    _is_second_fwd = value


class CustomSavedTensorsHooks:
    """
    Customize saved_tensors_hooks, add logic for switching
    variables related to the second forward propagation
    """

    def __init__(self, pack_hook, unpack_hook) -> None:
        """
        initialize the CustomSavedTensorsHooks object
        """
        self.pack_hook = pack_hook
        self.unpack_hook = unpack_hook

        self._prev = is_second_fwd()
        pack_hook_name = f"{pack_hook.__module__}.{pack_hook.__name__}"
        unpack_hook_name = f"{unpack_hook.__module__}.{unpack_hook.__name__}"
        self._is_second_fwd = (
            pack_hook_name == "paddle.distributed.fleet.recompute.recompute.inner_pack"
            and unpack_hook_name == "paddle.distributed.fleet.recompute.recompute.inner_unpack"
        )

    def __enter__(self) -> None:
        """
        enter the context of CustomSavedTensorsHooks
        """
        set_second_fwd(self._is_second_fwd)
        core.eager.register_saved_tensors_hooks(self.pack_hook, self.unpack_hook)

    def __exit__(self, *args: object) -> None:
        """
        exit the context of CustomSavedTensorsHooks
        """
        set_second_fwd(self._prev)
        core.eager.reset_saved_tensors_hooks()


# hack saved_tensors_hooks add set_second_fwd decorator
paddle.autograd.saved_tensors_hooks = CustomSavedTensorsHooks


def create_skip_config_for_refined_recompute(layer_idx, config):
    """
    Creates a configuration for skipping recomputation based on the configuration file,
    effective only at the specified layer index.

    Args:
        layer_idx (int): The layer index used to check whether recomputation should be skipped.
        config (dict): The configuration file of the input model.

    Returns:
        dict: Returns an updated configuration file containing the following key-value pairs:
            - skip_recompute_ops (dict): A dictionary with each model layer's each operation's name and a boolean
                                         indicating whether to skip recomputation, defaults to None.
            - If the refined_recompute key does not exist or recompute is set to False,
              the original configuration file is returned.

    """
    if not config.recompute:
        return config
    skip_config = dict()

    if len(config.refined_recompute) > 0 and config.recompute_granularity != "full":
        raise ValueError(
            "Selective recompute only support full recompute now, " "please set recompute_granularity to `full`."
        )

    for op_name, skip_num in config.refined_recompute.items():
        if skip_num == 0:  # 0 means all recompute
            skip_config[op_name] = False
        elif skip_num < 0:  # < 0 means all skip recompute
            skip_config[op_name] = True
        else:
            if layer_idx < skip_num:  # < the number of layers to skip recompute
                skip_config[op_name] = True
            else:
                skip_config[op_name] = False

    config.skip_recompute_ops[layer_idx] = skip_config
    return config


class RefinedRcomputeQueue:
    """
    Thread-safe queue management system for recomputation operations.

    Provides a mechanism to track and validate multiple recomputation queues
    with automatic naming and existence checking capabilities.
    """

    def __init__(self):
        """
        Initializes an empty queue registry.
        """
        self.rr_queue = defaultdict(queue.Queue)

    def update(self, queue: queue.Queue, queue_name="unknown"):
        """
        Registers a new queue in the management system.

        Args:
            queue (queue.Queue): The queue object to register
            queue_name (str): Base identifier for the queue (default: "unknown")
                Note: Automatically appends the queue's memory address for uniqueness

        Raises:
            ValueError: If a queue with the generated name already exists
        """
        queue_name = f"{queue_name}_{id(queue)}"
        if queue_name in self.rr_queue:
            raise ValueError(f"Queue name '{queue_name}' already exists.")
        self.rr_queue[queue_name] = queue

    def check(self):
        """
        Validates all registered queues are empty.

        Raises:
            ValueError: If any registered queue contains pending items
                Reports all non-empty queue names in the error message
        """
        non_empty_queues = [name for name, queue in self.rr_queue.items() if queue.qsize() != 0]
        if non_empty_queues:
            raise ValueError(f"Queues {', '.join(non_empty_queues)} are not empty.")


global_rr_queue_log = RefinedRcomputeQueue()


class _NoopSaveInputs(paddle.autograd.PyLayer):
    """
    This layer does nothing but save all input tensors.
    This is used to prevent the gradients of the inputs being computed.
    """

    @staticmethod
    def forward(ctx, *args):
        """This function does nothing but save all input tensors."""
        tensors = [o.detach() for o in args if isinstance(o, paddle.Tensor)]
        ctx.save_for_backward(*tensors)
        # Return a dummy tensor which will be automatically released by the framework.
        return paddle.empty((0,), dtype=tensors[0].dtype)

    @staticmethod
    def backward(ctx, *args):
        """Should not be called since we don't support backward on this graph."""
        raise AssertionError("Did not expect to backward on this graph")


class RefinedRecomputeFunction:
    """refined recompute for function"""

    def __init__(self):
        """
        initialize the RefinedRecomputeFunction object.
        """
        self.is_init = False

    def post_init(self, function, function_name=None):
        """
        post init the RefinedRecomputeFunction object.
        """
        if not self.is_init:
            if function_name is None:
                function_name = f"{function.__module__}.{function.__name__}"
            self._hold_tensors_queue = queue.Queue()
            global_rr_queue_log.update(self._hold_tensors_queue, function_name)
            self.function = function
            self.function_name = function_name
            self.is_init = True

    def __call__(self, function, *args, **kwargs):
        """
        call the RefinedRecomputeFunction object.
        """
        # in paddle.no_grad(), return the original output
        if not framework._dygraph_tracer()._has_grad:
            return function(*args, **kwargs)
        self.post_init(function)
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Refined Recompute Forward"""
        if is_second_fwd():
            output = self._second_fwd(*args, **kwargs)
        else:
            output = self._first_fwd(*args, **kwargs)
        return output

    def _first_fwd(self, *args, **kwargs):
        """
        do the first forward
        """
        input_args = self.parse_to_args(*args, **kwargs)

        # chose the right function
        if self.function_name in [
            "paddle.nn.functional.linear",
            "paddle.nn.functional.common.linear",
            "paddle.incubate.nn.functional.fused_linear",
            "paddle.incubate.nn.functional.fused_matmul_bias.fused_linear",
        ] or self.function_name.endswith("linear_reduce_scatter"):
            # is linear function
            outputs = self.function(*input_args)
            self._hold_tensors_queue.put([outputs])
            return outputs
        else:
            if self.function_name == "paddle.nn.functional.flash_attention.flashmask_attention":
                kwargs["return_softmax_lse"] = True
                kwargs["return_seed_offset"] = True
                outputs = self.function(*args, **kwargs)  # outputs is [out, result_softmax_lse, result_seed_offset]
            elif self.function_name == "paddle.nn.functional.flash_attention.flash_attention_with_sparse_mask":
                kwargs["return_softmax"] = False
                kwargs["return_softmax_lse"] = True
                kwargs["return_seed_offset"] = True
                outputs = self.function(*args, **kwargs)  # outputs is [out, result_softmax_lse, result_seed_offset]
            elif self.function_name in [
                "paddle.nn.functional.scaled_dot_product_attention",
                "paddle.nn.functional.flash_attention.scaled_dot_product_attention",
            ]:
                fixed_seed_offset = (None,)
                return_softmax = False
                rng_name = ""
                outputs = list(
                    paddle._C_ops.flash_attn(
                        *input_args[:3],
                        fixed_seed_offset,
                        *input_args[3:6],
                        return_softmax,
                        not input_args[6],
                        rng_name,
                    )
                )
                outputs.pop(1)  # outputs is [out, result_softmax_lse, result_seed_offset]
            else:
                raise ValueError(f"Unknown function: {self.function_name}, please implement it first!")
            self._hold_tensors_queue.put(outputs)
            return outputs[0]

    def _second_fwd(self, *args, **kwargs):
        """
        do the second forward
        """
        assert not self._hold_tensors_queue.empty(), "queue should not be empty"
        input_args = self.parse_to_args(*args, **kwargs)
        hold_tensors = self._hold_tensors_queue.get()
        if len(hold_tensors) == 1:  # is linear function
            _NoopSaveInputs.apply(*input_args[:2])
        else:  # is flash function
            _NoopSaveInputs.apply(*input_args, *hold_tensors)
        return hold_tensors[0]

    def parse_to_args(self, *args, **kwargs):
        """
        parse the input arguments and keywords to a list of arguments.
        """
        input_args = []
        dyfunc_sig = inspect.signature(self.function)
        bound_args = dyfunc_sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        for arg, param in zip(bound_args.arguments.values(), dyfunc_sig.parameters.values()):
            if param.kind == param.VAR_POSITIONAL:
                input_args.extend(arg)
            elif param.kind in (
                param.POSITIONAL_ONLY,
                param.POSITIONAL_OR_KEYWORD,
            ):
                input_args.append(arg)
            elif param.kind == param.VAR_KEYWORD:
                input_args.extend(arg.values())
            elif param.kind == param.KEYWORD_ONLY:
                input_args.append(arg)
            else:
                raise ValueError("Unknown parameter kind.")
        return input_args
