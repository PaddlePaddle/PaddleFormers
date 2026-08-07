# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import inspect
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import paddle
from paddle import nn
from paddle.base.core import CUDAGraph
from paddle.device.cuda import graphs


def get_tensors(obj: Any) -> list[paddle.Tensor]:
    if isinstance(obj, paddle.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        tensors = []
        for x in obj:
            tensors.extend(get_tensors(x))
        return tensors
    if isinstance(obj, dict):
        tensors = []
        for k in sorted(obj.keys()):
            tensors.extend(get_tensors(obj[k]))
        return tensors
    return []


def set_tensors(obj: Any, tensors: list[paddle.Tensor], pos: list[int]) -> Any:
    if isinstance(obj, paddle.Tensor):
        res = tensors[pos[0]]
        pos[0] += 1
        return res
    if isinstance(obj, list):
        return [set_tensors(x, tensors, pos) for x in obj]
    if isinstance(obj, tuple):
        return tuple(set_tensors(x, tensors, pos) for x in obj)
    if isinstance(obj, dict):
        return {
            k: set_tensors(obj[k], tensors, pos) for k in sorted(obj.keys())
        }
    return obj


@dataclass
class CUDAGraphContext:
    eager_warmup_steps: int = 0
    captured: bool = False

    inputs_tensor_buffer: list[paddle.Tensor] | None = field(
        default_factory=list
    )
    inputs_grads_buffer: list[paddle.Tensor | None] = field(
        default_factory=list
    )

    static_params: list[paddle.Tensor] = field(default_factory=list)
    static_params_grads: list[paddle.Tensor | None] = field(
        default_factory=list
    )

    outputs_tensor_buffer: list[paddle.Tensor] | None = field(
        default_factory=list
    )
    outputs_grads_buffer: list[paddle.Tensor | None] = field(
        default_factory=list
    )

    fwd_cudagraph: graphs.CUDAGraph | None = None
    bwd_cudagraph: graphs.CUDAGraph | None = None

    runner: paddle.autograd.PyLayer | None = None


def autocudagraph(
    warmup_steps: int = 1,
    max_graphs: int = 2,
    dispatch_key_fn: Callable[..., tuple] | None = None,
):
    """
    Automated CUDAGraph acceleration decorator.

    This decorator seamlessly captures the forward and backward passes of a dynamic
    computational graph and converts them into highly efficient static CUDAGraph
    execution streams. It automatically manages memory pointer tracking, gradient
    accumulation, and is fully compatible with Philox-based RNG operations (e.g., Dropout).

    Args:
        warmup_steps (int, optional): The number of initial eager execution steps
            required to stabilize the memory pool and workspace before capture.
            Defaults to 1.
        max_graphs (int, optional): The maximum number of cached CUDAGraph instances
            allowed in memory. If dynamic shapes or control flows trigger new graph
            creations beyond this limit, the engine safely falls back to eager mode.
            Defaults to 2.
        dispatch_key_fn (Callable[..., tuple] | None, optional): A routing function
            that takes the bound arguments of the decorated function and returns a
            unique hashable signature (usually a tuple) to identify the specific
            graph topology. If it returns None, graph execution is bypassed entirely
            in favor of eager mode. Defaults to None.
    """

    def decorator(func: Callable):
        sig = inspect.signature(func)
        state_registry: dict[Any, CUDAGraphContext] = {}

        dummy_trigger = paddle.empty([1], dtype="float32")
        dummy_trigger.stop_gradient = False

        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            is_grad_enabled = paddle.is_grad_enabled()
            self_instance = bound.arguments.get("self", None)
            self_instance = (
                self_instance if isinstance(self_instance, nn.Layer) else None
            )

            if dispatch_key_fn is not None:
                key = dispatch_key_fn(bound.arguments)
                if key is None:
                    return func(*args, **kwargs)
            else:
                key = (id(self_instance), is_grad_enabled)

            if key not in state_registry:
                if len(state_registry) >= max_graphs:
                    warnings.warn(
                        f"CUDAGraph cache limit ({max_graphs}) reached for function '{func.__name__}' "
                        f"with dispatch key: {key}. "
                        f"Falling back to eager execution. This is usually caused by highly dynamic "
                        f"input shapes or continuously creating new instances without clearing cache. "
                        f"Consider stabilizing inputs, increasing max_graphs, or calling .clear_cache().",
                        category=RuntimeWarning,
                        stacklevel=2,
                    )
                    return func(*args, **kwargs)
                state_registry[key] = CUDAGraphContext()

            ctx = state_registry[key]

            if ctx.eager_warmup_steps < warmup_steps:
                ctx.eager_warmup_steps += 1
                return func(*args, **kwargs)

            if not ctx.captured:
                static_bound = sig.bind(*args, **kwargs)
                static_bound.apply_defaults()
                # Do NOT use `static_bound = bound` (Reference Assignment).
                # Modifying `static_bound` with detached tensors will mutate the original `bound`.
                # This causes the Autograd engine to receive detached tensors during execution,
                # breaking the computation graph and resulting in zero gradients.
                # Always use `sig.bind` to instantiate a strictly isolated object.

                inputs_tensors = get_tensors(static_bound.arguments)
                ctx.inputs_tensor_buffer = []
                for t in inputs_tensors:
                    c = t.clone().detach()
                    c.stop_gradient = t.stop_gradient
                    ctx.inputs_tensor_buffer.append(c)

                static_bound.arguments = set_tensors(
                    static_bound.arguments, ctx.inputs_tensor_buffer, [0]
                )

                ctx.outputs_grads_buffer = [
                    paddle.empty_like(t) if not t.stop_gradient else None
                    for t in get_tensors(
                        func(*static_bound.args, **static_bound.kwargs)
                    )
                ]

                # [Gradient Accumulation Backup]
                # Snapshot internal weight gradients (e.g., nn.Layer parameters)
                # to prevent dummy backward passes from overwriting them during capture.
                if self_instance is not None:
                    with paddle.no_grad():
                        saved_grads = {
                            p: p.grad.clone()
                            for p in self_instance.parameters()
                            if p.grad is not None
                        }

                pool_id = CUDAGraph.gen_new_memory_pool_id()
                ctx.fwd_cudagraph = graphs.CUDAGraph(pool_id=pool_id)
                ctx.bwd_cudagraph = graphs.CUDAGraph(pool_id=pool_id)

                # warmup
                paddle.device.synchronize()
                for _ in range(3):
                    outputs = func(
                        *static_bound.args,
                        **static_bound.kwargs,
                    )
                    bwd_outputs = []
                    bwd_outputs_grads = []
                    for ot, og in zip(
                        get_tensors(outputs), ctx.outputs_grads_buffer
                    ):
                        if og is not None:
                            bwd_outputs.append(ot)
                            bwd_outputs_grads.append(og)

                    if bwd_outputs:
                        paddle.autograd.backward(
                            bwd_outputs,
                            bwd_outputs_grads,
                        )

                # forward capture
                paddle.device.synchronize()
                ctx.fwd_cudagraph.capture_begin()
                ctx.outputs_tensor_buffer = func(
                    *static_bound.args,
                    **static_bound.kwargs,
                )
                ctx.fwd_cudagraph.capture_end()

                # backward capture
                paddle.device.synchronize()
                ctx.bwd_cudagraph.capture_begin()
                output_static_list, grad_static_list = [], []
                for ot, og in zip(
                    get_tensors(ctx.outputs_tensor_buffer),
                    ctx.outputs_grads_buffer,
                ):
                    if og is not None:
                        output_static_list.append(ot)
                        grad_static_list.append(og)
                assert output_static_list or not paddle.is_grad_enabled()

                if output_static_list:
                    paddle.autograd.backward(
                        output_static_list, grad_static_list, retain_graph=True
                    )
                ctx.bwd_cudagraph.capture_end()
                paddle.device.synchronize()

                ctx.inputs_grads_buffer = [
                    t.grad for t in ctx.inputs_tensor_buffer
                ]

                # [Gradient Accumulation Restore]
                # Recover internal weight gradients post-capture
                # to ensure cross-step gradient accumulation remains intact.
                if self_instance is not None:
                    ctx.static_params = list(self_instance.parameters())
                    ctx.static_params_grads = [
                        p.grad for p in ctx.static_params
                    ]

                class CUDAGraphRunner(paddle.autograd.PyLayer):
                    @staticmethod
                    def forward(ctx_runner, dummy_trigger, *dynamic_in_tensors):
                        ctx_runner.needs_grad = [
                            not t.stop_gradient for t in dynamic_in_tensors
                        ]

                        for stat_t, dyn_t in zip(
                            ctx.inputs_tensor_buffer, dynamic_in_tensors
                        ):
                            stat_t.copy_(dyn_t)

                        ctx.fwd_cudagraph.replay()

                        out_ts = get_tensors(ctx.outputs_tensor_buffer)
                        detached_outputs = [t.clone().detach() for t in out_ts]
                        return tuple(detached_outputs)

                    @staticmethod
                    def backward(ctx_runner, *grad_outputs):
                        for sg, go in zip(
                            ctx.outputs_grads_buffer, grad_outputs
                        ):
                            if sg is not None and go is not None:
                                sg.copy_(go)

                        for static_grad in ctx.inputs_grads_buffer:
                            if static_grad is not None:
                                with paddle.no_grad():
                                    static_grad.zero_()

                        ctx.bwd_cudagraph.replay()

                        grads = [None]  # Dummy trigger
                        for static_grad, stat_t, ng in zip(
                            ctx.inputs_grads_buffer,
                            ctx.inputs_tensor_buffer,
                            ctx_runner.needs_grad,
                        ):
                            actual_grad = (
                                static_grad
                                if static_grad is not None
                                else stat_t.grad
                            )
                            grads.append(
                                actual_grad.clone()
                                if (ng and actual_grad is not None)
                                else None
                            )
                        return tuple(grads)

                ctx.runner = CUDAGraphRunner
                ctx.captured = True

                if self_instance is not None:
                    with paddle.no_grad():
                        for p, saved_grad in saved_grads.items():
                            p.grad.copy_(saved_grad, False)

            dynamic_in_tensors = get_tensors(bound.arguments)
            flat_returned_tensors = ctx.runner.apply(
                dummy_trigger, *dynamic_in_tensors
            )
            final_outputs = set_tensors(
                ctx.outputs_tensor_buffer, flat_returned_tensors, [0]
            )

            return final_outputs

        wrapper.state_registry = state_registry
        wrapper.clear_cache = lambda: state_registry.clear()
        return wrapper

    return decorator
