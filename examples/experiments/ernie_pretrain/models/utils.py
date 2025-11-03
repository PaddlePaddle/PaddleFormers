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
from typing import Any, Callable, List

import paddle
from paddle import framework

logger = logging.getLogger(__name__)

try:
    import moe_permutation

except ImportError:
    moe_permutation = None
    logger.warning("moe_permutation is not installed.")


def get_global_training_logs():
    try:
        from src.utils.misc import global_training_logs

        return global_training_logs
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        from rl.utils.stat_utils import global_training_logs

        return global_training_logs
    except (ImportError, ModuleNotFoundError):
        pass
    return {}


def global_training_logs_enabled():
    global_training_logs = get_global_training_logs()
    return isinstance(global_training_logs, dict) or global_training_logs.is_enabled()


def inplace_offload(tensor):
    tmp = tensor.pin_memory() if paddle.is_compiled_with_cuda() else tensor.cpu()
    tmp._share_buffer_to(tensor)


def detach_and_requires_grad_(*args):
    ret = [a.detach() if a is not None else None for a in args]
    for r, a in zip(ret, args):
        if a is not None:
            r.stop_gradient = a.stop_gradient
    return ret


class FakeClone(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input):
        if input.is_contiguous():
            fake_output = paddle.empty_like(input)
            input._share_buffer_to(fake_output)
        else:
            fake_output = input.clone()
        return fake_output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def manual_backward(f: Callable, is_first_fwd: bool, *args: List[Any]):
    tracer = framework._dygraph_tracer()
    orig = tracer._has_grad
    if not is_first_fwd:
        tracer._has_grad = True

    detached_args = detach_and_requires_grad_(*args)
    detached_args_clone = [FakeClone.apply(a) if a is not None else None for a in detached_args]
    out = f(*detached_args_clone)
    if isinstance(out, list):
        out = tuple(out)
    elif not isinstance(out, tuple):
        out = (out,)

    if is_first_fwd:
        tracer._has_grad = orig
        return None, out

    out_cached = [FakeClone.apply(o) for o in out if o is not None]

    for o in out_cached:
        o._clear_dataptr()
    tracer._has_grad = orig

    def bwd_f(*grad):
        nonlocal out_cached, detached_args, f
        grad = list(grad)
        grad = [g for g in grad if g is not None]
        assert grad and out_cached, (len(grad), len(out_cached))
        grad, out_cached = zip(*[(g, o) for g, o in zip(grad, out_cached) if not o.stop_gradient])

        assert len(grad) == len(out_cached), (len(grad), len(out_cached), f)
        paddle.autograd.backward(out_cached, grad)
        return tuple([t.grad for t in detached_args if t is not None])

    return bwd_f, out


class FakeGather(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input, indices):
        assert len(indices.shape) == 1
        ctx.save_for_backward(indices, input.shape)
        if indices.shape[0] == 0:
            out_shape = input.shape
            out_shape[0] = 0
            return paddle.zeros(out_shape, dtype=input.dtype)
        return paddle.index_select(input, axis=0, index=indices)

    @staticmethod
    def backward(ctx, grad_output):
        indices, input_shape = ctx.saved_tensor()

        grad_input = paddle.zeros(input_shape, dtype=grad_output.dtype)
        if indices.shape[0] != 0:
            paddle.scatter_(grad_input, indices.unsqueeze(-1), grad_output, overwrite=False)
        return grad_input, None


class FusedUnpermutation(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        output_tokens,
        permuted_tokens,
        token_permuted_indices,
        dispatched_probs,
        prob_permuted_indices,
    ):
        assert token_permuted_indices.stop_gradient, "token_permuted_indices must be stop_gradient"
        if dispatched_probs is not None:
            assert (
                prob_permuted_indices is not None and prob_permuted_indices.stop_gradient
            ), "dispatched_probs must be stop_gradient"

        output_tokens.stop_gradient = False

        src_token_num = permuted_tokens.shape[0]
        if src_token_num > 0:
            output_tokens = moe_permutation.unpermute(
                output_tokens,
                permuted_tokens,
                token_permuted_indices,
                dispatched_probs,
                prob_permuted_indices,
            )
        else:
            output_tokens = FakeClone.apply(output_tokens)

        ctx.save_for_backward(
            permuted_tokens,
            token_permuted_indices,
            dispatched_probs,
            prob_permuted_indices,
        )
        return output_tokens

    @staticmethod
    def backward(ctx, output_tokens_grad):
        (
            permuted_tokens,
            token_permuted_indices,
            dispatched_probs,
            prob_permuted_indices,
        ) = ctx.saved_tensor()

        src_token_num = permuted_tokens.shape[0]
        if src_token_num > 0:
            permuted_tokens_grad, dispatched_probs_grad = moe_permutation.unpermute_grad(
                output_tokens_grad,
                permuted_tokens,
                token_permuted_indices,
                dispatched_probs,
                prob_permuted_indices,
            )
        else:
            permuted_tokens_grad = paddle.zeros_like(permuted_tokens)
            if dispatched_probs is not None:
                dispatched_probs_grad = paddle.zeros_like(dispatched_probs)

        if dispatched_probs is None:
            return output_tokens_grad, permuted_tokens_grad, None
        else:
            return (
                output_tokens_grad,
                permuted_tokens_grad,
                None,
                dispatched_probs_grad,
                None,
            )
