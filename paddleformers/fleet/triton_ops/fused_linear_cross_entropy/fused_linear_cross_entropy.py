#!/usr/bin/env python3

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

"""
Fused Linear Cross Entropy：将线性层 + 交叉熵合并为一个显存高效的操作。

基于 Liger-Kernel 的分块思路：在每个 chunk 上只物化 [chunk, V] 的 logits，
Triton kernel 前向计算 loss 的同时原地写回梯度，避免保留完整
[BT, V] softmax/logits 中间张量。
"""

import paddle
import triton

from .cross_entropy import (
    liger_cross_entropy_kernel,
    liger_cross_entropy_multimax_kernel,
)
from .utils import element_mul_kernel

MAX_FUSED_SIZE = 65536 // 2


def fused_linear_cross_entropy_forward(
    _input,
    weight,
    target,
    bias=None,
    ignore_index=-100,
    reduction="none",
    num_chunks=1,
    ec_align=False,
    multimax_ranges=None,
    multimax_ts=None,
):
    """前向：分 chunk 计算 logits / loss / grad_input / grad_weight。

    参数:
        _input: [BT, H] 输入 hidden states (bf16/fp16)。
        weight: [V, H] 线性层权重 (与 F.linear 一致)。
        target: [BT] 整数目标标签。
        bias: [V] 可选偏置。
        ignore_index: 被忽略的标签值。
        reduction: "none" / "mean" / "sum"。
        num_chunks: 分多少个 chunk 做计算。
        ec_align: True 时启用与 ernie-core 的精度对齐模式：
                  grad_weight 使用 [H, V] 布局（GEMM [H,C]@[C,V]），
                  与 ernie-core 的 fused_linear_param_grad_add 调用完全相同。
                  backward 中 main_grad.add_(grad_weight.T)。
        multimax_ranges, multimax_ts: 可选 [4]-shape 张量。若两者均提供，
            则使用 `liger_cross_entropy_multimax_kernel` 在 Triton 内核中
            将 SegLU 段式调制 + CE + SegLU 反向 全部融合在两次扫描内完成；
            grad_multimax_{ranges,ts} 通过 atomic_add 写入返回缓冲。
            相比在 Python 中逐 chunk 计算 SegLU 前向 / 反向，可避免在 HBM
            上物化 4 个 ReLU 中间张量与 SegLU 输出，将每个 chunk 的 vocab
            轴显存峰值从 ~10× 降回 ~1×。
            SegLU(x) = x + t0·max(r0-x,0) + t1·max(x-r1,0)
                     + t2·max(r2-x,0)^2 + t3·max(x-r3,0)^2
            初始化为 0 时 SegLU 为恒等映射，与未启用路径数值一致。
    """
    input_requires_grad = not _input.stop_gradient
    weight_requires_grad = not weight.stop_gradient
    bias_requires_grad = bias is not None and not bias.stop_gradient
    # The Triton kernel writes softmax-grad (= grad_logits) in-place into
    # logits_chunk whenever ANY of {input, weight, bias} needs a gradient,
    # since grad_logits feeds all three downstream GEMMs. Gating this on
    # `input_requires_grad` alone breaks the freeze-backbone / train-head
    # case where _input.stop_gradient=True but weight.stop_gradient=False.
    needs_grad_logits = (
        input_requires_grad or weight_requires_grad or bias_requires_grad
    )
    orig_dtype = _input.dtype

    BT, H = _input.shape
    V = weight.shape[0]  # weight 是 [V, H]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    chunk_size = triton.cdiv(BT, num_chunks)

    grad_input = (
        paddle.zeros([BT, H], dtype=paddle.float32)
        if input_requires_grad
        else None
    )
    # ec_align 模式：grad_weight 使用 [H, V] 布局，与 ernie-core 的 GEMM shape 一致。
    # 默认模式：grad_weight 使用 [V, H] 布局（与 weight.shape 一致，main_grad 可直接 add_）。
    if weight_requires_grad:
        grad_weight = (
            paddle.zeros([H, V], dtype=paddle.float32)
            if ec_align
            else paddle.zeros(weight.shape, dtype=paddle.float32)
        )
    else:
        grad_weight = None
    grad_bias = (
        paddle.zeros([bias.shape[0]], dtype=paddle.float32)
        if bias_requires_grad
        else None
    )

    # Multimax: SegLU is applied inside the Triton kernel
    # (`liger_cross_entropy_multimax_kernel`), which fuses SegLU forward,
    # online lse, softmax-grad, and SegLU backward in a single two-pass
    # scan over the chunk. Per-row partial sums for grad_ranges/grad_ts
    # are atomic-added into these fp32 [4] buffers from the kernel.
    use_multimax = multimax_ranges is not None and multimax_ts is not None
    if use_multimax:
        grad_multimax_ranges = paddle.zeros([4], dtype=paddle.float32)
        grad_multimax_ts = paddle.zeros([4], dtype=paddle.float32)
        # Multimax param grads must be computed whenever EITHER param is
        # trainable, independently of whether the upstream hidden states
        # require grad. This covers the freeze-backbone / train-head-only
        # regime where _input.stop_gradient=True but multimax_*.stop_gradient=False.
        multimax_requires_grad = (
            not multimax_ranges.stop_gradient or not multimax_ts.stop_gradient
        )
        # Extract scalar values once (one host<->device sync for an [4]
        # tensor; negligible vs the chunk's GEMM/CE cost). Using fp32 for
        # numerical stability inside the kernel.
        _r_vals = [float(v) for v in multimax_ranges.cast("float32").tolist()]
        _t_vals = [float(v) for v in multimax_ts.cast("float32").tolist()]
    else:
        grad_multimax_ranges = None
        grad_multimax_ts = None
        multimax_requires_grad = False
        _r_vals = None
        _t_vals = None

    loss_1d = paddle.zeros([BT], dtype=paddle.float32)

    target_mask = target != ignore_index
    # n_non_ignore is only used inside the kernel when reduction=="mean".
    # reduction is a tl.constexpr so the mean branch is dead code for
    # other modes; skip the D2H sync entirely in those cases.
    total_n_non_ignore = (
        int(target_mask.sum().item()) if reduction == "mean" else 0
    )

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]

        logits_chunk = paddle.compat.nn.functional.linear(
            _input_chunk, weight, bias
        )

        target_chunk = target[start_idx:end_idx]
        n_rows = logits_chunk.shape[0]

        loss_1d_slice = loss_1d[start_idx:end_idx]

        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        if use_multimax:
            liger_cross_entropy_multimax_kernel[(n_rows,)](
                X_ptr=logits_chunk,
                X_stride=logits_chunk.stride(-2),
                Y_ptr=target_chunk,
                Y_stride=target_chunk.stride(-1),
                loss_ptr=loss_1d_slice,
                loss_stride=loss_1d_slice.stride(-1),
                n_cols=V,
                n_non_ignore=total_n_non_ignore,
                ignore_index=ignore_index,
                r0=_r_vals[0],
                r1=_r_vals[1],
                r2=_r_vals[2],
                r3=_r_vals[3],
                t0=_t_vals[0],
                t1=_t_vals[1],
                t2=_t_vals[2],
                t3=_t_vals[3],
                grad_r_ptr=grad_multimax_ranges,
                grad_t_ptr=grad_multimax_ts,
                reduction=reduction,
                HAS_GRADIENTS=needs_grad_logits,
                HAS_MULTIMAX_GRADIENTS=multimax_requires_grad,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32,
            )
        else:
            liger_cross_entropy_kernel[(n_rows,)](
                X_ptr=logits_chunk,
                X_stride=logits_chunk.stride(-2),
                Y_ptr=target_chunk,
                Y_stride=target_chunk.stride(-1),
                loss_ptr=loss_1d_slice,
                loss_stride=loss_1d_slice.stride(-1),
                n_cols=V,
                n_non_ignore=total_n_non_ignore,
                ignore_index=ignore_index,
                reduction=reduction,
                HAS_GRADIENTS=needs_grad_logits,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32,
            )

        loss_1d[start_idx:end_idx] = loss_1d_slice
        # kernel 已将梯度（含 SegLU 反向链式法则）原地写回 logits_chunk
        grad_logits_chunk = logits_chunk

        if input_requires_grad:
            grad_input[start_idx:end_idx] = paddle.matmul(
                grad_logits_chunk, weight
            )

        if grad_weight is not None:
            with paddle.amp.auto_cast(False):
                if ec_align:
                    # ec_align: grad_weight=[H,V]，GEMM [H,C]@[C,V]，与 ernie-core 一致。
                    # fused_linear_param_grad_add(x,y,dw): dw += x.T @ y
                    # 令 x=_input_chunk[C,H], y=grad_logits_chunk[C,V] → dw += [H,C]@[C,V] = [H,V] ✓
                    paddle._C_ops.fused_linear_param_grad_add(
                        _input_chunk,  # [C, H]
                        grad_logits_chunk,  # [C, V]
                        grad_weight,  # [H, V]
                        None,
                        True,
                        False,
                    )
                else:
                    # 默认: grad_weight=[V,H]，GEMM [V,C]@[C,H]
                    # fused_linear_param_grad_add(x,y,dw): dw += x.T @ y
                    # 令 x=grad_logits_chunk[C,V], y=_input_chunk[C,H] → dw += [V,C]@[C,H] = [V,H] ✓
                    paddle._C_ops.fused_linear_param_grad_add(
                        grad_logits_chunk,  # [C, V]
                        _input_chunk,  # [C, H]
                        grad_weight,  # [V, H]
                        None,
                        True,
                        False,
                    )

        if grad_bias is not None:
            grad_bias.add_(grad_logits_chunk.sum(axis=0))

    if input_requires_grad:
        grad_input = grad_input.cast(orig_dtype)
    if grad_bias is not None:
        grad_bias = grad_bias.cast(bias.dtype)

    if reduction == "none":
        loss = loss_1d
    else:
        loss = paddle.sum(loss_1d)

    # Backward compatibility: return 4-tuple when multimax is disabled,
    # 6-tuple when enabled. Direct callers expecting 4 values won't break.
    if multimax_ranges is None:
        return loss, grad_input, grad_weight, grad_bias
    else:
        return (
            loss,
            grad_input,
            grad_weight,
            grad_bias,
            grad_multimax_ranges,
            grad_multimax_ts,
        )


def fused_linear_cross_entropy_backward(
    grad_output, grad_input, grad_weight, grad_bias
):
    """反向：当 grad_output != 1.0 时，对已保存的梯度做缩放。"""
    if grad_output.shape == [] and float(grad_output) == 1.0:
        return grad_input, grad_weight, grad_bias

    # none 模式：grad_output 是 [BT] 向量 (有效 token = 1/N，无效 = 0)。
    # forward 已将无效 token 的 grad_logits 清零，所有有效 token 缩放因子相同。
    if grad_output.ndim >= 1:
        grad_output = grad_output.max().reshape([])

    if grad_input is not None:
        BT, H = grad_input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        element_mul_kernel[(n_rows,)](
            grad_input,
            grad_input.stride(-2),
            grad_output,
            H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )
    else:
        BLOCK_SIZE = MAX_FUSED_SIZE

    if grad_weight is not None:
        n_rows_w = grad_weight.shape[0]
        n_cols_w = grad_weight.shape[1]
        BLOCK_SIZE_W = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols_w))
        element_mul_kernel[(n_rows_w,)](
            grad_weight,
            grad_weight.stride(-2),
            grad_output,
            n_cols_w,
            BLOCK_SIZE=BLOCK_SIZE_W,
            num_warps=32,
        )

    if grad_bias is not None:
        V = grad_bias.shape[0]
        element_mul_kernel[(V,)](
            grad_bias,
            grad_bias.stride(-1),
            grad_output,
            1,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )

    return grad_input, grad_weight, grad_bias


class LigerFusedLinearCrossEntropyFunction(paddle.autograd.PyLayer):
    """fused linear + cross entropy 的自定义前向 / 反向。

    使用方式:
        loss = LigerFusedLinearCrossEntropyFunction.apply(
            _input,             # [BT, H]
            weight,             # [V, H]
            target,             # [BT]
            bias,               # [V] 或 None
            ignore_index,
            reduction,
            num_chunks,
            ec_align,
            multimax_ranges,    # [4] 可选，启用 multimax lm_head 时传入
            multimax_ts,        # [4] 可选，启用 multimax lm_head 时传入
        )
    """

    @staticmethod
    def forward(ctx, *args):
        _input = args[0]
        weight = args[1]
        target = args[2]
        bias = args[3]
        ignore_index = args[4]
        reduction = args[5]
        num_chunks = args[6]
        ec_align = args[7]
        multimax_ranges = args[8] if len(args) > 8 else None
        multimax_ts = args[9] if len(args) > 9 else None

        ret = fused_linear_cross_entropy_forward(
            _input=_input,
            weight=weight,
            target=target,
            bias=bias,
            ignore_index=ignore_index,
            reduction=reduction,
            num_chunks=num_chunks,
            ec_align=ec_align,
            multimax_ranges=multimax_ranges,
            multimax_ts=multimax_ts,
        )
        # Handle both 4-tuple (multimax disabled) and 6-tuple (enabled)
        if len(ret) == 4:
            loss, grad_input, grad_weight, grad_bias = ret
            grad_mm_ranges = grad_mm_ts = None
        else:
            (
                loss,
                grad_input,
                grad_weight,
                grad_bias,
                grad_mm_ranges,
                grad_mm_ts,
            ) = ret

        ctx.save_for_backward(
            grad_input.detach() if grad_input is not None else None,
            grad_weight.detach() if grad_weight is not None else None,
            grad_bias.detach() if grad_bias is not None else None,
            grad_mm_ranges.detach() if grad_mm_ranges is not None else None,
            grad_mm_ts.detach() if grad_mm_ts is not None else None,
        )
        ctx.has_bias = bias is not None
        ctx.has_multimax = multimax_ranges is not None
        ctx.weight_ref = weight
        ctx.weight_requires_grad = not weight.stop_gradient
        ctx.multimax_ranges_ref = multimax_ranges
        ctx.multimax_ts_ref = multimax_ts
        # Cache stop_gradient at forward time. multimax_requires_grad in the
        # forward (the kernel-level flag) is the OR of the two params' grad
        # requirements, so when only one is frozen the kernel still produces
        # both grads. The backward must respect each param's individual
        # stop_gradient: a frozen param contributes nothing to main_grad and
        # returns None at its PyLayer slot (cf. tensor_parallel/layers.py
        # PyLayer contract: forward Tensor inputs with stop_gradient=True
        # MUST get None at the matching backward position).
        ctx.multimax_ranges_requires_grad = (
            multimax_ranges is not None and not multimax_ranges.stop_gradient
        )
        ctx.multimax_ts_requires_grad = (
            multimax_ts is not None and not multimax_ts.stop_gradient
        )
        ctx.ec_align = ec_align
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (
            grad_input,
            grad_weight,
            grad_bias,
            grad_mm_ranges,
            grad_mm_ts,
        ) = ctx.saved_tensor()
        grad_input, grad_weight, grad_bias = (
            fused_linear_cross_entropy_backward(
                grad_output, grad_input, grad_weight, grad_bias
            )
        )

        # Scale multimax grads by the same factor as the other grads.
        # `fused_linear_cross_entropy_backward` already collapsed
        # `grad_output` to a scalar when needed; replicate the same
        # detection here to stay consistent.
        if ctx.has_multimax and (grad_mm_ranges is not None):
            scale = grad_output
            if scale.shape == [] and float(scale) == 1.0:
                pass
            else:
                if scale.ndim >= 1:
                    scale = scale.max().reshape([])
                grad_mm_ranges = grad_mm_ranges * scale.cast(
                    grad_mm_ranges.dtype
                )
                grad_mm_ts = grad_mm_ts * scale.cast(grad_mm_ts.dtype)

        if ctx.weight_requires_grad and grad_weight is not None:
            weight = ctx.weight_ref
            if hasattr(weight, "main_grad"):
                if weight.main_grad is None:
                    weight.main_grad = paddle.zeros(
                        weight.shape, dtype=paddle.float32
                    )
                if ctx.ec_align:
                    # ec_align: grad_weight=[H,V]，main_grad=[V,H]，需转置后累加
                    weight.main_grad.add_(grad_weight.T)
                else:
                    # 默认: grad_weight=[V,H]，与 main_grad=[V,H] 相同，直接累加
                    weight.main_grad.add_(grad_weight)
                if hasattr(weight, "_apply_backward_hook"):
                    weight._apply_backward_hook()
                grad_weight = None

        # Multimax params: accumulate into main_grad when present (matches
        # the weight pattern); otherwise fall back to returning the grad
        # so the standard autograd accumulator handles it. Frozen params
        # (stop_gradient=True) MUST get None at their PyLayer slot and
        # MUST NOT touch main_grad / fire backward hooks, even though the
        # kernel produced a grad tensor for them (multimax_requires_grad
        # is the OR of the two params' grad-requirements).
        mm_ranges_out = None
        mm_ts_out = None
        if ctx.has_multimax:
            for param, g, slot, requires_grad in (
                (
                    ctx.multimax_ranges_ref,
                    grad_mm_ranges,
                    "ranges",
                    ctx.multimax_ranges_requires_grad,
                ),
                (
                    ctx.multimax_ts_ref,
                    grad_mm_ts,
                    "ts",
                    ctx.multimax_ts_requires_grad,
                ),
            ):
                if param is None or g is None:
                    continue
                if not requires_grad:
                    # Param was frozen at forward time; respect that here.
                    # Leave mm_*_out as None for this slot.
                    continue
                if hasattr(param, "main_grad"):
                    if param.main_grad is None:
                        param.main_grad = paddle.zeros(
                            param.shape, dtype=paddle.float32
                        )
                    param.main_grad.add_(g.cast(param.main_grad.dtype))
                    if hasattr(param, "_apply_backward_hook"):
                        param._apply_backward_hook()
                    # multimax params have stop_gradient=False so Paddle's
                    # autograd requires a non-None gradient at this position.
                    # Return zeros: real gradient is already in main_grad for
                    # the optimizer to consume; param.grad zeros are ignored.
                    out_grad = paddle.zeros(param.shape, dtype=param.dtype)
                else:
                    out_grad = g.cast(param.dtype)
                if slot == "ranges":
                    mm_ranges_out = out_grad
                else:
                    mm_ts_out = out_grad

        result = [grad_input, grad_weight, None]
        if ctx.has_bias:
            result.append(grad_bias)
        if ctx.has_multimax:
            result.append(mm_ranges_out)
            result.append(mm_ts_out)
        return tuple(result)
