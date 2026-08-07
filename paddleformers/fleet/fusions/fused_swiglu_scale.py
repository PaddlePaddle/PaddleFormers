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

import paddle
import paddle.nn.functional as F
from paddle.nn.functional import swiglu


def _broadcast_scale(scale, target_dtype, target_ndim):
    """Cast scale to target dtype and unsqueeze to match target rank."""
    scale_exp = scale.cast(target_dtype)
    while scale_exp.ndim < target_ndim:
        scale_exp = scale_exp.unsqueeze(-1)
    return scale_exp


def fused_swiglu_scale_forward(x, scale, clamp_value=None):
    if paddle.is_compiled_with_cuda():
        if clamp_value is not None and clamp_value > 0:
            from paddlefleet_ops import fused_swiglu_scale_clamp

            return fused_swiglu_scale_clamp(x, scale, clamp_value)
        else:
            from paddlefleet_ops import fused_swiglu_scale

            return fused_swiglu_scale(x, scale)
    else:
        if clamp_value is not None and clamp_value > 0:
            # cast to float32, clamp, silu, cast back
            hidden = x.shape[-1] // 2
            x_fp32 = x.cast(paddle.float32)
            gate = paddle.clip(x_fp32[..., :hidden], max=clamp_value)
            val = paddle.clip(
                x_fp32[..., hidden:], min=-clamp_value, max=clamp_value
            )
            out = (F.silu(gate) * val).cast(x.dtype)
        else:
            out = swiglu(x)
        scale_exp = _broadcast_scale(scale, x.dtype, out.ndim)
        return out * scale_exp


def fused_swiglu_scale_backward(x, scale, out_grad, clamp_value=None):
    """Backward for fused SwiGLU * scale.

    When clamp_value is not None, clamps gate to (-inf, clamp_value] and
    value to [-clamp_value, clamp_value], then zeros gradients where inputs
    were saturated.  The gradient output order is [d_gate, d_val] which
    matches ``paddle.chunk(x, 2, axis=-1)`` -> [gate, value].
    """
    if paddle.is_compiled_with_cuda():
        if clamp_value is not None and clamp_value > 0:
            from paddlefleet_ops import fused_swiglu_scale_clamp_bwd

            return fused_swiglu_scale_clamp_bwd(
                x, scale, out_grad, float(clamp_value)
            )
        from paddlefleet_ops import fused_swiglu_scale_bwd

        return fused_swiglu_scale_bwd(x, scale, out_grad)
    else:
        # ----------------------------
        # XPU / CPU fallback
        # ----------------------------
        hidden = x.shape[-1] // 2

        if clamp_value is not None and clamp_value > 0:
            # Cast to float32 for the backward pass
            x_fp32 = x.cast(paddle.float32)
            gate_fp32 = x_fp32[..., :hidden]
            val_fp32 = x_fp32[..., hidden:]

            # Clamp the raw inputs and build saturation masks matching g.dtype
            gate_raw = gate_fp32
            val_raw = val_fp32
            gate_fp32 = paddle.clip(gate_raw, max=clamp_value)
            val_fp32 = paddle.clip(val_raw, min=-clamp_value, max=clamp_value)
            g_mask = (gate_raw <= clamp_value).cast(out_grad.dtype)
            v_mask = (
                (val_raw <= clamp_value) & (val_raw >= -clamp_value)
            ).cast(out_grad.dtype)

            sig = F.sigmoid(gate_fp32)  # float32
            silu = gate_fp32 * sig  # float32
            swiglu_val = silu * val_fp32  # float32

            scale_fp32 = scale.cast(paddle.float32)
            scale_exp = _broadcast_scale(
                scale_fp32, paddle.float32, out_grad.ndim
            )
            d_u = out_grad * scale_exp

            # d_val (gradient w.r.t. value / second half)
            d_val = d_u * silu * v_mask

            # d_gate (gradient w.r.t. gate / first half)
            d_gate = (
                d_u * sig * (1.0 + gate_fp32 * (1.0 - sig)) * val_fp32 * g_mask
            )

            # Output order must be [d_gate, d_val] matching
            # chunk(x,2)=[gate,value]
            d_x = paddle.concat([d_gate, d_val], axis=-1).cast(x.dtype)

            # d_scale:
            #   sum(swiglu_val.cast(x.dtype) * out_grad.cast(scale_dtype))
            scale_dtype = scale.dtype
            d_scale = paddle.sum(
                swiglu_val.cast(x.dtype) * out_grad.cast(scale_dtype),
                axis=-1,
                keepdim=True,
            ).cast(scale_dtype)

            return d_x, d_scale

        # Original non-clamp logic
        gate = x[..., :hidden]
        val = x[..., hidden:]

        sig = F.sigmoid(gate).cast(x.dtype)
        silu = gate * sig
        swiglu_val = silu * val

        scale_exp = _broadcast_scale(scale, x.dtype, out_grad.ndim)
        d_u = out_grad * scale_exp

        # ----------------------------
        # dv
        # ----------------------------
        d_val = d_u * silu

        # ----------------------------
        # dg
        # ----------------------------
        d_gate = d_u * val * sig * (1.0 + gate * (1.0 - sig))

        # ----------------------------
        # d_x concat back
        # ----------------------------
        d_x = paddle.concat([d_gate, d_val], axis=-1).cast(x.dtype)

        # ----------------------------
        # d_scale
        # sum(dout * swiglu) over hidden dim
        # ----------------------------
        d_scale = paddle.sum(
            out_grad.cast(paddle.float32) * swiglu_val.cast(paddle.float32),
            axis=-1,
        ).cast(scale.dtype)

        return d_x, d_scale
