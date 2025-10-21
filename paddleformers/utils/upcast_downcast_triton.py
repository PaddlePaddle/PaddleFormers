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
import paddle.nn.functional as F
import triton
import triton.language as tl

from paddleformers.utils.log import logger

try:
    import use_triton_in_paddle

    use_triton_in_paddle.make_triton_compatible_with_paddle()
except:
    raise RuntimeError(
        "Triton is installed, but not yet compatible with Paddle. "
        "Please run 'python -m pip install use-triton-in-paddle' to enable Triton support in Paddle."
    )
DTYPE_MAPPING = {
    paddle.bfloat16: tl.bfloat16,
    paddle.float32: tl.float32,
    paddle.float16: tl.float16,
}

MXFP_BLOCK_SIZE = tl.constexpr(32)


@triton.jit
def _get_max_quant_val(dtype: tl.constexpr):
    if dtype == tl.uint8:
        return 6.0
    elif dtype == tl.float8e5:
        return 57344.0
    elif dtype == tl.float8e4nv:
        return 448.0
    else:
        tl.static_assert(False, f"Invalid {dtype=}")


@triton.jit
def _compute_quant_and_scale(
    src_tensor, valid_src_mask, mx_tensor_dtype: tl.constexpr, DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr = 0
):
    is_fp8: tl.constexpr = mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    BLOCK_SIZE_OUT_DIM: tl.constexpr = src_tensor.shape[0]
    BLOCK_SIZE_QUANT_DIM: tl.constexpr = src_tensor.shape[1]
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = src_tensor.shape[1] // MXFP_BLOCK_SIZE

    # Explicit cast to fp32 since most ops are not supported on bfloat16. We avoid needless conversions to and from bf16
    f32_tensor = src_tensor.to(tl.float32)
    abs_tensor = tl.abs(f32_tensor)
    abs_tensor = tl.where(valid_src_mask, abs_tensor, -1.0)  # Don't consider padding tensors in scale computation
    abs_tensor = tl.reshape(abs_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE])
    max_val = tl.max(abs_tensor, axis=2, keep_dims=True)
    dequant_scale = max_val / _get_max_quant_val(mx_tensor_dtype)
    if DEQUANT_SCALE_ROUNDING_MODE == 0:
        # DequantScaleRoundingMode.ROUND_UP
        # compute 2 ** ceil(log2(dequant_scale))
        # Adding 0x007FFFFF adds exponent by 1 unless mantissa is all zeros
        # A corner case: exponent is 0xFF that will overflow but that's already
        # NaN so assume we don't care.
        dequant_scale_exponent = (dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000
    else:
        # DequantScaleRoundingMode.ROUND_DOWN
        # compute 2 ** floor(log2(dequant_scale))
        assert DEQUANT_SCALE_ROUNDING_MODE == 1
        dequant_scale_exponent = dequant_scale.to(tl.uint32, bitcast=True) & 0x7F800000
    dequant_scale_rounded = dequant_scale_exponent.to(tl.float32, bitcast=True)
    quant_scale = tl.where(dequant_scale_rounded == 0, 0, 1.0 / dequant_scale_rounded)

    f32_tensor = tl.reshape(f32_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE])
    quant_tensor = f32_tensor * quant_scale

    # Reshape the tensors after scaling
    quant_tensor = quant_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    # Set the invalid portions of the tensor to 0. This will ensure that any padding tensors are 0 in the mx format.
    quant_tensor = tl.where(valid_src_mask, quant_tensor, 0)
    dequant_scale_exponent = dequant_scale_exponent.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE])

    # First, we simply extract the exponent part of the scales and store the result
    dequant_scale_exponent = (dequant_scale_exponent >> 23).to(tl.uint8)
    # Now we must convert the tensors to the mx format.
    if is_fp8:
        out_tensor = quant_tensor.to(mx_tensor_dtype)
    else:
        quant_tensor = quant_tensor.to(tl.uint32, bitcast=True)
        signs = quant_tensor & 0x80000000
        exponents = (quant_tensor >> 23) & 0xFF
        mantissas = quant_tensor & 0x7FFFFF

        # For RTNE: 0.25 < x < 0.75 maps to 0.5 (denormal); exactly 0.25 maps to 0.0
        E8_BIAS = 127
        E2_BIAS = 1
        # Move implicit bit 1 at the beginning to mantissa for denormals
        adjusted_exponents = tl.core.sub(E8_BIAS, exponents + 1, sanitize_overflow=False)
        mantissas = tl.where(exponents < E8_BIAS, (0x400000 | (mantissas >> 1)) >> adjusted_exponents, mantissas)

        # For normal numbers, we change the bias from 127 to 1, and for subnormals, we keep exponent as 0.
        exponents = tl.maximum(exponents, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

        # Combine sign, exponent, and mantissa, while saturating
        # Round to nearest, ties to even (RTNE): use guard/sticky and LSB to decide increment
        m2bits = mantissas >> 21
        lsb_keep = (m2bits >> 1) & 0x1
        guard = m2bits & 0x1
        sticky = ((mantissas & 0x1FFFFF) != 0).to(tl.uint32)
        round_inc = guard & (sticky | lsb_keep)
        e2m1_tmp = tl.minimum((((exponents << 2) | m2bits) + round_inc) >> 1, 0x7)
        e2m1_value = ((signs >> 28) | e2m1_tmp).to(tl.uint8)

        e2m1_value = tl.reshape(e2m1_value, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2])
        evens, odds = tl.split(e2m1_value)
        out_tensor = evens | (odds << 4)

    return out_tensor, dequant_scale_exponent


@triton.jit
def _downcast_to_mxfp(
    mx_tensor_ptr,
    stride_mxt_outer,
    stride_mxt_quant: tl.constexpr,
    mx_scale_ptr,
    stride_mx_scale_outer,
    stride_mx_scale_quant,
    src_ptr,
    stride_src_outer,
    stride_src_quant,
    outer_dim,
    quant_dim,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
    DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr,
):

    tl.static_assert(stride_mxt_quant == 1, f"Output stride, {stride_mxt_quant=} must be 1.")
    tl.static_assert(BLOCK_SIZE_QUANT_DIM % MXFP_BLOCK_SIZE == 0, f"{BLOCK_SIZE_QUANT_DIM=} must be a multiple of 32")

    # uint8 signifies two fp4 e2m1 values packed into a single byte
    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    tl.static_assert(
        mx_tensor_dtype == tl.uint8 or (mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5),
        f"Invalid {mx_tensor_dtype=}. Must be uint8 or float8.",
    )

    src_dtype: tl.constexpr = src_ptr.dtype.element_ty
    tl.static_assert(mx_scale_ptr.dtype.element_ty == tl.uint8, f"{mx_scale_ptr.dtype.element_ty=} must be uint8")
    tl.static_assert(
        (src_dtype == tl.bfloat16) or (src_dtype == tl.float16) or (src_dtype == tl.float32),
        f"{src_dtype=} must be bfloat16 or float16 or float32",
    )
    is_fp4: tl.constexpr = mx_tensor_dtype == tl.uint8

    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    K_DIVISOR: tl.constexpr = 2 if is_fp4 else 1
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // MXFP_BLOCK_SIZE
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR

    start_src_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_mx_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    src_ptr += start_src_quant * stride_src_quant + start_out * stride_src_outer
    mx_scale_ptr += start_mx_scale_quant * stride_mx_scale_quant + start_out * stride_mx_scale_outer
    mx_tensor_ptr += start_mx_quant * stride_mxt_quant + start_out * stride_mxt_outer

    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_mxt_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_scale_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)

    mask_src_quant = start_src_quant + offs_src_quant < quant_dim
    mask_n = start_out + offs_outer < outer_dim
    full_mask_src = mask_src_quant & mask_n

    mask_mxt_quant = start_mx_quant + offs_mxt_quant < tl.cdiv(quant_dim, K_DIVISOR)
    full_mask_mxt = mask_mxt_quant & mask_n

    scale_mask_k = start_mx_scale_quant + offs_scale_quant < tl.cdiv(quant_dim, MXFP_BLOCK_SIZE)
    full_scale_mask = scale_mask_k & mask_n

    src_tensor_offsets = offs_src_quant * stride_src_quant + offs_outer * stride_src_outer
    mx_scale_offsets = offs_scale_quant * stride_mx_scale_quant + offs_outer * stride_mx_scale_outer
    mx_tensor_offsets = offs_mxt_quant * stride_mxt_quant + offs_outer * stride_mxt_outer
    src_tensor = tl.load(src_ptr + src_tensor_offsets, mask=full_mask_src)

    out_tensor, scale_tensor = _compute_quant_and_scale(
        src_tensor, full_mask_src, mx_tensor_dtype, DEQUANT_SCALE_ROUNDING_MODE
    )

    tl.store(mx_scale_ptr + mx_scale_offsets, scale_tensor, mask=full_scale_mask)
    tl.store(mx_tensor_ptr + mx_tensor_offsets, out_tensor, mask=full_mask_mxt)


@triton.jit(repr=lambda _: "_dequantize_mxfp8")
def _quantize_mxfp8_fn(input, mask, pid=None):
    return _compute_quant_and_scale(input, mask, tl.float8e4nv)


@triton.jit
def _upcast_from_mxfp(
    out_ptr,
    stride_o_outer,
    stride_o_quant: tl.constexpr,
    mx_scale_ptr,
    stride_scale_outer,
    stride_scale_quant,
    mx_tensor_ptr,
    stride_tensor_outer,
    stride_tensor_quant: tl.constexpr,
    outer_dim,
    quant_dim,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
):

    tl.static_assert(stride_o_quant == 1, "the weight must be contiguous in the k dimension for mx")
    tl.static_assert(BLOCK_SIZE_QUANT_DIM % MXFP_BLOCK_SIZE == 0, "BLOCK_SIZE_K must be a multiple of 32")
    # uint8 signifies two fp4 e2m1 values packed into a single byte
    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    dst_dtype: tl.constexpr = out_ptr.dtype.element_ty
    tl.static_assert(dst_dtype == tl.float16 or dst_dtype == tl.bfloat16 or dst_dtype == tl.float32)
    tl.static_assert(
        mx_tensor_dtype == tl.uint8
        or ((mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5) or mx_tensor_dtype == dst_dtype),
        "mx_tensor_ptr must be uint8 or float8 or dst_dtype",
    )
    tl.static_assert(mx_scale_ptr.dtype.element_ty == tl.uint8, "mx_scale_ptr must be uint8")

    # Determine if we are dealing with fp8 types.
    is_fp4: tl.constexpr = mx_tensor_dtype == tl.uint8
    is_fp8: tl.constexpr = mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    K_DIVISOR: tl.constexpr = 2 if is_fp4 else 1
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // MXFP_BLOCK_SIZE
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR

    # Compute starting indices for the quantized (packed) dimension and the outer dimension.
    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    start_mxt_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    mx_tensor_ptr += start_mxt_quant * stride_tensor_quant + start_out * stride_tensor_outer
    mx_scale_ptr += start_mx_scale_quant * stride_scale_quant + start_out * stride_scale_outer
    out_ptr += start_out * stride_o_outer + start_out_quant * stride_o_quant

    # Compute offsets and masks.
    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_out_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)
    offs_scale = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)

    mask_outer = start_out + offs_outer < outer_dim
    mask_out_quant = start_out_quant + offs_out_quant < quant_dim
    full_mask_out = mask_out_quant & mask_outer

    mask_src_quant = start_mxt_quant + offs_src_quant < tl.cdiv(quant_dim, K_DIVISOR)
    full_mask_src = mask_src_quant & mask_outer

    mask_scale = start_mx_scale_quant + offs_scale < tl.cdiv(quant_dim, MXFP_BLOCK_SIZE)
    full_scale_mask = mask_scale & mask_outer

    tensor_offsets = offs_src_quant * stride_tensor_quant + offs_outer * stride_tensor_outer
    scale_offsets = offs_scale * stride_scale_quant + offs_outer * stride_scale_outer
    out_offsets = offs_out_quant * stride_o_quant + offs_outer * stride_o_outer

    # Load the packed tensor and scale.
    tensor = tl.load(mx_tensor_ptr + tensor_offsets, mask=full_mask_src)
    scale = tl.load(mx_scale_ptr + scale_offsets, mask=full_scale_mask)

    # Upcast the scale to the destination type.
    if dst_dtype == tl.bfloat16:
        dst_scale = (scale.to(tl.uint16) << 7).to(dst_dtype, bitcast=True)
    else:
        dst_scale = (scale.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        if dst_dtype == tl.float16:
            dst_scale = dst_scale.to(tl.float16)

    # Now upcast the tensor.
    intermediate_dtype: tl.constexpr = tl.bfloat16 if dst_dtype == tl.float32 else dst_dtype
    if is_fp8:
        dst_tensor = tensor.to(intermediate_dtype)
        if tensor.dtype == tl.float8e5:
            from_e_bits: tl.constexpr = 5
            from_m_bits: tl.constexpr = 2
            to_e_bits: tl.constexpr = 8 if intermediate_dtype == tl.bfloat16 else 5
            to_m_bits: tl.constexpr = 7 if intermediate_dtype == tl.bfloat16 else 10

            # Preserve infs and nans. FIXME Fp8E5M2_to_Bf16 doesn't preserve them!
            non_finite_mask_src: tl.constexpr = ((1 << from_e_bits) - 1) << from_m_bits
            non_finite_mask_dst: tl.constexpr = ((1 << to_e_bits) - 1) << to_m_bits
            dst_tensor = tl.where(
                (tensor.to(tl.uint8, bitcast=True) & non_finite_mask_src) == non_finite_mask_src,
                (dst_tensor.to(tl.uint16, bitcast=True) | non_finite_mask_dst).to(intermediate_dtype, bitcast=True),
                dst_tensor,
            )
    else:
        assert is_fp4
        dst_bias: tl.constexpr = 127 if intermediate_dtype == tl.bfloat16 else 15
        dst_0p5: tl.constexpr = 16128 if intermediate_dtype == tl.bfloat16 else 0x3800
        dst_m_bits: tl.constexpr = 7 if intermediate_dtype == tl.bfloat16 else 10
        # e2m1
        em0 = tensor & 0x07
        em1 = tensor & 0x70
        x0 = (em0.to(tl.uint16) << (dst_m_bits - 1)) | ((tensor & 0x08).to(tl.uint16) << 12)
        x1 = (em1.to(tl.uint16) << (dst_m_bits - 5)) | ((tensor & 0x80).to(tl.uint16) << 8)
        # Three cases:
        # 1) x is normal and non-zero: Correct bias
        x0 = tl.where((em0 & 0x06) != 0, x0 + ((dst_bias - 1) << dst_m_bits), x0)
        x1 = tl.where((em1 & 0x60) != 0, x1 + ((dst_bias - 1) << dst_m_bits), x1)
        # 2) x is subnormal (x == 0bs001 where s is the sign): Map to +-0.5 in the dst type
        x0 = tl.where(em0 == 0x01, dst_0p5 | (x0 & 0x8000), x0)
        x1 = tl.where(em1 == 0x10, dst_0p5 | (x1 & 0x8000), x1)
        # 3) x is zero, do nothing
        dst_tensor = tl.interleave(x0, x1).to(intermediate_dtype, bitcast=True)
    dst_tensor = dst_tensor.to(dst_dtype)

    # Reshape for proper broadcasting: the scale was stored with a 32‐sized “inner” grouping.
    dst_tensor = dst_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE])
    dst_scale = dst_scale.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 1])
    scale = scale.reshape(dst_scale.shape)

    out_tensor = dst_tensor * dst_scale
    # Correct any NaNs encoded via the scale.
    out_tensor = tl.where(scale == 0xFF, float("nan"), out_tensor)
    out_tensor = out_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    tl.store(out_ptr + out_offsets, out_tensor, mask=full_mask_out)


# -----------------------------------------------------------------------------
#                      Dequantization / Quantization Utilities
# -----------------------------------------------------------------------------


class DequantScaleRoundingMode:
    ROUND_UP = 0
    ROUND_DOWN = 1


def downcast_to_mxfp(
    src_tensor: paddle.Tensor,
    out_quant_type: paddle.dtype,
    axis: int,
    DEQUANT_SCALE_ROUNDING_MODE: DequantScaleRoundingMode = DequantScaleRoundingMode.ROUND_UP,
):
    """
    Convert the src weights to mx format. The src weight is quantized along the axis dimension.

    If weight_quant_type is paddle.uint8, we output mxfp4 where two e2m1 values are packed into a single byte.
    Note that this means the k_dim of the tensor will be half of the logical k_dim.

    If weight_quant_type is paddle.float8_e4m3fn or paddle.float8_e5m2, we output mxfp8 with the float8s are stored
    in their respective formats.
    """
    ndim = src_tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim
    # downcast
    src_tensor = src_tensor.transpose(axis, src_tensor.ndim - 1)
    is_fp4 = out_quant_type == paddle.uint8
    is_fp8 = out_quant_type in (paddle.float8_e4m3fn, paddle.float8_e5m2)
    assert is_fp4 or is_fp8
    divisor = 2 if is_fp4 else 1
    L = src_tensor.shape[-1]
    if is_fp4:
        assert L % 2 == 0, f"axis dim must be divisible by 2 for e2m1. Got {L}"
    out_shape = src_tensor.shape[:-1] + (L // divisor,)
    out_scale_shape = src_tensor.shape[:-1] + (triton.cdiv(L, MXFP_BLOCK_SIZE),)

    out_quant_tensor = src_tensor.new_empty(out_shape, dtype=out_quant_type)
    out_scale = src_tensor.new_empty(out_scale_shape, dtype=paddle.uint8)

    if src_tensor.numel() > 0:
        kernel_src_tensor = src_tensor.reshape(-1, src_tensor.shape[-1])
        kernel_quant_tensor = out_quant_tensor.view(-1, out_quant_tensor.shape[-1])
        kernel_scale = out_scale.view(-1, out_scale.shape[-1])

        BLOCK_OUT_DIM = 128
        BLOCK_QUANT_DIM = MXFP_BLOCK_SIZE.value
        grid_out = triton.cdiv(kernel_src_tensor.shape[0], BLOCK_OUT_DIM)
        grid_quant = triton.cdiv(kernel_src_tensor.shape[1], BLOCK_QUANT_DIM)

        _downcast_to_mxfp[(grid_out, grid_quant)](
            kernel_quant_tensor,
            *kernel_quant_tensor.stride(),
            kernel_scale,
            *kernel_scale.stride(),
            kernel_src_tensor,
            *kernel_src_tensor.stride(),
            *kernel_src_tensor.shape,
            BLOCK_OUT_DIM,
            BLOCK_QUANT_DIM,
            DEQUANT_SCALE_ROUNDING_MODE.value,
            num_warps=8,
        )

    out_quant_tensor = out_quant_tensor.transpose(axis, src_tensor.ndim - 1)
    out_scale = out_scale.transpose(axis, src_tensor.ndim - 1)
    return out_quant_tensor, out_scale


def upcast_from_mxfp(tensor: paddle.Tensor, scale: paddle.Tensor, target_dtype: paddle.dtype, axis: int):
    """
    Upcasts an mxfp (packed) weight tensor back to float16 or bfloat16.

    The function assumes that the tensors were quantized along the given axis.
    It permutes the tensor so that the quantized axis is last, reshapes to 2D,
    launches the Triton upcast kernel, and then unpermutes back to the original order.
    """
    ndim = tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim
    assert tensor.ndim == scale.ndim, (
        f"Weight and scale must have the same number of dimensions. " f"Got {tensor.ndim=} and {scale.ndim=}"
    )
    # dtype checks
    assert tensor.dtype in {
        paddle.uint8,
        paddle.float8_e5m2,
        paddle.float8_e4m3fn,
    }, f"Invalid tensor dtype {tensor.dtype=}"
    assert scale.dtype == paddle.uint8, f"Invalid scale dtype {scale.dtype=}"
    assert target_dtype in (paddle.float16, paddle.bfloat16, paddle.float32), f"Invalid output dtype {target_dtype=}"
    # upcast
    logical_quant_dim = tensor.shape[axis] * (2 if tensor.dtype == paddle.uint8 else 1)
    tensor = tensor.transpose(axis, tensor.ndim - 1).contiguous()
    scale = scale.transpose(axis, scale.ndim - 1).contiguous()
    out = paddle.empty((*tensor.shape[:-1], logical_quant_dim), dtype=target_dtype, device=tensor.device)

    if tensor.numel() > 0:
        reshaped_out = out.view(-1, out.shape[-1])
        reshaped_tensor = tensor.view(-1, tensor.shape[-1])
        reshaped_scale = scale.view(-1, scale.shape[-1])
        BLOCK_OUT_DIM = 128
        BLOCK_QUANT_DIM = MXFP_BLOCK_SIZE.value
        blocks_out_dim = triton.cdiv(reshaped_out.shape[0], BLOCK_OUT_DIM)
        blocks_quant_dim = triton.cdiv(reshaped_out.shape[1], BLOCK_QUANT_DIM)
        _upcast_from_mxfp[(blocks_out_dim, blocks_quant_dim)](
            reshaped_out,
            *reshaped_out.stride(),
            reshaped_scale,
            *reshaped_scale.stride(),
            reshaped_tensor,
            *reshaped_tensor.stride(),
            *reshaped_out.shape,
            BLOCK_OUT_DIM,
            BLOCK_QUANT_DIM,
            num_warps=8,
        )
    out = out.transpose(axis, scale.ndim - 1).contiguous()
    return out


# ------------


def right_shift_unsigned(x, shift):
    # CUDA paddle does not support bit ops on uint32, so we need to mask to get unsigned right shift
    return (x >> shift) & paddle.to_tensor(((1 << (32 - shift)) - 1), dtype=paddle.int32)


def get_max_quant_val(dtype: paddle.dtype):
    d = {paddle.uint8: 6.0, paddle.float8_e5m2: 57344.0, paddle.float8_e4m3fn: 448.0}
    assert dtype in d
    return d[dtype]


def downcast_to_mxfp_paddle(
    src_tensor: paddle.Tensor,
    out_quant_type: paddle.dtype,
    axis: int,
    DEQUANT_SCALE_ROUNDING_MODE: DequantScaleRoundingMode = DequantScaleRoundingMode.ROUND_UP,
):
    """
    Converts the src tensor to the output format specified by out_quant_type.
      axis: The axis along which the tensors are contiguous and quantization is applied.
      DEQUANT_SCALE_ROUNDING_MODE: 0 for ROUND_UP, 1 for ROUND_DOWN.

    Returns:
      out_quant_tensor: Quantized tensor in mx format.
         • For mxfp8, the output has the same shape as src_tensor.
         • For mxfp4, the size along the axis is halved, and the tensor is returned as a paddle.uint8.
      scale: Scale tensor (stored as uint8) computed per group of 32 elements along the axis.
             Its shape is the same as src_tensor except that the axis is replaced by ceil(L/32),
             where L is the original length along that axis.
    """
    # This should probably be packed into its own tiny class
    ndim = src_tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    assert src_tensor.dtype in {
        paddle.float32,
        paddle.bfloat16,
        paddle.float16,
    }, f"Invalid input tensor dtype {src_tensor.dtype}"

    axis = axis if axis >= 0 else axis + ndim
    is_fp4 = out_quant_type == paddle.uint8
    is_fp8 = "float8" in str(out_quant_type)
    assert is_fp4 or is_fp8, f"Invalid input tensor dtype {out_quant_type}"

    place = src_tensor.place

    # For mxfp4 conversion, we assume the contiguous axis length is even.
    if is_fp4:
        axis_shape = src_tensor.shape[axis]
        assert axis_shape % 2 == 0, "For mxfp4 conversion the contiguous axis length must be even."

    # Permute the tensor so that the contiguous axis becomes the last dimension.
    src = src_tensor.transpose(axis, src_tensor.ndim - 1).to(paddle.float32)
    axis_shape = src.shape[-1]

    # Pad the axis to be divisible by 32, in case it is not.
    next_multiple = triton.cdiv(axis_shape, MXFP_BLOCK_SIZE) * MXFP_BLOCK_SIZE
    pad_amount = next_multiple - axis_shape
    padded_src = F.pad(src, (0, int(pad_amount)))
    valid_mask = F.pad(paddle.ones_like(src), (0, int(pad_amount))).to(paddle.bool)
    padded_axis_shape = padded_src.shape[-1]  # now divisible by 32

    # --- Compute per-group maximums for scale ---
    # Set padded entries to -1 so they don’t affect the max.
    abs_f = paddle.abs(padded_src)
    abs_f = paddle.where(valid_mask, abs_f, paddle.to_tensor(-1.0, place=place, dtype=padded_src.dtype))
    # Reshape the last dimension into groups of 32.
    new_shape = padded_src.shape[:-1] + [int(padded_axis_shape // MXFP_BLOCK_SIZE), int(MXFP_BLOCK_SIZE)]
    abs_groups = abs_f.view(*new_shape)
    # Compute maximum along the group dimension (of size 32).
    max_val = abs_groups.max(axis=-1, keepdim=True)

    # Choose a max quantization value depending on type.
    max_quant_val = get_max_quant_val(out_quant_type)
    dequant_scale = max_val / max_quant_val  # shape: (..., padded_axis_shape//32, 1)

    # Convert to int to round the FP32 scale, prior to quantization!
    ds_int = dequant_scale.view(paddle.int32)
    if DEQUANT_SCALE_ROUNDING_MODE == DequantScaleRoundingMode.ROUND_UP:
        ds_int_rounded = (ds_int + paddle.to_tensor(0x007FFFFF, dtype=paddle.int32)) & paddle.to_tensor(
            0x7F800000, dtype=paddle.int32
        )
    else:
        ds_int_rounded = ds_int & paddle.to_tensor(0x7F800000, dtype=paddle.int32)
    # Reinterpret back as float32.
    dequant_scale_rounded = ds_int_rounded.view(paddle.float32)

    # Compute the quantization scale.
    quant_scale = paddle.where(
        dequant_scale_rounded == 0, paddle.to_tensor(0.0, place=place), 1.0 / dequant_scale_rounded
    )

    # Quantize the tensor
    orig_padded_shape = padded_src.shape
    padded_src_groups = padded_src.view(*new_shape)
    quant_tensor = padded_src_groups * quant_scale
    # Reshape back to the original shape and trim padding
    quant_tensor = quant_tensor.view(orig_padded_shape)
    quant_tensor = quant_tensor[..., :axis_shape]

    # Finally, convert the quantized tensor to the target format
    if is_fp8:
        # Conversion must use satfinite PTX, so clamp before the conversion in paddle to emulate this behavior
        quant_tensor = paddle.clamp(quant_tensor, -max_quant_val, max_quant_val)
        out_weight = quant_tensor.to(out_quant_type)
    else:
        assert is_fp4, f"Invalid output quantization type {out_quant_type}"
        # For mxfp4, perform bit-level manipulation and pack two 4-bit values per uint8.
        # First, reinterpret the quantized tensor bits.
        q_int = quant_tensor.contiguous().view(paddle.int32)
        # Extract sign, exponent, and mantissa.
        signs = q_int & paddle.to_tensor(0x80000000, dtype=paddle.int32)
        exponents = right_shift_unsigned(q_int, 23) & paddle.to_tensor(0xFF, dtype=paddle.int32)
        mantissas = q_int & paddle.to_tensor(0x7FFFFF, dtype=paddle.int32)

        E8_BIAS = 127
        E2_BIAS = 1
        # Adjust mantissas for subnormals.
        mantissas = paddle.where(
            exponents < E8_BIAS,
            (paddle.to_tensor(0x400000, dtype=paddle.int32) | right_shift_unsigned(mantissas, 1))
            >> (E8_BIAS - exponents - 1),
            mantissas,
        )
        exponents = paddle.maximum(exponents, paddle.to_tensor(E8_BIAS - E2_BIAS, place=place, dtype=paddle.int32)) - (
            E8_BIAS - E2_BIAS
        )
        # Round to nearest, ties to even (RTNE)
        m2bits = right_shift_unsigned(mantissas, 21) & paddle.to_tensor(0x3, dtype=paddle.int32)
        lsb_keep = right_shift_unsigned(m2bits, 1) & paddle.to_tensor(0x1, dtype=paddle.int32)
        guard = m2bits & paddle.to_tensor(0x1, dtype=paddle.int32)
        sticky = (mantissas & paddle.to_tensor(((1 << 21) - 1), dtype=paddle.int32)) != 0
        round_inc = guard & (sticky.to(paddle.int32) | lsb_keep)
        e2m1_tmp = right_shift_unsigned(((exponents << 2) | m2bits) + round_inc, 1)
        e2m1_tmp = paddle.minimum(e2m1_tmp, paddle.to_tensor(0x7, place=place, dtype=paddle.int32))
        e2m1_value = (right_shift_unsigned(signs, 28) | e2m1_tmp).to(paddle.uint8)  # shape: (..., even_axis_shape)

        # Pack pairs of 4-bit values along the last dimension.
        e2m1_value = e2m1_value.view(*e2m1_value.shape[:-1], axis_shape // 2, 2)
        evens = e2m1_value[..., 0]
        odds = e2m1_value[..., 1]
        out_weight = evens | (odds << 4)  # shape: (..., axis_shape//2)

    # --- Process and output the scale ---
    dq_scale = (ds_int_rounded.view(*dequant_scale.shape) >> 23).to(paddle.uint8)  # shape: (..., axis_shape//32, 1)
    dq_scale = dq_scale.squeeze(-1)
    out_weight = out_weight.transpose(axis, src_tensor.ndim - 1)
    dq_scale = dq_scale.transpose(axis, src_tensor.ndim - 1)
    return out_weight, dq_scale


def cvt_e2m1_to_fp32(input_tensor):
    assert input_tensor.dtype == paddle.uint8

    input_tensor = input_tensor.to(paddle.int32)
    evens = input_tensor & paddle.to_tensor(0xF, paddle.int32)
    odds = (input_tensor >> 4) & paddle.to_tensor(0xF, paddle.int32)

    vals = [0.0, 0.5, 1, 1.5, 2, 3, 4, 6]
    outputs = paddle.to_tensor(vals, dtype=paddle.float32, place=input_tensor.place)
    outputs = paddle.cat([outputs, -outputs])

    even_floats = outputs[evens]
    odd_floats = outputs[odds]
    output_tensor = paddle.stack([even_floats, odd_floats], dim=-1)
    output_tensor = output_tensor.view(*input_tensor.shape[:-1], input_tensor.shape[-1] * 2)
    return output_tensor


def upcast_from_mxfp_paddle(tensor: paddle.Tensor, scale: paddle.Tensor, target_dtype: paddle.dtype, axis: int):
    """
    Converts the mxfp4/mxfp8 tensor to the target format specified by target_dtype.
      axis: The axis along which dequantization is applied.

    Returns:
      out_weight: Tensor in the target format.
    """

    ndim = tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    is_fp8 = tensor.dtype == paddle.float8_e4m3fn or tensor.dtype == paddle.float8_e5m2
    assert is_fp8 or tensor.dtype == paddle.uint8, f"Invalid input quantization type {tensor.dtype}"

    # Permute the tensor and scale so that the quantization axis becomes the last dimension
    axis = axis if axis >= 0 else axis + ndim
    scale = scale.transpose(axis, scale.ndim - 1)
    tensor = tensor.transpose(axis, tensor.ndim - 1)

    dq_scale = (scale.to(paddle.int32) << 23).view(paddle.float32)  # Shift to the exponent and bitcast to fp32
    if tensor.dtype == paddle.uint8:
        fp32_tensor = cvt_e2m1_to_fp32(tensor)
    else:
        fp32_tensor = tensor.to(paddle.float32)
    logical_quant_dim = tensor.shape[-1] * (2 if tensor.dtype == paddle.uint8 else 1)
    axis_shape = fp32_tensor.shape[-1]
    padded_axis_shape = triton.cdiv(logical_quant_dim, MXFP_BLOCK_SIZE) * MXFP_BLOCK_SIZE
    pad_size = padded_axis_shape - axis_shape
    padded_tensor = F.pad(fp32_tensor, (0, int(pad_size)))
    new_axis_shape = padded_tensor.shape[-1]
    new_shape = padded_tensor.shape[:-1] + [int(new_axis_shape // MXFP_BLOCK_SIZE), int(MXFP_BLOCK_SIZE)]

    padded_tensor = padded_tensor.view(*new_shape)
    dq_scale_padded = dq_scale.view(*new_shape[:-1])
    dq_scale_padded = dq_scale_padded.unsqueeze(-1)  # shape: [..., ceil(axis_shape/32), 1]
    out_padded = padded_tensor * dq_scale_padded
    # Flatten back and remove the padded tail
    out_padded = out_padded.view(*padded_tensor.shape[:-2], new_axis_shape)
    out_tensor = out_padded[..., :axis_shape]
    out_tensor = out_tensor.to(target_dtype).contiguous()
    out_tensor = out_tensor.transpose(axis, tensor.ndim - 1)

    return out_tensor


FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def convert_moe_packed_tensors(
    blocks,
    scales,
    *,
    dtype: paddle.dtype = paddle.bfloat16,
    rows_per_chunk: int = 32768 * 1024,  # TODO these values are not here by mistake ;)
) -> paddle.Tensor:
    """
    Convert the mxfp4 weights again, dequantizing and makes them compatible with the forward
    pass of GPT_OSS.
    """
    import math

    # # Check if blocks and scales are on CPU, and move to GPU if so
    # if not blocks.is_cuda and paddle.cuda.is_available():
    #     blocks = blocks.cuda()
    #     scales = scales.cuda()

    scales = scales.to(paddle.int32) - 127  # TODO that's because 128=2**7

    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape[:-1]=} does not match {scales.shape=}"

    lut = paddle.to_tensor(FP4_VALUES, dtype=dtype, place=blocks.place)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = paddle.empty(rows_total, B * 2, dtype=dtype)
    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)
        blk = blocks[r0:r1]
        exp = scales[r0:r1]

        # nibble indices -> int64
        idx_lo = (blk & paddle.to_tensor(0x0F, dtype=paddle.uint8)).to(paddle.int64)
        idx_hi = (blk >> 4).to(paddle.int64)
        sub = out[r0:r1]

        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]

        out[r0:r1] = paddle.ldexp(sub, exp)
        del idx_lo, idx_hi, blk, exp, sub

    out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
    del blocks, scales, lut
    return out.transpose(1, 2).contiguous()


def upcast_dict(param_origin_dict):
    logger.info(f"upcast len of origin dict is : {len(param_origin_dict)}")
    remove_list = []
    param_new_dict = {}

    for key, block_value in param_origin_dict.items():
        if key.endswith("blocks"):
            scale_key = key
            scale_key = scale_key.replace("blocks", "scales")
            assert scale_key in param_origin_dict.keys(), f"{scale_key} not in param_origin_dict.keys()"

            scale_value = param_origin_dict[scale_key]

            bf16_key = key[: -len("_blocks")]
            # bf16_value = upcast_from_mxfp_paddle(block_value, scale_value, paddle.uint8, axis=1)
            bf16_value = convert_moe_packed_tensors(block_value, scale_value)
            param_new_dict[bf16_key] = bf16_value
            remove_list.append(scale_key)
            remove_list.append(key)

    for key in remove_list:
        param_origin_dict.pop(key)

    param_origin_dict.update(param_new_dict)
    logger.info(f"upcast len of new dict is : {len(param_origin_dict)}")


FP4_LIST = ["down_proj", "gate_up_proj"]


def endswith(key, list):
    for suffix in list:
        if key.endswith(suffix):
            return True
    return False


def downcast_dict(param_origin_dict):
    logger.info(f"downcast len of origin dict is : {len(param_origin_dict)}")
    remove_list = []
    param_new_dict = {}

    for key, value in param_origin_dict.items():
        if endswith(key, FP4_LIST):
            block_key = key + "_blocks"
            scale_key = key + "_scales"

            fp4_blocks, fp4_scales = downcast_to_mxfp_paddle(value, paddle.uint8, axis=1)
            fp4_scales.transpose_([0, 2, 1])
            fp4_blocks.transpose_([0, 2, 1])
            origin_scale_shape = fp4_scales.shape
            origin_block_shape = fp4_blocks.shape
            fp4_blocks.reshape_(
                [
                    origin_block_shape[0],
                    origin_block_shape[1],
                    origin_scale_shape[2],
                    origin_block_shape[2] // origin_scale_shape[2],
                ]
            )
            param_new_dict[block_key] = fp4_blocks
            param_new_dict[scale_key] = fp4_scales
            remove_list.append(key)

    for key in remove_list:
        param_origin_dict.pop(key)

    param_origin_dict.update(param_new_dict)
    logger.info(f"downcast len of new dict is : {len(param_origin_dict)}")
