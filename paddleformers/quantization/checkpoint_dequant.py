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

from __future__ import annotations

import math
from typing import Protocol

import paddle

# -----------------------------------------------------------------------------
# Dequantizer contract
# -----------------------------------------------------------------------------


class CheckpointDequantizer(Protocol):
    method: str
    value_formats: frozenset[str]
    scale_formats: frozenset[str]
    value_format: str | None
    scale_format: str | None
    block_axes: tuple[int, ...] | None
    block_shape: tuple[int, ...] | None

    def configure_formats(self, value_format: str, scale_format: str) -> "CheckpointDequantizer":
        ...

    def configure_geometry(
        self,
        block_axes: tuple[int, ...],
        block_shape: tuple[int, ...],
    ) -> "CheckpointDequantizer":
        ...

    def logical_shape(self, physical_shape: tuple[int, ...]) -> tuple[int, ...]:
        ...

    def physical_qweight_shape(self, logical_shape: tuple[int, ...]) -> tuple[int, ...]:
        ...

    def physical_qweight_slice(
        self,
        global_offset: tuple[int, ...],
        local_shape: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        ...

    def logical_shard_is_aligned(
        self,
        global_shape: tuple[int, ...],
        local_shape: tuple[int, ...],
        global_offset: tuple[int, ...],
    ) -> bool:
        ...

    def dequantize(
        self,
        components: dict[str, paddle.Tensor],
        output_dtype: paddle.dtype,
    ) -> paddle.Tensor:
        ...


# -----------------------------------------------------------------------------
# Raw checkpoint decoding helpers
# -----------------------------------------------------------------------------


def _build_e4m3_lut() -> tuple[float, ...]:
    values = []
    for code in range(256):
        exponent = (code >> 3) & 0xF
        mantissa = code & 0x7
        if exponent == 0xF and mantissa == 0x7:
            values.append(float("nan"))
            continue
        magnitude = mantissa * 2.0**-9 if exponent == 0 else (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
        values.append(-magnitude if code & 0x80 else magnitude)
    return tuple(values)


_E4M3_LUT = _build_e4m3_lut()
_UE8M0_LUT = tuple(float("inf") if exponent == 255 else 2.0 ** (exponent - 127) for exponent in range(256))


def _dtype_name(tensor: paddle.Tensor) -> str:
    return str(tensor.dtype).split(".")[-1].lower()


def _as_uint8_codes(tensor: paddle.Tensor, component_name: str) -> paddle.Tensor:
    dtype = _dtype_name(tensor)
    if dtype not in {"uint8", "int8"}:
        raise TypeError(
            f"{component_name} must use raw uint8/int8 storage, got {tensor.dtype}. "
            "The Paddle safetensors reader must preserve unsupported 8-bit formats as raw bytes."
        )
    return tensor.astype("uint8").astype("int32")


def _decode_e4m3(tensor: paddle.Tensor) -> paddle.Tensor:
    dtype = _dtype_name(tensor)
    if "float8_e4m3" in dtype:
        return tensor.astype("float32")

    raw = _as_uint8_codes(tensor, "qweight")
    lut = paddle.to_tensor(_E4M3_LUT, dtype="float32", place=tensor.place)
    return paddle.gather(lut, raw.flatten()).reshape(raw.shape)


def _decode_ue8m0(tensor: paddle.Tensor) -> paddle.Tensor:
    if "float8_e8m0" in _dtype_name(tensor) or "e8m0" in _dtype_name(tensor):
        return tensor.astype("float32")
    raw = _as_uint8_codes(tensor, "scale")
    lut = paddle.to_tensor(_UE8M0_LUT, dtype="float32", place=tensor.place)
    return paddle.gather(lut, raw.flatten()).reshape(raw.shape)


def _decode_e2m1_packed(tensor: paddle.Tensor) -> paddle.Tensor:
    raw = _as_uint8_codes(tensor, "qweight")
    low = paddle.bitwise_and(raw, paddle.full_like(raw, 0xF))
    high = paddle.bitwise_and(paddle.bitwise_right_shift(raw, paddle.full_like(raw, 4)), paddle.full_like(raw, 0xF))
    codes = paddle.stack([low, high], axis=-1).reshape([*tensor.shape[:-1], tensor.shape[-1] * 2])
    lut = paddle.to_tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype="float32",
        place=tensor.place,
    )
    return paddle.gather(lut, codes.flatten()).reshape(codes.shape)


def _require_components(components: dict[str, paddle.Tensor], required: set[str]) -> None:
    missing = required - set(components)
    if missing:
        raise KeyError(f"Missing checkpoint quantization components: {sorted(missing)}.")


def _expand_scale_grid(
    scales: paddle.Tensor,
    logical_shape: tuple[int, ...],
    block_axes: tuple[int, ...],
    block_shape: tuple[int, ...],
) -> paddle.Tensor:
    if len(scales.shape) != len(logical_shape):
        raise ValueError(
            f"Scale rank must match logical weight rank: scale={tuple(scales.shape)}, logical={logical_shape}."
        )

    axis_to_block = dict(zip(block_axes, block_shape))
    expected_scale_shape = tuple(
        math.ceil(dim / axis_to_block[axis]) if axis in axis_to_block else dim
        for axis, dim in enumerate(logical_shape)
    )
    if tuple(scales.shape) != expected_scale_shape:
        raise ValueError(
            f"Invalid scale grid shape: expected {expected_scale_shape}, got {tuple(scales.shape)} "
            f"for logical shape {logical_shape}."
        )

    expanded = scales
    for axis, block_size in sorted(axis_to_block.items()):
        expanded = paddle.repeat_interleave(expanded, repeats=block_size, axis=axis)
    slices = tuple(slice(0, dim) for dim in logical_shape)
    return expanded[slices]


def _decode_scales(scale: paddle.Tensor, scale_format: str) -> paddle.Tensor:
    scale_format = scale_format.strip().lower()
    if scale_format in {"ue8m0", "e8m0", "float8_e8m0"}:
        return _decode_ue8m0(scale)
    if scale_format in {"float", "fp16", "fp32", "bf16", "float16", "float32", "bfloat16"}:
        return scale.astype("float32")
    raise ValueError(f"Unsupported scale_format {scale_format!r}.")


# -----------------------------------------------------------------------------
# Configured dequantizer implementations
# -----------------------------------------------------------------------------


class _ConfiguredCheckpointDequantizer:
    """Common configured state and geometry behavior for concrete methods."""

    method = ""
    value_formats = frozenset()
    scale_formats = frozenset()

    def __init__(
        self,
        value_format: str | None = None,
        scale_format: str | None = None,
        block_axes: tuple[int, ...] | None = None,
        block_shape: tuple[int, ...] | None = None,
    ):
        self.value_format = value_format
        self.scale_format = scale_format
        self.block_axes = block_axes
        self.block_shape = block_shape

    def configure_formats(self, value_format: str, scale_format: str) -> "_ConfiguredCheckpointDequantizer":
        value_format = value_format.strip().lower()
        scale_format = scale_format.strip().lower()
        if value_format not in self.value_formats:
            raise ValueError(f"Unsupported {self.method} value_format {value_format!r}.")
        if scale_format not in self.scale_formats:
            raise ValueError(f"Unsupported {self.method} scale_format {scale_format!r}.")
        return type(self)(value_format, scale_format, self.block_axes, self.block_shape)

    def configure_geometry(
        self,
        block_axes: tuple[int, ...],
        block_shape: tuple[int, ...],
    ) -> "_ConfiguredCheckpointDequantizer":
        if not block_axes or len(block_axes) != len(block_shape) or len(set(block_axes)) != len(block_axes):
            raise ValueError(f"Invalid block geometry for {self.method}.")
        if any(axis < 0 for axis in block_axes) or any(size <= 0 for size in block_shape):
            raise ValueError(f"Invalid block geometry for {self.method}.")
        return type(self)(self.value_format, self.scale_format, tuple(block_axes), tuple(block_shape))

    def _geometry(self, rank: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if self.block_axes is None or self.block_shape is None:
            raise ValueError(f"{self.method} requires block geometry.")
        axes = tuple(axis + rank if axis < 0 else axis for axis in self.block_axes)
        if any(axis >= rank for axis in axes):
            raise ValueError(f"Invalid block_axes {self.block_axes} for a rank-{rank} tensor.")
        return axes, self.block_shape

    def logical_shape(self, physical_shape: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(physical_shape)

    def physical_qweight_shape(self, logical_shape: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(logical_shape)

    def physical_qweight_slice(
        self,
        global_offset: tuple[int, ...],
        local_shape: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return tuple(global_offset), tuple(local_shape)

    def logical_shard_is_aligned(
        self,
        global_shape: tuple[int, ...],
        local_shape: tuple[int, ...],
        global_offset: tuple[int, ...],
    ) -> bool:
        axes, block_shape = self._geometry(len(global_shape))
        for axis, block_size in zip(axes, block_shape):
            start = global_offset[axis]
            end = start + local_shape[axis]
            if start % block_size != 0 and start != 0:
                return False
            if end % block_size != 0 and end != global_shape[axis]:
                return False
        return True

    def _scales(self, scale: paddle.Tensor, logical_shape: tuple[int, ...]) -> paddle.Tensor:
        axes, block_shape = self._geometry(len(logical_shape))
        return _expand_scale_grid(
            _decode_scales(scale, self.scale_format),
            logical_shape,
            axes,
            block_shape,
        )


class FP8BlockCheckpointDequantizer(_ConfiguredCheckpointDequantizer):
    method = "fp8_block"
    value_formats = frozenset({"e4m3", "fp8_e4m3", "float8_e4m3"})
    scale_formats = frozenset(
        {"ue8m0", "e8m0", "float8_e8m0", "float", "fp16", "fp32", "bf16", "float16", "float32", "bfloat16"}
    )

    def dequantize(
        self,
        components: dict[str, paddle.Tensor],
        output_dtype: paddle.dtype,
    ) -> paddle.Tensor:
        _require_components(components, {"qweight", "scale"})
        qweight = _decode_e4m3(components["qweight"])
        logical_shape = tuple(qweight.shape)
        return (qweight * self._scales(components["scale"], logical_shape)).astype(output_dtype)


class MXFP4GroupCheckpointDequantizer(_ConfiguredCheckpointDequantizer):
    method = "mxfp4_group"
    value_formats = frozenset({"e2m1", "fp4_e2m1"})
    scale_formats = frozenset(
        {"ue8m0", "e8m0", "float8_e8m0", "float", "fp16", "fp32", "bf16", "float16", "float32", "bfloat16"}
    )

    def dequantize(
        self,
        components: dict[str, paddle.Tensor],
        output_dtype: paddle.dtype,
    ) -> paddle.Tensor:
        _require_components(components, {"qweight", "scale"})
        qweight = _decode_e2m1_packed(components["qweight"])
        logical_shape = tuple(qweight.shape)
        return (qweight * self._scales(components["scale"], logical_shape)).astype(output_dtype)

    def logical_shape(self, physical_shape: tuple[int, ...]) -> tuple[int, ...]:
        if not physical_shape:
            raise ValueError(f"Invalid MXFP4 weight shape: {physical_shape}.")
        return (*physical_shape[:-1], physical_shape[-1] * 2)

    def physical_qweight_shape(self, logical_shape: tuple[int, ...]) -> tuple[int, ...]:
        if not logical_shape or logical_shape[-1] % 2:
            raise ValueError(f"Invalid MXFP4 logical weight shape: {logical_shape}.")
        return (*logical_shape[:-1], logical_shape[-1] // 2)

    def physical_qweight_slice(
        self,
        global_offset: tuple[int, ...],
        local_shape: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if global_offset[-1] % 2 or local_shape[-1] % 2:
            raise ValueError("MXFP4 qweight slices must have even last-axis offsets and sizes.")
        return (
            (*global_offset[:-1], global_offset[-1] // 2),
            (*local_shape[:-1], local_shape[-1] // 2),
        )

    def logical_shard_is_aligned(
        self,
        global_shape: tuple[int, ...],
        local_shape: tuple[int, ...],
        global_offset: tuple[int, ...],
    ) -> bool:
        if not super().logical_shard_is_aligned(global_shape, local_shape, global_offset):
            return False
        if global_offset[-1] % 2 or local_shape[-1] % 2:
            return False
        return True


# -----------------------------------------------------------------------------
# Dequantizer registry
#
# The registry stores unconfigured method prototypes. Descriptor validation
# obtains a prototype, configures its formats, and later configures its shared
# block geometry before storing it in the quantization metadata.
# -----------------------------------------------------------------------------

_CHECKPOINT_DEQUANTIZERS: dict[str, CheckpointDequantizer] = {}


def _normalize_method(method: str) -> str:
    if not isinstance(method, str):
        raise TypeError(f"Checkpoint quantization method must be a string, got {type(method).__name__}.")
    normalized = method.strip().lower()
    if not normalized:
        raise ValueError("Checkpoint quantization method must not be empty.")
    return normalized


def register_checkpoint_dequantizer(method: str, dequantizer: CheckpointDequantizer) -> None:
    method = _normalize_method(method)
    if method in _CHECKPOINT_DEQUANTIZERS:
        raise ValueError(f"Checkpoint dequantizer {method!r} is already registered.")
    if not callable(getattr(dequantizer, "dequantize", None)):
        raise TypeError("Checkpoint dequantizer must provide a callable dequantize() method.")
    _CHECKPOINT_DEQUANTIZERS[method] = dequantizer


def get_checkpoint_dequantizer(method: str) -> CheckpointDequantizer:
    method = _normalize_method(method)
    try:
        return _CHECKPOINT_DEQUANTIZERS[method]
    except KeyError as exc:
        supported = sorted(_CHECKPOINT_DEQUANTIZERS)
        raise ValueError(
            f"Unsupported checkpoint quantization method {method!r}; registered methods: {supported}."
        ) from exc


register_checkpoint_dequantizer("fp8_block", FP8BlockCheckpointDequantizer())
register_checkpoint_dequantizer("mxfp4_group", MXFP4GroupCheckpointDequantizer())
