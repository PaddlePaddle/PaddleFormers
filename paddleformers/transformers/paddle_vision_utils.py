# coding=utf-8
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
"""
API for image and video processing, serving as a backend for PaddlePaddle processors.
"""

from typing import Optional, Union

import paddle
from paddle.nn.functional import interpolate
from paddle.nn.functional import pad as paddle_pad


def _pad_symmetric(img: paddle.Tensor, padding: list[int]) -> paddle.Tensor:
    # padding is left, right, top, bottom

    # crop if needed
    if padding[0] < 0 or padding[1] < 0 or padding[2] < 0 or padding[3] < 0:
        neg_min_padding = [-min(x, 0) for x in padding]
        crop_left, crop_right, crop_top, crop_bottom = neg_min_padding
        img = img[..., crop_top : img.shape[-2] - crop_bottom, crop_left : img.shape[-1] - crop_right]
        padding = [max(x, 0) for x in padding]

    in_sizes = img.size()

    _x_indices = [i for i in range(in_sizes[-1])]  # [0, 1, 2, 3, ...]
    left_indices = [i for i in range(padding[0] - 1, -1, -1)]  # e.g. [3, 2, 1, 0]
    right_indices = [-(i + 1) for i in range(padding[1])]  # e.g. [-1, -2, -3]
    x_indices = paddle.to_tensor(left_indices + _x_indices + right_indices).contiguous()

    _y_indices = [i for i in range(in_sizes[-2])]
    top_indices = [i for i in range(padding[2] - 1, -1, -1)]
    bottom_indices = [-(i + 1) for i in range(padding[3])]
    y_indices = paddle.to_tensor(top_indices + _y_indices + bottom_indices).contiguous()

    ndim = img.ndim
    if ndim == 3:
        return img[:, y_indices[:, None], x_indices[None, :]]
    elif ndim == 4:
        return img[:, :, y_indices[:, None], x_indices[None, :]]
    else:
        raise RuntimeError("Symmetric padding of N-D tensors are not supported yet")


def _compute_resized_output_size(
    image_size: tuple[int, int],
    size: Optional[list[int]],
    max_size: Optional[int] = None,
    allow_size_none: bool = False,
) -> list[int]:
    h, w = image_size
    short, long = (w, h) if w <= h else (h, w)
    if size is None:
        if not allow_size_none:
            raise ValueError("This should never happen!!")
        if not isinstance(max_size, int):
            raise ValueError(f"max_size must be an integer when size is None, but got {max_size} instead.")
        new_short, new_long = int(max_size * short / long), max_size
        new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    elif len(size) == 1:
        requested_new_short = size if isinstance(size, int) else size[0]
        new_short, new_long = requested_new_short, int(requested_new_short * long / short)

        if max_size is not None:
            if max_size <= requested_new_short:
                raise ValueError(
                    f"max_size = {max_size} must be strictly greater than the requested "
                    f"size for the smaller edge size = {size}"
                )
            if new_long > max_size:
                new_short, new_long = int(max_size * new_short / new_long), max_size

        new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    else:  # specified both h and w
        new_w, new_h = size[1], size[0]
    return [new_h, new_w]


def _should_use_native_uint8_kernel(interpolation):
    if interpolation == "bilinear" and paddle.in_dynamic_mode():
        return True

    return interpolation == "bicubic"


def resize(
    image: paddle.Tensor,
    size: Optional[list[int]],
    interpolation: Optional[str] = "bilinear",
    max_size: Optional[int] = None,
    align_corners: Optional[bool] = None,
) -> paddle.Tensor:
    if interpolation == "bilinear" or interpolation == "bicubic":
        align_corners = False

    shape = image.shape
    numel = image.size
    num_channels, old_height, old_width = shape[-3:]
    new_height, new_width = _compute_resized_output_size((old_height, old_width), size=size, max_size=max_size)

    if (new_height, new_width) == (old_height, old_width):
        return image
    elif numel > 0:
        dtype = image.dtype
        acceptable_dtypes = [paddle.float32, paddle.float64]
        if interpolation == "nearest":
            # uint8 dtype can be included for cpu and cuda input if nearest mode
            acceptable_dtypes.append(paddle.uint8)
        elif _should_use_native_uint8_kernel(interpolation):
            acceptable_dtypes.append(paddle.uint8)

        image = image.reshape(-1, num_channels, old_height, old_width)
        strides = image.stride()
        if image.is_contiguous() and image.shape[0] == 1 and numel != strides[0]:
            new_strides = list(strides)
            new_strides[0] = numel
            image = image.as_strided((1, num_channels, old_height, old_width), new_strides)

        need_cast = dtype not in acceptable_dtypes
        if need_cast:
            image = image.to(dtype=paddle.float32)

        image = interpolate(
            image,
            size=[new_height, new_width],
            mode=interpolation,
            align_corners=align_corners,
        )

        if need_cast:
            if interpolation == "bicubic" and dtype == paddle.uint8:
                image = image.clip_(min=0, max=255)
            if dtype in (paddle.uint8, paddle.int8, paddle.int16, paddle.int32, paddle.int64):
                image = image.round_()
            image = image.to(dtype=dtype)

    return image.reshape(shape[:-3] + [num_channels, new_height, new_width])


def _parse_pad_padding(padding: Union[int, list[int]]) -> list[int]:
    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif isinstance(padding, (tuple, list)):
        if len(padding) == 1:
            pad_left = pad_right = pad_top = pad_bottom = padding[0]
        elif len(padding) == 2:
            pad_left = pad_right = padding[0]
            pad_top = pad_bottom = padding[1]
        elif len(padding) == 4:
            pad_left = padding[0]
            pad_top = padding[1]
            pad_right = padding[2]
            pad_bottom = padding[3]
        else:
            raise ValueError(
                f"Padding must be an int or a 1, 2, or 4 element tuple, not a {len(padding)} element tuple"
            )
    else:
        raise TypeError(f"`padding` should be an integer or tuple or list of integers, but got {padding}")

    return [pad_left, pad_right, pad_top, pad_bottom]


def pad(
    image: paddle.Tensor,
    padding: list[int],
    fill: Optional[Union[int, float, list[float]]] = None,
    padding_mode: str = "constant",
) -> paddle.Tensor:
    # Be aware that while `padding` has order `[left, top, right, bottom]`, `paddle_padding` uses
    # `[left, right, top, bottom]`. This stems from the fact that we align our API with PIL, but need to use `paddle_pad`
    # internally.
    paddle_padding = _parse_pad_padding(padding)

    if padding_mode not in ("constant", "edge", "reflect", "symmetric"):
        raise ValueError(
            f"`padding_mode` should be either `'constant'`, `'edge'`, `'reflect'` or `'symmetric'`, "
            f"but got `'{padding_mode}'`."
        )

    if fill is None:
        fill = 0

    if isinstance(fill, (int, float)):
        return _pad_with_scalar_fill(image, paddle_padding, fill=fill, padding_mode=padding_mode)
    elif len(fill) == 1:
        return _pad_with_scalar_fill(image, paddle_padding, fill=fill[0], padding_mode=padding_mode)
    else:
        return _pad_with_vector_fill(image, paddle_padding, fill=fill, padding_mode=padding_mode)


def _pad_with_scalar_fill(
    image: paddle.Tensor,
    paddle_padding: list[int],
    fill: Union[int, float],
    padding_mode: str,
) -> paddle.Tensor:
    shape = image.shape
    num_channels, height, width = shape[-3:]

    batch_size = 1
    for s in shape[:-3]:
        batch_size *= s

    image = image.reshape(batch_size, num_channels, height, width)

    if padding_mode == "edge":
        # Similar to the padding order, `paddle_pad`'s PIL's padding modes don't have the same names. Thus, we map
        # the PIL name for the padding mode, which we are also using for our API, to the corresponding `paddle_pad`
        # name.
        padding_mode = "replicate"

    if padding_mode == "constant":
        image = paddle_pad(image, paddle_padding, mode=padding_mode, value=float(fill))
    elif padding_mode in ("reflect", "replicate"):
        dtype = image.dtype
        if not image.is_floating_point():
            needs_cast = True
            image = image.to(paddle.float32)
        else:
            needs_cast = False

        image = paddle_pad(image, paddle_padding, mode=padding_mode)

        if needs_cast:
            image = image.to(dtype)
    else:  # padding_mode == "symmetric"
        image = _pad_symmetric(image, paddle_padding)

    new_height, new_width = image.shape[-2:]

    return image.reshape(shape[:-3] + [num_channels, new_height, new_width])


def _pad_with_vector_fill(
    image: paddle.Tensor,
    paddle_padding: list[int],
    fill: list[float],
    padding_mode: str,
) -> paddle.Tensor:
    if padding_mode != "constant":
        raise ValueError(f"Padding mode '{padding_mode}' is not supported if fill is not scalar")

    output = _pad_with_scalar_fill(image, paddle_padding, fill=0, padding_mode="constant")
    left, right, top, bottom = paddle_padding

    fill = paddle.to_tensor(fill).to(dtype=image.dtype).reshape(-1, 1, 1).contiguous()

    if top > 0:
        output[..., :top, :] = fill
    if left > 0:
        output[..., :, :left] = fill
    if bottom > 0:
        output[..., -bottom:, :] = fill
    if right > 0:
        output[..., :, -right:] = fill
    return output


def crop(image: paddle.Tensor, top: int, left: int, height: int, width: int) -> paddle.Tensor:
    h, w = image.shape[-2:]

    right = left + width
    bottom = top + height

    if left < 0 or top < 0 or right > w or bottom > h:
        image = image[..., max(top, 0) : bottom, max(left, 0) : right]
        torch_padding = [
            max(min(right, 0) - left, 0),
            max(right - max(w, left), 0),
            max(min(bottom, 0) - top, 0),
            max(bottom - max(h, top), 0),
        ]
        return _pad_with_scalar_fill(image, torch_padding, fill=0, padding_mode="constant")
    return image[..., top:bottom, left:right]


def _rgb_to_grayscale_image(
    image: paddle.Tensor, num_output_channels: int = 1, preserve_dtype: bool = True
) -> paddle.Tensor:
    if image.shape[-3] == 1 and num_output_channels == 1:
        return image.clone()
    if image.shape[-3] == 1 and num_output_channels == 3:
        s = [1] * len(image.shape)
        s[-3] = 3
        return image.repeat(s)
    r, g, b = image.unbind(dim=-3)
    l_img = r.mul(0.2989).add_(g, alpha=0.587).add_(b, alpha=0.114)
    l_img = l_img.unsqueeze(dim=-3)
    if preserve_dtype:
        l_img = l_img.to(image.dtype)
    if num_output_channels == 3:
        l_img = l_img.expand(image.shape)
    return l_img


def grayscale_to_rgb(image: paddle.Tensor) -> paddle.Tensor:
    if image.shape[-3] >= 3:
        return image
    return _rgb_to_grayscale_image(image, num_output_channels=3, preserve_dtype=True)


def normalize(image: paddle.Tensor, mean: list[float], std: list[float], inplace: bool = False) -> paddle.Tensor:
    if not image.is_floating_point():
        raise TypeError(f"Input tensor should be a float tensor. Got {image.dtype}.")

    if image.ndim < 3:
        raise ValueError(f"Expected tensor to be a tensor image of size (..., C, H, W). Got {image.shape}.")

    if isinstance(std, (tuple, list)):
        divzero = not all(std)
    elif isinstance(std, (int, float)):
        divzero = std == 0
    else:
        divzero = False
    if divzero:
        raise ValueError("std evaluated to zero, leading to division by zero.")

    dtype = image.dtype
    mean = paddle.to_tensor(mean, dtype=dtype)
    std = paddle.to_tensor(std, dtype=dtype)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    if inplace:
        image = image.sub_(mean)
    else:
        image = image.sub(mean)

    return image.div_(std)
