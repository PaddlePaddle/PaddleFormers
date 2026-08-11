# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
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

import copy
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import paddle


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 256, 256),
        "pool_size": None,
        "crop_pct": 0.95,
        "interpolation": "bicubic",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {"fastvit_t": _cfg(crop_pct=0.9), "fastvit_s": _cfg(crop_pct=0.9), "fastvit_m": _cfg(crop_pct=0.95)}


class DropPath(paddle.nn.Layer):
    """Per-sample stochastic depth."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
        random_tensor = paddle.floor(random_tensor)
        return x * random_tensor / keep_prob


class SEBlock(paddle.nn.Module):
    """Squeeze and Excite module.

    Pytorch implementation of `Squeeze-and-Excitation Networks` -
    https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        """Construct a Squeeze and Excite Module.

        Args:
            in_channels: Number of input channels.
            rd_ratio: Input channel reduction ratio.
        """
        super(SEBlock, self).__init__()
        self.reduce = paddle.nn.Conv2d(
            in_channels=in_channels, out_channels=int(in_channels * rd_ratio), kernel_size=1, stride=1, bias=True
        )
        self.expand = paddle.nn.Conv2d(
            in_channels=int(in_channels * rd_ratio), out_channels=in_channels, kernel_size=1, stride=1, bias=True
        )

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        """Apply forward pass."""
        b, c, h, w = inputs.size()
        x = paddle.nn.functional.avg_pool2d(x=inputs, kernel_size=[h, w], exclusive=False)
        x = self.reduce(x)
        x = paddle.nn.functional.relu(x)
        x = self.expand(x)
        x = paddle.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


class MobileOneBlock(paddle.nn.Module):
    """MobileOne building block.

    This block has a multi-branched architecture at train-time
    and plain-CNN style architecture at inference time
    For more details, please refer to our paper:
    `An Improved One millisecond Mobile Backbone` -
    https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        inference_mode: bool = False,
        use_se: bool = False,
        use_act: bool = True,
        use_scale_branch: bool = True,
        num_conv_branches: int = 1,
        activation: paddle.nn.Module = paddle.nn.GELU(),
    ) -> None:
        """Construct a MobileOneBlock module.

        Args:
            in_channels: Number of channels in the input.
            out_channels: Number of channels produced by the block.
            kernel_size: Size of the convolution kernel.
            stride: Stride size.
            padding: Zero-padding size.
            dilation: Kernel dilation factor.
            groups: Group number.
            inference_mode: If True, instantiates model in inference mode.
            use_se: Whether to use SE-ReLU activations.
            use_act: Whether to use activation. Default: ``True``
            use_scale_branch: Whether to use scale branch. Default: ``True``
            num_conv_branches: Number of linear conv branches.
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = paddle.nn.Identity()
        if use_act:
            self.activation = activation
        else:
            self.activation = paddle.nn.Identity()
        if inference_mode:
            self.reparam_conv = paddle.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            norm_layer = paddle.compat.nn.BatchNorm2d(num_features=in_channels)
            if norm_layer.weight.shape[0] == 0:
                norm_layer.weight = paddle.nn.Parameter(paddle.zeros(in_channels))
            if norm_layer.bias.shape[0] == 0:
                norm_layer.bias = paddle.nn.Parameter(paddle.zeros(in_channels))
            self.rbr_skip = norm_layer if out_channels == in_channels and stride == 1 else None
            if num_conv_branches > 0:
                rbr_conv = list()
                for _ in range(self.num_conv_branches):
                    rbr_conv.append(self._conv_bn(kernel_size=kernel_size, padding=padding))
                self.rbr_conv = paddle.nn.ModuleList(rbr_conv)
            else:
                self.rbr_conv = None
            self.rbr_scale = None
            if not isinstance(kernel_size, int):
                kernel_size = kernel_size[0]
            if kernel_size > 1 and use_scale_branch:
                self.rbr_scale = self._conv_bn(kernel_size=1, padding=0)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Apply forward pass."""
        if self.inference_mode:
            if x.dtype == paddle.bfloat16 and self.groups == self.in_channels:
                output = paddle.nn.functional.conv2d(
                    x.astype("float32"),
                    self.reparam_conv.weight.astype("float32"),
                    self.reparam_conv.bias.astype("float32"),
                    stride=self.reparam_conv._stride,
                    padding=self.reparam_conv._padding,
                    dilation=self.reparam_conv._dilation,
                    groups=self.reparam_conv._groups,
                ).astype(x.dtype)
            else:
                output = self.reparam_conv(x)
            return self.activation(self.se(output))
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)
        out = scale_out + identity_out
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                out += self.rbr_conv[ix](x)
        return self.activation(self.se(out))

    def reparameterize(self):
        """Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = paddle.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias
        self.__delattr__("rbr_conv")
        self.__delattr__("rbr_scale")
        if hasattr(self, "rbr_skip"):
            self.__delattr__("rbr_skip")
        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        Returns:
            Tuple of (kernel, bias) after fusing branches.
        """
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            pad = self.kernel_size // 2
            kernel_scale = paddle.compat.nn.functional.pad(kernel_scale, [pad, pad, pad, pad])
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)
        kernel_conv = 0
        bias_conv = 0
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
                kernel_conv += _kernel
                bias_conv += _bias
        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(
        self, branch: Union[paddle.nn.Sequential, paddle.compat.nn.BatchNorm2d]
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        Args:
            branch: Sequence of ops to be fused.

        Returns:
            Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, paddle.nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, paddle.compat.nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_size = self.kernel_size
                if isinstance(self.kernel_size, int):
                    kernel_size = self.kernel_size, self.kernel_size
                kernel_value = paddle.zeros(
                    (self.in_channels, input_dim, kernel_size[0], kernel_size[1]),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, kernel_size[0] // 2, kernel_size[1] // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, kernel_size: int, padding: int) -> paddle.nn.Sequential:
        """Helper method to construct conv-batchnorm layers.

        Args:
            kernel_size: Size of the convolution kernel.
            padding: Zero-padding size.

        Returns:
            Conv-BN module.
        """
        norm_layer = paddle.compat.nn.BatchNorm2d(num_features=self.out_channels)
        if norm_layer.weight.shape[0] == 0:
            norm_layer.weight = paddle.nn.Parameter(paddle.zeros(self.out_channels))
        if norm_layer.bias.shape[0] == 0:
            norm_layer.bias = paddle.nn.Parameter(paddle.zeros(self.out_channels))
        mod_list = paddle.nn.Sequential()
        mod_list.add_module(
            "conv",
            paddle.nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False,
            ),
        )
        mod_list.add_module("bn", norm_layer)
        return mod_list


class ReparamLargeKernelConv(paddle.nn.Module):
    """Building Block of RepLKNet

    This class defines overparameterized large kernel conv block
    introduced in `RepLKNet <https://arxiv.org/abs/2203.06717>`_

    Reference: https://github.com/DingXiaoH/RepLKNet-pytorch
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        small_kernel: int,
        inference_mode: bool = False,
        use_se: bool = False,
        activation: paddle.nn.Module = paddle.nn.GELU(),
    ) -> None:
        """Construct a ReparamLargeKernelConv module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size of the large kernel conv branch.
            stride: Stride size. Default: 1
            groups: Group number. Default: 1
            small_kernel: Kernel size of small kernel conv branch.
            inference_mode: If True, instantiates model in inference mode. Default: ``False``
            activation: Activation module. Default: ``nn.GELU``
        """
        super(ReparamLargeKernelConv, self).__init__()
        self.stride = stride
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.padding = kernel_size // 2
        if use_se:
            self.se = SEBlock(out_channels, rd_ratio=0.25)
        else:
            self.se = paddle.nn.Identity()
        if inference_mode:
            self.lkb_reparam = paddle.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=self.padding,
                dilation=1,
                groups=groups,
                bias=True,
            )
        else:
            self.lkb_origin = self._conv_bn(kernel_size=kernel_size, padding=self.padding)
            if small_kernel is not None:
                assert (
                    small_kernel <= kernel_size
                ), "The kernel size for re-param cannot be larger than the large kernel!"
                self.small_conv = self._conv_bn(kernel_size=small_kernel, padding=small_kernel // 2)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Apply forward pass."""
        if hasattr(self, "lkb_reparam"):
            if x.dtype == paddle.bfloat16 and self.groups == self.in_channels:
                out = paddle.nn.functional.conv2d(
                    x.astype("float32"),
                    self.lkb_reparam.weight.astype("float32"),
                    self.lkb_reparam.bias.astype("float32"),
                    stride=self.lkb_reparam._stride,
                    padding=self.lkb_reparam._padding,
                    dilation=self.lkb_reparam._dilation,
                    groups=self.lkb_reparam._groups,
                ).astype(x.dtype)
            else:
                out = self.lkb_reparam(x)
        else:
            out = self.lkb_origin(x)
            if hasattr(self, "small_conv"):
                out += self.small_conv(x)
        return self.activation(self.se(out))

    def get_kernel_bias(self) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepLKNet-pytorch

        Returns:
            Tuple of (kernel, bias) after fusing branches.
        """
        eq_k, eq_b = self._fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, "small_conv"):
            small_k, small_b = self._fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += paddle.compat.nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        return eq_k, eq_b

    def reparameterize(self) -> None:
        """
        Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        eq_k, eq_b = self.get_kernel_bias()
        self.lkb_reparam = paddle.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.lkb_origin.conv.dilation,
            groups=self.groups,
            bias=True,
        )
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__("lkb_origin")
        if hasattr(self, "small_conv"):
            self.__delattr__("small_conv")

    @staticmethod
    def _fuse_bn(conv: paddle.Tensor, bn: paddle.compat.nn.BatchNorm2d) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Method to fuse batchnorm layer with conv layer.

        Args:
            conv: Convolutional kernel weights.
            bn: Batchnorm 2d layer.

        Returns:
            Tuple of (kernel, bias) after fusing batchnorm.
        """
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, kernel_size: int, padding: int = 0) -> paddle.nn.Sequential:
        """Helper method to construct conv-batchnorm layers.

        Args:
            kernel_size: Size of the convolution kernel.
            padding: Zero-padding size.

        Returns:
            A nn.Sequential Conv-BN module.
        """
        norm_layer = paddle.compat.nn.BatchNorm2d(num_features=self.out_channels)
        if norm_layer.weight.shape[0] == 0:
            norm_layer.weight = paddle.nn.Parameter(paddle.zeros(self.out_channels))
        if norm_layer.bias.shape[0] == 0:
            norm_layer.bias = paddle.nn.Parameter(paddle.zeros(self.out_channels))
        mod_list = paddle.nn.Sequential()
        mod_list.add_module(
            "conv",
            paddle.nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False,
            ),
        )
        mod_list.add_module("bn", norm_layer)
        return mod_list


def convolutional_stem(
    in_channels: int, out_channels: int, inference_mode: bool = False, use_scale_branch: bool = True
) -> paddle.nn.Sequential:
    """Build convolutional stem with MobileOne blocks.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        inference_mode: Flag to instantiate model in inference mode. Default: ``False``

    Returns:
        nn.Sequential object with stem elements.
    """
    return paddle.nn.Sequential(
        MobileOneBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
            use_scale_branch=use_scale_branch,
        ),
        MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=out_channels,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
            use_scale_branch=use_scale_branch,
        ),
        MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
            use_scale_branch=use_scale_branch,
        ),
    )


class LayerNormChannel(paddle.nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_features, eps=1e-05) -> None:
        super().__init__()
        self.weight = paddle.nn.Parameter(paddle.ones(num_features))
        self.bias = paddle.nn.Parameter(paddle.zeros(num_features))
        self.eps = eps

    def forward(self, x) -> paddle.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / paddle.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class MHSA(paddle.nn.Module):
    """Multi-headed Self Attention module.

    Source modified from:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(
        self, dim: int, head_dim: int = 32, qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0
    ) -> None:
        """Build MHSA module that can handle 3D or 4D input tensors.

        Args:
            dim: Number of embedding dimensions.
            head_dim: Number of hidden dimensions per head. Default: ``32``
            qkv_bias: Use bias or not. Default: ``False``
            attn_drop: Dropout rate for attention tensor.
            proj_drop: Dropout rate for projection tensor.
        """
        super().__init__()
        assert dim % head_dim == 0, "dim should be divisible by head_dim"
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5
        self.qkv = paddle.compat.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = paddle.nn.Dropout(attn_drop)
        self.proj = paddle.compat.nn.Linear(dim, dim)
        self.proj_drop = paddle.nn.Dropout(proj_drop)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        shape = x.shape
        B, C, H, W = shape
        N = H * W
        if len(shape) == 4:
            x = paddle.flatten(x, start_dim=2).transpose(-2, -1)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = q * self.scale @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if len(shape) == 4:
            x = x.transpose(-2, -1).reshape(B, C, H, W)
        return x


class PatchEmbed(paddle.nn.Module):
    """Convolutional patch embedding layer."""

    def __init__(
        self,
        patch_size: int,
        stride: int,
        in_channels: int,
        embed_dim: int,
        inference_mode: bool = False,
        use_se: bool = False,
    ) -> None:
        """Build patch embedding layer.

        Args:
            patch_size: Patch size for embedding computation.
            stride: Stride for convolutional embedding layer.
            in_channels: Number of channels of input tensor.
            embed_dim: Number of embedding dimensions.
            inference_mode: Flag to instantiate model in inference mode. Default: ``False``
            use_se: If ``True`` SE block will be used.
        """
        super().__init__()
        block = list()
        block.append(
            ReparamLargeKernelConv(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=stride,
                groups=in_channels,
                small_kernel=3,
                inference_mode=inference_mode,
                use_se=use_se,
            )
        )
        block.append(
            MobileOneBlock(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                inference_mode=inference_mode,
                use_se=False,
                num_conv_branches=1,
            )
        )
        self.proj = paddle.nn.Sequential(*block)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.proj(x)
        return x


class RepMixer(paddle.nn.Module):
    """Reparameterizable token mixer.

    For more details, please refer to our paper:
    `FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization <https://arxiv.org/pdf/2303.14189.pdf>`_
    """

    def __init__(
        self, dim, kernel_size=3, use_layer_scale=True, layer_scale_init_value=1e-05, inference_mode: bool = False
    ):
        """Build RepMixer Module.

        Args:
            dim: Input feature map dimension. :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, H, W)`.
            kernel_size: Kernel size for spatial mixing. Default: 3
            use_layer_scale: If True, learnable layer scale is used. Default: ``True``
            layer_scale_init_value: Initial value for layer scale. Default: 1e-5
            inference_mode: If True, instantiates model in inference mode. Default: ``False``
        """
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode
        if inference_mode:
            self.reparam_conv = paddle.nn.Conv2d(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                groups=self.dim,
                bias=True,
            )
        else:
            self.norm = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                use_act=False,
                use_scale_branch=False,
                num_conv_branches=0,
            )
            self.mixer = MobileOneBlock(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, use_act=False)
            self.use_layer_scale = use_layer_scale
            if use_layer_scale:
                self.layer_scale = paddle.nn.Parameter(
                    layer_scale_init_value * paddle.ones((dim, 1, 1)), requires_grad=True
                )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if hasattr(self, "reparam_conv"):
            if x.dtype == paddle.bfloat16:
                x = paddle.nn.functional.conv2d(
                    x.astype("float32"),
                    self.reparam_conv.weight.astype("float32"),
                    self.reparam_conv.bias.astype("float32"),
                    stride=self.reparam_conv._stride,
                    padding=self.reparam_conv._padding,
                    dilation=self.reparam_conv._dilation,
                    groups=self.reparam_conv._groups,
                ).astype(x.dtype)
            else:
                x = self.reparam_conv(x)
            return x
        else:
            if self.use_layer_scale:
                x = x + self.layer_scale * (self.mixer(x) - self.norm(x))
            else:
                x = x + self.mixer(x) - self.norm(x)
            return x

    def reparameterize(self) -> None:
        """Reparameterize mixer and norm into a single
        convolutional layer for efficient inference.
        """
        if self.inference_mode:
            return
        self.mixer.reparameterize()
        self.norm.reparameterize()
        if self.use_layer_scale:
            w = self.mixer.id_tensor + self.layer_scale.unsqueeze(-1) * (
                self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            )
            b = paddle.squeeze(self.layer_scale) * (self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias)
        else:
            w = self.mixer.id_tensor + self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            b = self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias
        self.reparam_conv = paddle.nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w
        self.reparam_conv.bias.data = b
        self.__delattr__("mixer")
        self.__delattr__("norm")
        if self.use_layer_scale:
            self.__delattr__("layer_scale")


class ConvFFN(paddle.nn.Module):
    """Convolutional FFN Module."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: paddle.nn.Module = paddle.nn.GELU,
        drop: float = 0.0,
    ) -> None:
        """Build convolutional FFN module.

        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of channels after expansion. Default: None
            out_channels: Number of output channels. Default: None
            act_layer: Activation layer. Default: ``GELU``
            drop: Dropout rate. Default: ``0.0``.
        """
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = paddle.nn.Sequential()
        self.conv.add_module(
            "conv",
            paddle.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                groups=in_channels,
                bias=False,
            ),
        )
        norm_layer = paddle.compat.nn.BatchNorm2d(num_features=out_channels)
        if norm_layer.weight.shape[0] == 0:
            norm_layer.weight = paddle.nn.Parameter(paddle.zeros(out_channels))
        if norm_layer.bias.shape[0] == 0:
            norm_layer.bias = paddle.nn.Parameter(paddle.zeros(out_channels))
        self.conv.add_module("bn", norm_layer)
        self.fc1 = paddle.nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = paddle.nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.drop = paddle.nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: paddle.nn.Module) -> None:
        if isinstance(m, paddle.nn.Conv2d):
            paddle.nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                paddle.nn.init.constant_(m.bias, 0)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if x.dtype == paddle.bfloat16:
            depthwise = self.conv.conv
            x = paddle.nn.functional.conv2d(
                x.astype("float32"),
                depthwise.weight.astype("float32"),
                None,
                stride=depthwise._stride,
                padding=depthwise._padding,
                dilation=depthwise._dilation,
                groups=depthwise._groups,
            ).astype(x.dtype)
            x = self.conv.bn(x)
        else:
            x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RepCPE(paddle.nn.Module):
    """Implementation of conditional positional encoding.

    For more details refer to paper:
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_

    In our implementation, we can reparameterize this module to eliminate a skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 768,
        spatial_shape: Union[int, Tuple[int, int]] = (7, 7),
        inference_mode=False,
    ) -> None:
        """Build reparameterizable conditional positional encoding

        Args:
            in_channels: Number of input channels.
            embed_dim: Number of embedding dimensions. Default: 768
            spatial_shape: Spatial shape of kernel for positional encoding. Default: (7, 7)
            inference_mode: Flag to instantiate block in inference mode. Default: ``False``
        """
        super(RepCPE, self).__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = tuple([spatial_shape] * 2)
        assert isinstance(
            spatial_shape, Tuple
        ), f'"spatial_shape" must by a sequence or int, get {type(spatial_shape)} instead.'
        assert len(spatial_shape) == 2, f'Length of "spatial_shape" should be 2, got {len(spatial_shape)} instead.'
        self.spatial_shape = spatial_shape
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.groups = embed_dim
        if inference_mode:
            self.reparam_conv = paddle.nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.embed_dim,
                kernel_size=self.spatial_shape,
                stride=1,
                padding=int(self.spatial_shape[0] // 2),
                groups=self.embed_dim,
                bias=True,
            )
        else:
            self.pe = paddle.nn.Conv2d(
                in_channels, embed_dim, spatial_shape, 1, int(spatial_shape[0] // 2), bias=True, groups=embed_dim
            )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if hasattr(self, "reparam_conv"):
            if x.dtype == paddle.bfloat16:
                x = paddle.nn.functional.conv2d(
                    x.astype("float32"),
                    self.reparam_conv.weight.astype("float32"),
                    self.reparam_conv.bias.astype("float32"),
                    stride=self.reparam_conv._stride,
                    padding=self.reparam_conv._padding,
                    dilation=self.reparam_conv._dilation,
                    groups=self.reparam_conv._groups,
                ).astype(x.dtype)
            else:
                x = self.reparam_conv(x)
            return x
        else:
            x = self.pe(x) + x
            return x

    def reparameterize(self) -> None:
        input_dim = self.in_channels // self.groups
        kernel_value = paddle.zeros(
            (self.in_channels, input_dim, self.spatial_shape[0], self.spatial_shape[1]),
            dtype=self.pe.weight.dtype,
            device=self.pe.weight.device,
        )
        for i in range(self.in_channels):
            kernel_value[i, i % input_dim, self.spatial_shape[0] // 2, self.spatial_shape[1] // 2] = 1
        id_tensor = kernel_value
        w_final = id_tensor + self.pe.weight
        b_final = self.pe.bias
        self.reparam_conv = paddle.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.spatial_shape,
            stride=1,
            padding=int(self.spatial_shape[0] // 2),
            groups=self.embed_dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w_final
        self.reparam_conv.bias.data = b_final
        self.__delattr__("pe")


class RepMixerBlock(paddle.nn.Module):
    """Implementation of Metaformer block with RepMixer as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: paddle.nn.Module = paddle.nn.GELU,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-05,
        inference_mode: bool = False,
    ):
        """Build RepMixer Block.

        Args:
            dim: Number of embedding dimensions.
            kernel_size: Kernel size for repmixer. Default: 3
            mlp_ratio: MLP expansion ratio. Default: 4.0
            act_layer: Activation layer. Default: ``nn.GELU``
            drop: Dropout rate. Default: 0.0
            drop_path: Drop path rate. Default: 0.0
            use_layer_scale: Flag to turn on layer scale. Default: ``True``
            layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
            inference_mode: Flag to instantiate block in inference mode. Default: ``False``
        """
        super().__init__()
        self.token_mixer = RepMixer(
            dim,
            kernel_size=kernel_size,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
        )
        assert mlp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(mlp_ratio)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(in_channels=dim, hidden_channels=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else paddle.nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = paddle.nn.Parameter(
                layer_scale_init_value * paddle.ones((dim, 1, 1)), requires_grad=True
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.layer_scale * self.convffn(x))
        else:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.convffn(x))
        return x


class AttentionBlock(paddle.nn.Module):
    """Implementation of metaformer block with MHSA as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        act_layer: paddle.nn.Module = paddle.nn.GELU,
        norm_layer: paddle.nn.Module = paddle.compat.nn.BatchNorm2d,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-05,
    ):
        """Build Attention Block.

        Args:
            dim: Number of embedding dimensions.
            mlp_ratio: MLP expansion ratio. Default: 4.0
            act_layer: Activation layer. Default: ``nn.GELU``
            norm_layer: Normalization layer. Default: ``nn.BatchNorm2d``
            drop: Dropout rate. Default: 0.0
            drop_path: Drop path rate. Default: 0.0
            use_layer_scale: Flag to turn on layer scale. Default: ``True``
            layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
        """
        super().__init__()
        norm_layer_ = norm_layer(num_features=dim)
        if norm_layer_.weight.shape[0] == 0:
            norm_layer_.weight = paddle.nn.Parameter(paddle.zeros(dim))
        if norm_layer_.bias.shape[0] == 0:
            norm_layer_.bias = paddle.nn.Parameter(paddle.zeros(dim))
        self.norm = norm_layer_
        self.token_mixer = MHSA(dim=dim)
        assert mlp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(mlp_ratio)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(in_channels=dim, hidden_channels=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else paddle.nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = paddle.nn.Parameter(
                layer_scale_init_value * paddle.ones((dim, 1, 1)), requires_grad=True
            )
            self.layer_scale_2 = paddle.nn.Parameter(
                layer_scale_init_value * paddle.ones((dim, 1, 1)), requires_grad=True
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.layer_scale_2 * self.convffn(x))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.convffn(x))
        return x


def basic_blocks(
    dim: int,
    block_index: int,
    num_blocks: List[int],
    token_mixer_type: str,
    kernel_size: int = 3,
    mlp_ratio: float = 4.0,
    act_layer: paddle.nn.Module = paddle.nn.GELU,
    norm_layer: paddle.nn.Module = paddle.compat.nn.BatchNorm2d,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    use_layer_scale: bool = True,
    layer_scale_init_value: float = 1e-05,
    inference_mode=False,
) -> paddle.nn.Sequential:
    """Build FastViT blocks within a stage.

    Args:
        dim: Number of embedding dimensions.
        block_index: block index.
        num_blocks: List containing number of blocks per stage.
        token_mixer_type: Token mixer type.
        kernel_size: Kernel size for repmixer.
        mlp_ratio: MLP expansion ratio.
        act_layer: Activation layer.
        norm_layer: Normalization layer.
        drop_rate: Dropout rate.
        drop_path_rate: Drop path rate.
        use_layer_scale: Flag to turn on layer scale regularization.
        layer_scale_init_value: Layer scale value at initialization.
        inference_mode: Flag to instantiate block in inference mode.

    Returns:
        nn.Sequential object of all the blocks within the stage.
    """
    blocks = []
    for block_idx in range(num_blocks[block_index]):
        block_dpr = drop_path_rate * (block_idx + sum(num_blocks[:block_index])) / (sum(num_blocks) - 1)
        if token_mixer_type == "repmixer":
            blocks.append(
                RepMixerBlock(
                    dim,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    inference_mode=inference_mode,
                )
            )
        elif token_mixer_type == "attention":
            blocks.append(
                AttentionBlock(
                    dim,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                )
            )
        else:
            raise ValueError("Token mixer type: {} not supported".format(token_mixer_type))
    blocks = paddle.nn.Sequential(*blocks)
    return blocks


class GlobalPool2D(paddle.nn.Module):
    """This class implements global pooling with linear projection."""

    def __init__(self, in_dim: int, out_dim: int, *args, **kwargs) -> None:
        super().__init__()
        scale = in_dim**-0.5
        self.proj = paddle.nn.Parameter(scale * paddle.randn(size=(in_dim, out_dim)))
        self.in_dim = in_dim
        self.out_dim = out_dim

    def pool(self, x) -> paddle.Tensor:
        if x.dim() == 4:
            dims = [-2, -1]
        elif x.dim() == 5:
            dims = [-3, -2, -1]
        x = paddle.mean(x, dim=dims, keepdim=False)
        return x

    def forward(self, x: paddle.Tensor, *args, **kwargs) -> paddle.Tensor:
        assert x.dim() == 4, "Input should be 4-dimensional (Batch x in_dim x in_height x in_width). Got: {}".format(
            x.shape
        )
        x = self.pool(x)
        x = x @ self.proj
        return x


class FastViT(paddle.nn.Module):
    """
    This class implements `FastViT architecture <https://arxiv.org/pdf/2303.14189.pdf>`_
    """

    def __init__(
        self,
        layers,
        token_mixers: Tuple[str, ...],
        embed_dims=None,
        mlp_ratios=None,
        downsamples=None,
        se_downsamples=None,
        repmixer_kernel_size=3,
        norm_layer: paddle.nn.Module = paddle.compat.nn.BatchNorm2d,
        act_layer: paddle.nn.Module = paddle.nn.GELU,
        num_classes=1000,
        pos_embs=None,
        down_patch_size=7,
        down_stride=2,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-05,
        init_cfg=None,
        pretrained=None,
        cls_ratio=2.0,
        inference_mode=False,
        stem_scale_branch=True,
        **kwargs
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        if len(layers) == 4:
            self.out_indices = [0, 2, 4, 7]
        elif len(layers) == 5:
            self.out_indices = [0, 2, 4, 7, 10]
        else:
            raise NotImplementedError("FPN is not implemented for more than 5 stages.")
        if pos_embs is None:
            pos_embs = [None] * len(layers)
        if se_downsamples is None:
            se_downsamples = [False] * len(layers)
        self.patch_embed = convolutional_stem(3, embed_dims[0], inference_mode, use_scale_branch=stem_scale_branch)
        network = []
        for i in range(len(layers)):
            if pos_embs[i] is not None:
                network.append(pos_embs[i](embed_dims[i], embed_dims[i], inference_mode=inference_mode))
            stage = basic_blocks(
                embed_dims[i],
                i,
                layers,
                token_mixer_type=token_mixers[i],
                kernel_size=repmixer_kernel_size,
                mlp_ratio=mlp_ratios[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                inference_mode=inference_mode,
            )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        in_channels=embed_dims[i],
                        embed_dim=embed_dims[i + 1],
                        inference_mode=inference_mode,
                        use_se=se_downsamples[i + 1],
                    )
                )
        self.network = paddle.nn.ModuleList(network)
        self.conv_exp = MobileOneBlock(
            in_channels=embed_dims[-1],
            out_channels=int(embed_dims[-1] * cls_ratio),
            kernel_size=3,
            stride=1,
            padding=1,
            groups=embed_dims[-1],
            inference_mode=inference_mode,
            use_se=True,
            num_conv_branches=1,
        )
        self.head = (
            paddle.compat.nn.Linear(int(embed_dims[-1] * cls_ratio), num_classes)
            if num_classes > 0
            else paddle.nn.Identity()
        )
        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)

    def cls_init_weights(self, m: paddle.nn.Module) -> None:
        """Init. for classification"""
        if isinstance(m, paddle.compat.nn.Linear):
            paddle.nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, paddle.compat.nn.Linear) and m.bias is not None:
                paddle.nn.init.constant_(m.bias, 0)

    def forward_embeddings(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x: paddle.Tensor, *args, **kwargs) -> paddle.Tensor:
        for idx, block in enumerate(self.network):
            x = block(x)
        return x

    def forward(self, x: paddle.Tensor, *args, **kwargs) -> Union[paddle.Tensor, Dict[str, paddle.Tensor]]:
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        x = self.conv_exp(x)
        cls_out = self.head(x)
        out_dict = dict()
        if kwargs.get("return_image_embeddings", False):
            out_dict.update({"logits": cls_out})
            out_dict.update({"image_embeddings": x})
            return out_dict
        else:
            return cls_out


def fastvithd(pretrained=False, **kwargs):
    """Instantiate FastViTHD model variant."""
    layers = [2, 12, 24, 4, 2]
    embed_dims = [96, 192, 384, 768, 1536]
    mlp_ratios = [4, 4, 4, 4, 4]
    downsamples = [True, True, True, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7)), partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = "repmixer", "repmixer", "repmixer", "attention", "attention"
    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        norm_layer=LayerNormChannel,
        stem_scale_branch=False,
        inference_mode=True,
        **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_m"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model


class FastVLMVisionModel(paddle.nn.Layer):
    """MobileCLIP/FastViT-HD vision tower used by FastVLM."""

    def __init__(self, image_size=1024, projection_dim=768, trainable=True, vision_config=None):
        super().__init__()
        self.image_size = image_size
        self.patch_size = 64
        if vision_config is None:
            self.hidden_size = 3072
            self.vision_model = fastvithd()
        else:
            layers = vision_config.get("layers", [1, 1, 1, 1])
            embed_dims = vision_config.get("embed_dims", [8, 16, 32, 64])
            self.hidden_size = int(embed_dims[-1] * vision_config.get("cls_ratio", 2.0))
            self.patch_size = vision_config.get("patch_size", 16)
            self.vision_model = FastViT(
                layers=layers,
                token_mixers=vision_config.get("token_mixers", ["repmixer", "repmixer", "attention", "attention"]),
                embed_dims=embed_dims,
                mlp_ratios=vision_config.get("mlp_ratios", [2] * len(layers)),
                downsamples=[True] * len(layers),
                pos_embs=[None] * len(layers),
                norm_layer=LayerNormChannel,
                down_patch_size=3,
                cls_ratio=vision_config.get("cls_ratio", 2.0),
                stem_scale_branch=False,
                inference_mode=True,
            )
        self.vision_model.head = GlobalPool2D(self.hidden_size, projection_dim)
        if not trainable:
            for parameter in self.parameters():
                parameter.stop_gradient = True

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values, return_image_embeddings=True)
        image_features = outputs["image_embeddings"]
        batch_size, channels, height, width = image_features.shape
        return image_features.reshape([batch_size, channels, height * width]).transpose([0, 2, 1])

    @property
    def num_patches(self):
        return (self.image_size // self.patch_size) ** 2
