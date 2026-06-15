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
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol

from paddleformers.fleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    Linear,
    RowParallelLinear,
)
from paddleformers.fleet.transformer.dot_product_attention import DotProductAttention
from paddleformers.fleet.transformer.mlp import MLPSublayersSpec


# from paddleformers.fleet.transformer.moe.experts import GroupedMLP, SequentialMLP
# HACK(Guoxia Wang): need remove later
class GroupedMLP:
    pass


class SequentialMLP:
    pass


from paddleformers.fleet.transformer.paddle_norm import WrappedPaddleNorm

LNImpl = WrappedPaddleNorm


class BackendSpecProvider(Protocol):
    """A protocol for providing the sublayers_spec used in Spec building."""

    @abstractmethod
    def column_parallel_linear(self) -> type:
        """Which column parallel linear layer the backend uses"""
        ...

    @abstractmethod
    def row_parallel_linear(self) -> type:
        """Which row parallel linear layer the backend uses"""
        ...

    @abstractmethod
    def fuse_layernorm_and_linear(self) -> bool:
        """Does the backend support a single layer for layernorm and linear"""
        ...

    @abstractmethod
    def column_parallel_layer_norm_linear(self) -> type | None:
        """Which layer for sequential layernorm and linear"""
        ...

    @abstractmethod
    def layer_norm(self, rms_norm: bool = False, for_qk: bool = False) -> type:
        """Which layer for layernorm"""
        ...

    @abstractmethod
    def core_attention(self) -> type:
        """Which layer to use for attention"""
        ...

    @abstractmethod
    def grouped_mlp_layers(
        self, moe_use_grouped_gemm: bool, moe_use_legacy_grouped_gemm: bool
    ) -> tuple[type, MLPSublayersSpec | None]:
        """Which layer and sublayers_spec to use for grouped mlp"""
        ...

    @abstractmethod
    def hidden_act(self) -> type:
        """Which layer to use for activation function"""
        ...


class LocalSpecProvider(BackendSpecProvider):
    """A protocol for providing Local sublayers_spec used in Spec building."""

    def linear(self) -> type:
        """Which linear layer the backend uses"""
        return Linear

    def column_parallel_linear(self) -> type:
        """Which column parallel linear layer the backend uses"""
        return ColumnParallelLinear

    def row_parallel_linear(self) -> type:
        """Which row parallel linear layer the backend uses"""
        return RowParallelLinear

    def fuse_layernorm_and_linear(self) -> bool:
        """Does the backend choose a single layer for layernorm and linear"""
        return False

    def column_parallel_layer_norm_linear(self) -> type | None:
        """Which layer for sequential layernorm and linear"""
        return None

    def layer_norm(self, rms_norm: bool = False, for_qk: bool = False) -> type:
        """Which module to use for layer norm"""
        if rms_norm:
            # Matching get_gpt_layer_local_spec.
            # Why does the global need to be updated?
            global LNImpl
            LNImpl = WrappedPaddleNorm
        return LNImpl

    def core_attention(self) -> type:
        """Which layer to use for attention"""
        return DotProductAttention

    def grouped_mlp_layers(
        self, moe_use_grouped_gemm: bool, moe_use_legacy_grouped_gemm: bool
    ) -> tuple[type, MLPSublayersSpec | None]:
        """Which layer and sublayers_spec to use for grouped mlp"""
        if moe_use_grouped_gemm:
            return GroupedMLP, None
        else:
            return SequentialMLP, MLPSublayersSpec(up_gate_proj=ColumnParallelLinear, down_proj=RowParallelLinear)

    def hidden_act(self) -> type:
        """Which layer to use for activation function"""
        return None
