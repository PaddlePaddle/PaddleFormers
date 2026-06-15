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

"""Qwen3.5 model definitions for PaddleFleet.

This module defines:

* ``Qwen3_5RMSNorm`` -- 1-centered RMSNorm.
* ``Qwen3_5RMSNormPipe`` -- pipeline-compatible wrapper.
* ``Qwen3_5VisionModel`` -- vision encoder (ViT + patch merger).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
from paddle import Tensor
from paddle.distributed.fleet.meta_parallel import LayerDesc, ScheduleNode
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    mark_as_sequence_parallel_parameter,
)

from ...transformer.transformer_encoder import TransformerEncoder

if TYPE_CHECKING:
    from paddle.distributed.fleet.meta_parallel import LayerSpec

    from ...transformer.transformer_config import TransformerConfig

logger = logging.getLogger(__name__)


# ======================================================================
# 1-centered RMSNorm (matching HuggingFace Qwen3_5RMSNorm)
# ======================================================================


class Qwen3_5RMSNorm(paddle.nn.Layer):
    """RMSNorm with 1-centered parameterization.

    Weight is initialized to 0 and the forward computes::

        output = rms_norm(x) * (1.0 + weight)

    This matches the HuggingFace ``Qwen3_5RMSNorm`` so that weight
    decay regularizes deviations from identity scale rather than
    pushing the scale toward zero.

    The constructor accepts both calling conventions used internally:
    - ``(config, hidden_size, eps, input_is_parallel)``
      used by ``TransformerLayer.build_layer``
    - ``(config, normalized_shape=..., norm_eps=...)``
      used by the ``SelfAttention._build_norm`` else-branch
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int | None = None,
        eps: float | None = None,
        input_is_parallel: bool = False,
        normalized_shape: int | None = None,
        norm_eps: float | None = None,
        **kwargs,
    ):
        super().__init__()
        # Resolve hidden_size from either calling convention
        dim = hidden_size if hidden_size is not None else normalized_shape
        if dim is None:
            dim = config.hidden_size
        self.normalized_shape = dim

        # Resolve eps from either calling convention
        self.variance_epsilon = eps if eps is not None else (norm_eps if norm_eps is not None else config.rms_norm_eps)

        # Weight initialized to 0 (1-centered parameterization)
        self.weight = paddle.create_parameter(
            shape=[self.normalized_shape],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Constant(0.0),
        )
        self.config = config

        if input_is_parallel:
            self.enable_sequence_parallel()

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype("float32")
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return (hidden_states * (1.0 + self.weight.astype("float32"))).astype(input_dtype)

    def enable_sequence_parallel(self):
        mark_as_sequence_parallel_parameter(self.weight)


class Qwen3_5RMSNormPipe(paddle.nn.Layer):
    """Pipeline-compatible wrapper for ``Qwen3_5RMSNorm``.

    Follows the same pattern as ``WrappedPaddleNormPipe``:
    handles dict I/O and MTP tensor splitting.
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        input_is_parallel: bool | None = None,
    ):
        super().__init__()
        self.config = config
        self.norm = Qwen3_5RMSNorm(
            config,
            hidden_size,
            eps,
            input_is_parallel=input_is_parallel or False,
        )

    def forward(self, dict_args: dict):
        if self.config.num_nextn_predict_layers is not None and self.config.num_nextn_predict_layers > 0:
            hidden_states_concat = dict_args["hidden_states"]
            tensor_list = paddle.split(
                hidden_states_concat,
                self.config.num_nextn_predict_layers + 1,
            )
            dict_args["hidden_states"] = tensor_list[0]
        rst = {
            **dict_args,
            "hidden_states": self.norm(dict_args["hidden_states"]),
        }
        if self.config.num_nextn_predict_layers is not None and self.config.num_nextn_predict_layers > 0:
            hidden_states_concat = paddle.concat([rst["hidden_states"], *tensor_list[1:]])
            rst["hidden_states"] = hidden_states_concat
        return rst

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="Qwen3_5RMSNormPipe")


# ======================================================================
# Vision model
# ======================================================================


@dataclass
class Qwen3_5VisionSublayersSpec:
    """LayerSpecs for Qwen3.5 vision model: embedding + transformer layers + patch merger."""

    embedding: LayerSpec = None
    head_empty_layers: list[LayerSpec] = None
    transformer_layers: list[LayerSpec] = None
    tail_empty_layers: list[LayerSpec] = None
    merger: LayerSpec = None


class Qwen3_5VisionModel(TransformerEncoder):
    def get_layer_desc_list(self, spec: Qwen3_5VisionSublayersSpec):
        layers = []
        name_prefix = f"model.{self.modal}" if self.modal else "model"

        self.add_sequential_layer(layers, LayerDesc(spec.embedding), name_prefix)
        self.get_encoder_layer_desc_list(layers, spec, name_prefix)
        self.add_sequential_layer(layers, LayerDesc(spec.merger), f"{name_prefix}.merger")

        return layers
