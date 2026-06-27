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
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import paddle
from paddle.distributed.fleet.meta_parallel import LayerSpec
from paddle.nn.functional import layer_norm, rms_norm

try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        mark_as_sequence_parallel_parameter,
    )
except ImportError:
    logging.warn("Fail to import mark_as_sequence_parallel_parameter!")

    def mark_as_sequence_parallel_parameter(parameter):
        return parameter


from paddle.distributed.fleet.meta_parallel import ScheduleNode

from paddleformers.fleet.jit import jit_fuser

if TYPE_CHECKING:
    from paddle import Tensor

    from paddleformers.fleet.transformer import TransformerConfig


class RMSNorm(paddle.nn.Layer):
    def __init__(
        self,
        config: TransformerConfig,
        normalized_shape=None,
        norm_eps=None,
        input_is_parallel=False,
        **kwargs,
    ):
        super().__init__()
        self.normalized_shape = (
            config.hidden_size if normalized_shape is None else normalized_shape
        )
        self.variance_epsilon = (
            config.rms_norm_eps if norm_eps is None else norm_eps
        )

        self.weight = paddle.create_parameter(
            shape=[self.normalized_shape],
            dtype=config.params_dtype
            if config.params_dtype is not None
            else paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        self.config = config

        if input_is_parallel:
            self.enable_sequence_parallel()

    def forward(self, hidden_states: Tensor):
        # Ensure hidden_states dtype matches weight dtype for rms_norm
        if hidden_states.dtype != self.weight.dtype:
            hidden_states = hidden_states.astype(self.weight.dtype)
        rms_norm_out = rms_norm(
            hidden_states,
            hidden_states.shape[-1:],
            self.weight,
            self.variance_epsilon,
        )
        if isinstance(rms_norm_out, (tuple, list)):
            return rms_norm_out[0].astype(self.weight.dtype)
        else:
            return rms_norm_out.astype(self.weight.dtype)

    def enable_sequence_parallel(self):
        mark_as_sequence_parallel_parameter(self.weight)


class LayerNorm(paddle.nn.Layer):
    def __init__(
        self,
        config: TransformerConfig,
        normalized_shape=None,
        norm_eps=None,
        input_is_parallel=False,
        **kwargs,
    ):
        super().__init__()
        self.normalized_shape = (
            config.hidden_size if normalized_shape is None else normalized_shape
        )
        self.variance_epsilon = (
            config.rms_norm_eps if norm_eps is None else norm_eps
        )
        self.weight = paddle.create_parameter(
            shape=[self.normalized_shape],
            dtype=config.params_dtype
            if config.params_dtype is not None
            else paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        param_shape = [np.prod(self.normalized_shape)]
        self.bias = self.create_parameter(
            shape=param_shape,
            dtype=config.params_dtype
            if config.params_dtype is not None
            else paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Constant(0.0),
            is_bias=True,
        )
        self.config = config
        if input_is_parallel:
            self.enable_sequence_parallel()

    def forward(self, hidden_states: Tensor):
        output = layer_norm(
            hidden_states,
            normalized_shape=self.normalized_shape,
            weight=self.weight,
            bias=self.bias,
            epsilon=self.variance_epsilon,
        )
        return output.astype(self.weight.dtype)

    def enable_sequence_parallel(self):
        mark_as_sequence_parallel_parameter(self.weight)


class FusedRMSNorm(RMSNorm):
    def forward(self, hidden_states: Tensor):
        rms_norm_out = rms_norm(
            hidden_states,
            hidden_states.shape[-1:],
            self.weight,
            self.variance_epsilon,
        )
        if isinstance(rms_norm_out, (tuple, list)):
            return rms_norm_out[0].astype(self.weight.dtype)
        else:
            return rms_norm_out.astype(self.weight.dtype)


class RMSNormTriton(RMSNorm):
    """Wrapper for triton RMSNorm, used for fused QK norm."""

    def forward(self, hidden_states: Tensor):
        from paddleformers.fleet.triton_ops.rms_norm_fusion import (
            RMSNormFusionTriton,
        )

        return RMSNormFusionTriton.apply(
            hidden_states, self.weight, self.variance_epsilon
        )


class WrappedRMSNormTriton:
    """Factory class for RMSNormTriton, handles parameter name conversion.

    Converts build_spec_layer parameters (hidden_size, eps) to
    RMSNorm parameters (normalized_shape, norm_eps).
    """

    def __new__(
        cls,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        input_is_parallel: bool | None = None,
        **kwargs,
    ):
        return RMSNormTriton(
            config=config,
            normalized_shape=hidden_size,
            norm_eps=eps,
            input_is_parallel=input_is_parallel
            if input_is_parallel is not None
            else False,
        )


class WrappedPaddleNorm:
    def __new__(
        cls,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        input_is_parallel: bool | None = None,
    ):
        if config.normalization == "RMSNorm":
            norm_cls = RMSNorm
        elif config.normalization == "LayerNorm":
            norm_cls = LayerNorm
        else:
            raise Exception("Only RMSNorm for now.")

        if input_is_parallel is None:
            input_is_parallel = (
                config.sequence_parallel
                or config.tensor_model_parallel_size > 1
            )
        return norm_cls(
            config=config,
            normalized_shape=hidden_size,
            norm_eps=eps,
            input_is_parallel=input_is_parallel,
        )

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="WrappedPaddleNorm")


class WrappedPaddleNormPipe(paddle.nn.Layer):
    """Pipeline-compatible normalization layer.

    This layer is placed after transformer_layers and before MTP in the pipeline,
    aligning with Megatron-LM where hidden_states go through decoder.final_layernorm
    before being used by both MTP and LM Head.

    When MTP is enabled, the input is a concatenated tensor [main_hidden, mtp_emb_0, ...].
    Only main_hidden (tensor_list[0]) is normalized; the remaining MTP embeddings are
    passed through unchanged. The normalized main_hidden is then passed through MTP
    layers (which transparently forward it) to LM Head.
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
        self.norm = WrappedPaddleNorm(
            config, hidden_size, eps, input_is_parallel
        )

    def forward(self, dict_args: dict):
        if (
            self.config.num_nextn_predict_layers is not None
            and self.config.num_nextn_predict_layers > 0
            and not self.config.mtp_load_weight_only
            and not (
                not self.config.gpt_model_use_experimental_version
                and self.config.enable_mtp_magic_send
            )
        ):
            hidden_states_concat = dict_args["hidden_states"]
            tensor_list = paddle.split(
                hidden_states_concat, self.config.num_nextn_predict_layers + 1
            )
            dict_args["hidden_states"] = tensor_list[0]
        rst = {
            **dict_args,
            "hidden_states": self.norm(dict_args["hidden_states"]),
        }
        if (
            self.config.num_nextn_predict_layers is not None
            and self.config.num_nextn_predict_layers > 0
            and not self.config.mtp_load_weight_only
            and not (
                not self.config.gpt_model_use_experimental_version
                and self.config.enable_mtp_magic_send
            )
        ):
            # normalize MTP hidden_states
            if self.config.gpt_model_use_experimental_version:
                for i in range(1, len(tensor_list)):
                    tensor_list[i] = self.norm(tensor_list[i])
            hidden_states_concat = paddle.concat(
                [rst["hidden_states"], *tensor_list[1:]]
            )
            rst["hidden_states"] = hidden_states_concat
        rst = {**dict_args, **rst}

        # Loss-path MD5 probe: final_layernorm output
        if (
            os.environ.get("LOG_LAYER_MD5", "0") == "1"
            or os.environ.get("LOG_LOSS_MD5", "0") == "1"
        ):
            import hashlib

            rank = paddle.distributed.get_rank()
            h = rst["hidden_states"]
            md5 = hashlib.md5(h.cast("float32").numpy().tobytes()).hexdigest()
            print(
                f"[LOSS_PATH_MD5] rank={rank} final_layernorm_output shape={list(h.shape)} md5={md5}",
                flush=True,
            )

        return rst

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="WrappedPaddleNormPipe")


class L2Norm(paddle.nn.Layer):
    """
    Applies L2 normalization to the input tensor along the last dimension.

    This layer normalizes the input tensor such that the mean of the squared values
    along the last dimension is 1 (within a small epsilon for numerical stability).

    Args:
        hidden_size (int): Expected input shape for normalization (not used internally).
        eps (float, optional): A small value added to the denominator for numerical stability.
            Default: 1e-6.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    @jit_fuser
    def _norm(self, x):
        """
        Performs the actual L2 normalization.

        Args:
            x (paddle.Tensor): The input tensor to normalize.

        Returns:
            paddle.Tensor: The L2-normalized tensor.
        """
        x_float = x.float()
        return (
            x_float
            * paddle.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        ).astype(x.dtype)

    def forward(self, x):
        """
        Forward pass of the L2Norm module.

        Args:
            x (paddle.Tensor): Input tensor.

        Returns:
            paddle.Tensor: L2-normalized tensor with the same dtype as input.
        """
        return self._norm(x)


def get_norm_extra_args(
    layer_or_spec, config, output_size, eps, input_is_parallel
):
    """
    Handle the difference of arguments signature between
    WrappedPaddleNorm and other Norm implementation.
    """
    norm_cls = (
        layer_or_spec.layer
        if isinstance(layer_or_spec, LayerSpec)
        else layer_or_spec
    )
    extra_args = {
        "config": config,
        "input_is_parallel": input_is_parallel,
    }
    if norm_cls is WrappedPaddleNorm:
        extra_args["hidden_size"] = output_size
        extra_args["eps"] = eps
    else:
        extra_args["normalized_shape"] = output_size
        extra_args["norm_eps"] = eps

    return extra_args
