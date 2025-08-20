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
import paddle.nn as nn

from ..generation.configuration_utils import PretrainedConfig
from ..utils.log import logger
from .criterion.loss_utils import calc_lm_head_logits

__all__ = ["LMHead"]


class LMHead(nn.Layer):
    def __init__(self, config: PretrainedConfig):
        """
        transpose_y (bool): Whether to transpose the lm_head weight matrix before matrix multiplication.
        """
        super().__init__()
        self.config = config
        self.use_bias = config.get("lm_head_bias", False)
        self.transpose_y = config.get("tie_word_embeddings", False)
        self.vocab_parallel = False

        # apply vocab tensor parallel
        if config.tensor_parallel_degree > 1 and config.vocab_size % config.tensor_parallel_degree == 0:
            vocab_size = config.vocab_size // config.tensor_parallel_degree
            self.vocab_parallel = True
        else:
            vocab_size = config.vocab_size
            if config.tensor_parallel_degree > 1:
                logger.warning_once(
                    "lm_head vocab parallelism is disabled (vocab_size=%d %% tp_degree=%d != 0).",
                    vocab_size,
                    config.tensor_parallel_degree,
                )
        self.lm_head_shape = (
            [config.hidden_size, vocab_size] if not self.transpose_y else [vocab_size, config.hidden_size]
        )

        self.weight = self.create_parameter(
            shape=self.lm_head_shape,
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.XavierNormal(1.0),
        )

        # setting distributed attr for tensor parallel
        self.weight.is_distributed = self.vocab_parallel

        if self.weight.is_distributed:
            self.weight.split_axis = 0 if self.transpose_y else 1

        if self.use_bias:
            self.bias = self.create_parameter(
                shape=[vocab_size],
                dtype=paddle.get_default_dtype(),
                attr=paddle.ParamAttr(initializer=paddle.nn.initializer.constant.Constant(0.0)),
            )

            # setting distributed attr for tensor parallel
            self.bias.is_distributed = self.vocab_parallel
            if self.bias.is_distributed:
                self.bias.split_axis = 0
        else:
            self.bias = None

    def forward(self, hidden_states, tensor_parallel_output=None):
        """Project hidden states to vocabulary logits.

        Args:
            hidden_states (paddle.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]
            tensor_parallel_output (Optional[bool]): Whether to output parallel results. Defaults to None.

        Returns:
            Union[
                Tuple[paddle.Tensor, paddle.Tensor, Optional[paddle.Tensor]]:
                    # When use_recompute_loss_fn or use_sparse_head_and_loss_fn
                    - hidden_states: Original input
                    - weight: Projection weights
                    - bias: Optional bias term
                Tuple[paddle.Tensor, paddle.Tensor, Optional[paddle.Tensor], bool]:  # With tensor_parallel_output
                    Same as above plus tensor_parallel_output flag
                paddle.Tensor:  # Normal case
                    Logits tensor of shape [batch_size, seq_len, vocab_size]
            ]
        """
        if self.config.get("use_fused_head_and_loss_fn", False):
            return (
                hidden_states,
                self.weight,
                self.bias,
                self.config.tie_word_embeddings,
            )

        return calc_lm_head_logits(
            self.config,
            hidden_states,
            self.weight,
            self.bias,
            tensor_parallel_output,
            training=self.training,
            gather_hidden_states=True,
        )

    def extra_repr(self):
        hidden_size, vocab_size = self.lm_head_shape if not self.transpose_y else self.lm_head_shape[::-1]
        return f"hidden_size={hidden_size}, vocab_size={vocab_size}, dtype={self.weight.dtype}, vocab_parallel={self.vocab_parallel}"
