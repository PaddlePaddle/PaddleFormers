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

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from paddle.distributed.communication.group import Group

    from paddleformers.fleet.transformer.transformer_config import TransformerConfig

import paddle
from paddle import Tensor

from paddleformers.fleet import tensor_parallel
from paddleformers.fleet.transformer.layer import FleetLayer
from paddleformers.fleet.utils import get_tensor_model_parallel_group_if_none


class LanguageModelEmbedding(FleetLayer):
    """Language model embeddings.

    Args:
        config (TransformerConfig): config object with all necessary configs
        vocab_size (int): vocabulary size
        max_sequence_length (int): maximum size of sequence. This
                             is used for positional embedding
        add_position_embedding (bool): Add a position embedding.
        embedding_dropout_prob (float): dropout probability for embeddings
        num_tokentypes (int): Set to 0 without binary head, and 2 with a binary head. Defaults to 0.
        scatter_to_sequence_parallel (bool): Set to False to disable scatter of embedding
            across sequence parallel region. Defaults to True.
    """

    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: Literal[
            "learned_absolute", "rope", "none"
        ] = "learned_absolute",
        num_tokentypes: int = 0,
        scatter_to_sequence_parallel: bool = True,
        tp_group: Group | None = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config
        self.vocab_size: int = vocab_size
        self.max_sequence_length: int = max_sequence_length
        self.add_position_embedding: bool = (
            position_embedding_type == "learned_absolute"
        )
        self.sequence_parallel = self.config.sequence_parallel
        self.num_tokentypes = num_tokentypes
        self.scatter_to_sequence_parallel = scatter_to_sequence_parallel
        if self.sequence_parallel:
            assert self.scatter_to_sequence_parallel is True, (
                "If sequence parallel is turned on, scatter_to_sequence_parallel "
                "must be set to True."
            )
        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group)
        self.reduce_scatter_embeddings = (
            (not self.add_position_embedding)
            and self.num_tokentypes <= 0
            and self.sequence_parallel
            and self.scatter_to_sequence_parallel
        )

        # Word embeddings (parallel).
        self.embed_tokens = tensor_parallel.VocabParallelEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.config.hidden_size,
            init_method=self.config.embedding_init_method,
            reduce_scatter_embeddings=self.reduce_scatter_embeddings,
            config=self.config,
            tp_group=self.tp_group,
        )

        # Position embedding (serial).
        if self.add_position_embedding:
            self.position_embeddings = paddle.nn.Embedding(
                self.max_sequence_length, self.config.hidden_size
            )

            # Initialize the position embeddings.
            if self.config.perform_initialization:
                self.config.embedding_init_method(
                    self.position_embeddings.weight
                )

        if self.num_tokentypes > 0:
            self.tokentype_embeddings = paddle.nn.Embedding(
                self.num_tokentypes, self.config.hidden_size
            )
            # Initialize the token-type embeddings.
            if self.config.perform_initialization:
                self.config.embedding_init_method(
                    self.tokentype_embeddings.weight
                )
        else:
            self.tokentype_embeddings = None

        # Embeddings dropout
        self.embedding_dropout = paddle.nn.Dropout(
            self.config.hidden_dropout_prob
        )

    @property
    def embedding_weight(self):
        return self.embed_tokens.weight

    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.embed_tokens.weight.data.fill_(0)
        self.embed_tokens.weight.shared = True
        self.position_embeddings.weight.data.fill_(0)
        self.position_embeddings.weight.shared = True
        if self.num_tokentypes > 0:
            self.tokentype_embeddings.weight.data.fill_(0)
            self.tokentype_embeddings.weight.shared = True

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        tokentype_ids: int | None = None,
    ) -> Tensor:
        """Forward pass of the embedding layer.

        Args:
            input_ids (Tensor): The input tokens
            position_ids (Tensor): The position id's used to calculate position embeddings
            tokentype_ids (int): The token type ids. Used when args.bert_binary_head is
                set to True. Defaults to None

        Returns:
            Tensor: The output embeddings
        """
        embed_tokens = self.embed_tokens(input_ids)
        if self.add_position_embedding:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embed_tokens + position_embeddings
        else:
            embeddings = embed_tokens

        if (
            not self.reduce_scatter_embeddings
            and self.sequence_parallel
            and self.scatter_to_sequence_parallel
        ):
            # Data format change to avoid explicit transposes : [b s h] --> [s b h].
            embeddings = embeddings.transpose([1, 0, 2]).contiguous()

        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            # [b s h] -> [s b h] (So that it can be added with embeddings)
            # tokentype_embedding = self.tokentype_embeddings(tokentype_ids).permute(1, 0, 2)
            tokentype_embedding = self.tokentype_embeddings(tokentype_ids)
            if self.sequence_parallel and self.scatter_to_sequence_parallel:
                tokentype_embedding = tokentype_embedding.permute(
                    1, 0, 2
                ).contiguous()
            embeddings = embeddings + tokentype_embedding
        else:
            assert self.tokentype_embeddings is None

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.config.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.sequence_parallel:
            if (
                not self.reduce_scatter_embeddings
                and self.scatter_to_sequence_parallel
            ):
                embeddings = (
                    tensor_parallel.scatter_to_sequence_parallel_region(
                        embeddings, group=self.tp_group
                    )
                )
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            if (
                self.config.clone_scatter_output_in_embedding
                and self.scatter_to_sequence_parallel
            ):
                embeddings = embeddings.clone()
            with tensor_parallel.get_cuda_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = self.embedding_dropout(embeddings)

        return embeddings
