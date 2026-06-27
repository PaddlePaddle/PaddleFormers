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

"""MTP Embedding Layer for Magic Send.

Re-embeds broadcasted input_ids at the last pipeline stage.
Weights are shared with the "embed" layer.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import TYPE_CHECKING

from paddleformers.fleet.models.gpt.utils import fill_feature
from paddleformers.fleet.tensor_parallel import VocabParallelEmbedding
from paddleformers.fleet.transformer.layer import FleetLayer

if TYPE_CHECKING:
    from paddleformers.fleet.transformer.transformer_config import TransformerConfig

# Global deque holding one input_ids tensor per micro-batch.
# Populated by DistDataLoader.__next__() after broadcast,
# consumed by MTPEmbeddingLayer.forward() via popleft().
input_ids_for_mtp = deque()
from paddleformers.fleet.context_parallel_utils import (
    mark_context_parallel_parameter_disable_scale_grad,
)


class MTPEmbeddingLayer(FleetLayer):
    """MTP re-embedding layer for magic send mechanism.

    Pops input_ids from the global input_ids_for_mtp deque, re-embeds them,
    and stores the result in dict_args["mtp_input_embeds"].
    Weights are shared with the "embed" layer via weight tying in GPTModel.__init__.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)
        self.config = config

        # Create VocabParallelEmbedding aligned with GPTEmbedding config.
        # Weight will be replaced with the shared "embed" layer weight in GPTModel.__init__.
        # Skip weight initialization to avoid advancing the model-parallel RNG tracker,
        # which would cause downstream layers (e.g. MTP eh_proj) to get different
        # random seeds compared to the non-magic-send path.
        no_init_config = copy.copy(config)
        no_init_config.perform_initialization = False
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            init_method=config.embedding_init_method,
            reduce_scatter_embeddings=False,  # MTP does not need SP scatter
            config=no_init_config,
        )
        if self.config.context_parallel_size > 1:
            mark_context_parallel_parameter_disable_scale_grad(
                self.embed_tokens
            )

    @property
    def embedding_weight(self):
        """Expose embedding weight for weight tying."""
        return self.embed_tokens.weight

    def forward(self, dict_args: dict):
        """Pop input_ids from deque, embed them, store in dict_args["mtp_input_embeds"]."""
        if not self.config.enable_mtp_magic_send:
            raise RuntimeError(
                "MTPEmbeddingLayer should only be used when enable_mtp_magic_send=True"
            )

        global input_ids_for_mtp
        if len(input_ids_for_mtp) == 0:
            raise RuntimeError(
                "input_ids_for_mtp deque is empty, broadcast may have failed"
            )

        input_ids = input_ids_for_mtp.popleft()
        input_embeds = self.embed_tokens(input_ids).astype(
            self.embed_tokens.weight.dtype
        )

        # Zero out padding-token embeddings to match GPTEmbedding behavior.
        # GPTEmbedding applies fill_feature(decoder_input, input_ids==0, 0) when
        # expert_model_parallel_size > 1 and tensor_model_parallel_size < 2.
        # Without this, the shifted embeddings used by MTP differ from non-magic-send.
        if (
            self.config.expert_model_parallel_size > 1
            and self.config.tensor_model_parallel_size < 2
        ):
            pad_token_id = getattr(self.config, "pad_token_id", 0)
            if pad_token_id is None:
                pad_token_id = 0
            text_padding_indices = input_ids == pad_token_id
            input_embeds = fill_feature(input_embeds, text_padding_indices, 0)

        dict_args["mtp_input_embeds"] = input_embeds
        return dict_args
