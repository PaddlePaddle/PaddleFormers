# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from paddle import nn

from ..model_outputs import BaseModelOutputWithPooling
from ..qwen3_vl.modeling import Qwen3VLVisionModel
from .configuration import Qwen3_5VisionConfig


class Qwen3_5VisionModel(Qwen3VLVisionModel):
    config_class = Qwen3_5VisionConfig
    _no_split_modules = ["Qwen3VLVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        if not hasattr(self, "pos_embed"):
            self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        del self.deepstack_visual_indexes
        del self.deepstack_merger_list

    def forward(self, hidden_states: paddle.Tensor, grid_thw: paddle.Tensor, **kwargs) -> paddle.Tensor:
        """
        Args:
            hidden_states (`paddle.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`paddle.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `paddle.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.reshape([seq_len, -1])
        rotary_pos_emb = rotary_pos_emb.reshape([seq_len, -1])
        emb = paddle.concat([rotary_pos_emb, rotary_pos_emb], axis=-1)
        position_embeddings = (paddle.cos(emb), paddle.sin(emb))

        cu_seqlens = paddle.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            axis=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - paddle.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            dtype=grid_thw.dtype if not paddle.in_dynamic_mode() else "int32",
        )
        cu_seqlens = F.pad(cu_seqlens, [1, 0], value=0)

        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        indices_per_segment = paddle.stack(
            [
                cu_seqlens[1:],
                paddle.full_like(cu_seqlens[1:], cu_seqlens[-1]),
                paddle.zeros_like(cu_seqlens[:-1]),
                cu_seqlens[:-1],
            ],
            axis=1,
        )
        attn_mask_startend_row_indices = paddle.repeat_interleave(indices_per_segment, lengths, axis=0)[
            None, None, ...
        ]

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                **kwargs,
            )

        merged_hidden_states = self.merger(hidden_states)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )


__all__ = ["Qwen3_5VisionModel"]
