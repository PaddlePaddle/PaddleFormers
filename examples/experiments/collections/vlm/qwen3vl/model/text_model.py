# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from paddlefleet.packed_seq_params import PackedSeqParams
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.transformer.transformer_layer import TransformerLayer, TransformerLayerSublayersSpec


class Qwen3VLTextTransformerLayer(TransformerLayer):
    """Qwen3VL text model for adapt deepstack process"""
    
    def _forward_impl(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor = None,
        attn_mask_startend_row_indices: paddle.Tensor = None,
        context: paddle.Tensor = None,
        context_mask: paddle.Tensor = None,
        rotary_pos_emb: paddle.Tensor = None,
        rotary_pos_cos: paddle.Tensor = None,
        rotary_pos_sin: paddle.Tensor = None,
        attention_bias: paddle.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        deepstack_visual_emb: list[paddle.Tensor] = None,
        visual_pos_masks: paddle.Tensor = None,
    ):
        hidden_states, context = self._forward_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            context=context,
            context_mask=context_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )
        hidden_states = self._forward_mlp(hidden_states)
        if self.layer_number in range(len(deepstack_visual_emb)):
            hidden_states = self._deepstack_process(
                hidden_states=hidden_states,
                visual_embeds=deepstack_visual_emb[self.layer_number],
                visual_pos_masks=visual_pos_masks
            )
        if context is not None:
            return hidden_states, context
        return hidden_states
    
    def _deepstack_process(
        self, hidden_states: paddle.Tensor, visual_pos_masks: paddle.Tensor, visual_embeds: paddle.Tensor
    ):
        # Store original shape and flatten hidden_states to 2D [B*S, D]
        original_shape = hidden_states.shape
        if hidden_states.ndim > 2:
            hidden_states = hidden_states.flatten(start_axis=0, stop_axis=1)

        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)

        # complicated logic for squential parallelism
        if visual_pos_masks.ndim > 1:
            visual_pos_masks = visual_pos_masks.flatten()

        # This block handles Sequence Parallelism (Row Slicing)
        if visual_pos_masks.shape[0] > hidden_states.shape[0]:
            try:
                from paddle.distributed.fleet import get_hybrid_communicate_group
                
                hcg = get_hybrid_communicate_group()
                mp_rank = hcg.get_model_parallel_rank()
                mp_size = hcg.get_model_parallel_world_size()
            except (ImportError, AttributeError):
                mp_size = visual_pos_masks.shape[0] // hidden_states.shape[0]
                mp_rank = paddle.distributed.get_rank() % mp_size
            total_len = visual_pos_masks.shape[0]
            chunk_size = total_len // mp_size
            start_idx = mp_rank * chunk_size
            end_idx = start_idx + chunk_size
            if start_idx > 0:
                pre_mask = visual_pos_masks[:start_idx]
                visual_offset = paddle.sum(paddle.cast(pre_mask, "int32")).item()
            else:
                visual_offset = 0
            local_mask = visual_pos_masks[start_idx:end_idx]
            local_visual_count = paddle.sum(paddle.cast(local_mask, "int32")).item()

            visual_embeds = visual_embeds[visual_offset : visual_offset + local_visual_count]
            visual_pos_masks = local_mask

        # If TP is enabled, hidden_states has shape [..., Hidden_Dim / TP_Size],
        # but visual_embeds usually has full [Hidden_Dim]. We need to slice visual_embeds column-wise.
        if hidden_states.shape[-1] != visual_embeds.shape[-1]:
            try:
                from paddle.distributed.fleet import get_hybrid_communicate_group

                hcg = get_hybrid_communicate_group()
                tp_rank = hcg.get_model_parallel_rank()
                tp_size = hcg.get_model_parallel_world_size()
            except (ImportError, AttributeError):
                # Fallback simple estimation
                tp_size = visual_embeds.shape[-1] // hidden_states.shape[-1]
                tp_rank = paddle.distributed.get_rank() % tp_size

            if tp_size > 1:
                embed_dim = visual_embeds.shape[-1]
                slice_width = embed_dim // tp_size
                start_col = tp_rank * slice_width
                end_col = start_col + slice_width
                visual_embeds = visual_embeds[:, start_col:end_col]

        hidden_states = hidden_states.clone()
        local_this = hidden_states[visual_pos_masks, :] + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this  # 这个操作可能会导致paddle转静态图或推理时出问题，建议使用 scatter

        # [Supplement 3] Restore original shape [B*S, D] -> [B, S, D] if necessary
        if len(original_shape) > 2:
            hidden_states = hidden_states.reshape(original_shape)

        return hidden_states