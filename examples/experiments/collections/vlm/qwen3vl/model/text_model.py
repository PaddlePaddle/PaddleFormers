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
from paddlefleet.transformer.transformer_layer import TransformerLayer


class Qwen3VLTextLayer(TransformerLayer):
    """Qwen3VL text model for adapt deepstack process"""
    def _forward_impl(
        self,
        dict_args: dict
    ):
        deepstack_visual_emb = dict_args.pop("deepstack_visual_emb", None)
        visual_pos_masks = dict_args.pop("visual_pos_masks", None)
        hidden_states, context = self._forward_attention(**dict_args)
        hidden_states = self._forward_mlp(hidden_states)
        if self.layer_number in range(len(deepstack_visual_emb)):
            hidden_states = self._deepstack_process(
                hidden_states=hidden_states,
                visual_embeds=deepstack_visual_emb[self.layer_number],
                visual_pos_masks=visual_pos_masks
            )
        rst = {"hidden_states": hidden_states}
        if context is not None:
            rst["context"] = context
        rst = {**dict_args, **rst}
        return rst
    
    def _deepstack_process(
        self, hidden_states: paddle.Tensor, visual_pos_masks: paddle.Tensor, visual_embeds: paddle.Tensor
    ):
        visual_embeds = visual_embeds.to(hidden_states.dtype)
        hidden_states = hidden_states.clone()
        local_this = hidden_states[visual_pos_masks, :] + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states
