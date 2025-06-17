# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2023 DeepSeek. All rights reserved.
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
"""Paddle DeepSeek_V3 model."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import paddle
import paddle.distributed as dist

try:
    from paddle.incubate.nn.functional import fused_rotary_position_embedding
except ImportError:
    fused_rotary_position_embedding = None

try:
    from paddle.nn.functional.flash_attention import flash_attention
except:
    flash_attention = None

from ...utils.log import logger
from ..deepseek_v2.modeling_auto import (
    DeepseekV2LMHeadAuto,
    DeepseekV2ModelAuto,
    DeepseekV2PretrainedModelAuto,
    DeepseekV2PretrainingCriterion,
)
from ..model_outputs import CausalLMOutputWithPast
from ..model_utils import register_base_model
from .configuration import DeepseekV2Config

__all__ = [
    "DeepseekV3LMHeadAuto",
    "DeepseekV3ForCausalLMAuto",
    "DeepseekV3ModelAuto",
    "DeepseekV3PretrainedModelAuto",
]


class DeepseekV3PretrainedModelAuto(DeepseekV2PretrainedModelAuto):
    config_class = DeepseekV2Config
    base_model_prefix = "deepseek_v3"
    _no_split_modules = ["DeepseekV2DecoderLayerAuto"]


@register_base_model
class DeepseekV3ModelAuto(DeepseekV2ModelAuto):
    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)


class DeepseekV3LMHeadAuto(DeepseekV2LMHeadAuto):
    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)


class DeepseekV3ForCausalLMAuto(DeepseekV3PretrainedModelAuto):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)
        self.config = config
        self.deepseek_v3 = DeepseekV3ModelAuto(config)
        self.vocab_size = config.vocab_size
        self.lm_head = DeepseekV3LMHeadAuto(config)
        self.criterion = DeepseekV2PretrainingCriterion(config)

    def get_input_embeddings(self):
        return self.deepseek_v3.embed_tokens

    def set_input_embeddings(self, value):
        self.deepseek_v3.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.deepseek_v3 = decoder

    def get_decoder(self):
        return self.deepseek_v3

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, DeepseekV3ForCausalLMAuto

        >>> model = DeepseekV3ForCausalLMAuto.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        input_ids.stop_gradient = True
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attn_mask_startend_row_indices is not None and attention_mask is not None:
            logger.warning(
                "You have provided both attn_mask_startend_row_indices and attention_mask. "
                "The attn_mask_startend_row_indices will be used."
            )
            attention_mask = None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.deepseek_v3(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )

        hidden_states = outputs[0]
        mtp_outputs = outputs[-1]

        # if labels is None，means we need full output, instead of tensor_parallel_output
        # tensor_parallel_output is together with ParallelCrossEntropy
        tensor_parallel_output = self.config.tensor_parallel_output and self.config.tensor_parallel_degree > 1

        logits = self.lm_head(hidden_states, tensor_parallel_output=tensor_parallel_output)
        mtp_logits = [self.lm_head(_hidden_states) for _hidden_states in mtp_outputs] if len(mtp_outputs) > 0 else []

        return self.criterion(logits, labels, mtp_logits=mtp_logits)

    def auto_dist_config(self, prefix=""):
        if prefix != "":
            assert prefix.endswith(".")
        config = {
            "dp_config": {"sharding_level": 0, "offload": False, "exclude_layer": None},
            "pp_config": {
                "split_spec": [f"{prefix}deepseek_v3.layers", f"{prefix}lm_head"],
                "global_spec": "deepseek_v3.global_layer",
            },
            "mp_config": {
                "parallelize_plan": {
                    f"{prefix}deepseek_v3.embed_tokens": dist.ColWiseParallel(gather_output=True),
                    f"{prefix}deepseek_v3.layers.*.self_attn.q_b_proj": dist.ColWiseParallel(),
                    f"{prefix}deepseek_v3.layers.*.self_attn.q_proj": dist.ColWiseParallel(),
                    f"{prefix}deepseek_v3.layers.*.self_attn.kv_b_proj": dist.ColWiseParallel(),
                    f"{prefix}deepseek_v3.layers.*.self_attn.o_proj": dist.RowWiseParallel(),
                    f"{prefix}deepseek_v3.layers.*.mlp.gate_proj": dist.ColWiseParallel(),
                    f"{prefix}deepseek_v3.layers.*.mlp.up_proj": dist.ColWiseParallel(),
                    f"{prefix}deepseek_v3.layers.*.mlp.down_proj": dist.RowWiseParallel(),
                    f"{prefix}deepseek_v3.layers.*.mlp.shared_experts.gate_proj": dist.ColWiseParallel(),
                    f"{prefix}deepseek_v3.layers.*.mlp.shared_experts.up_proj": dist.ColWiseParallel(),
                    f"{prefix}deepseek_v3.layers.*.mlp.shared_experts.down_proj": dist.RowWiseParallel(),
                    f"{prefix}lm_head.weight": dist.ColWiseParallel(),
                }
            },
        }
        return config
