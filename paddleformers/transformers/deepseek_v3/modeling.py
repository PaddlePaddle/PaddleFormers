# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
"""Paddle DeepSeek model."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import paddle

from ..deepseek_v2.modeling import (
    DeepseekV2ForSequenceClassification,
    DeepseekV2LMHead,
    DeepseekV2Model,
    DeepseekV2PretrainedModel,
    DeepseekV2PretrainingCriterion,
)
from ..model_outputs import CausalLMOutputWithPast
from ..model_utils import register_base_model
from .configuration import DeepseekV3Config

__all__ = [
    "DeepseekV3ForCausalLM",
    "DeepseekV3ForSequenceClassification",
    "DeepseekV3Model",
    "DeepseekV3PretrainedModel",
]


class DeepseekV3PretrainedModel(DeepseekV2PretrainedModel):
    config_class = DeepseekV3Config
    base_model_prefix = "deepseek_v3"
    _no_split_modules = ["DeepseekV2DecoderLayer"]


@register_base_model
class DeepseekV3Model(DeepseekV2Model):
    def __init__(self, config: DeepseekV3Config):
        super().__init__(config)


class DeepseekV3ForCausalLM(DeepseekV3PretrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: DeepseekV3Config):
        super().__init__(config)
        self.deepseek_v3 = DeepseekV3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = DeepseekV2LMHead(config)
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
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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
        >>> from transformers import AutoTokenizer, DeepseekV3ForCausalLM

        >>> model = DeepseekV3ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.deepseek_v3(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        mtp_outputs = outputs[-1]

        logits = self.lm_head(hidden_states)
        mtp_logits = [self.lm_head(_hidden_states) for _hidden_states in mtp_outputs] if len(mtp_outputs) > 0 else []

        loss = None
        # TODO@DrownFish19: shift labels
        if labels is not None:
            loss = self.criterion(logits, labels, mtp_logits=mtp_logits)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DeepseekV3ForSequenceClassification(DeepseekV2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
