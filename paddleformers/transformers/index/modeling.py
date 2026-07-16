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

import paddle
from paddle import nn

from ..cache_utils import Cache
from ..llama.modeling import LlamaModel, LlamaPretrainedModel
from ..model_outputs import CausalLMOutputWithPast
from ..model_utils import register_base_model
from .configuration import IndexConfig


class NormHead(nn.Layer):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[vocab_size, hidden_size],
            default_initializer=nn.initializer.KaimingUniform(),
        )
        self.first_flag = True

    def forward(self, hidden_states):
        if self.training:
            norm_weight = paddle.nn.functional.normalize(self.weight, axis=-1)
            self.first_flag = True
        elif self.first_flag:
            self.first_flag = False
            self.weight.set_value(paddle.nn.functional.normalize(self.weight, axis=-1))
            norm_weight = self.weight
        else:
            norm_weight = self.weight
        return paddle.matmul(hidden_states, norm_weight, transpose_y=True)


class IndexPretrainedModel(LlamaPretrainedModel):
    config_class = IndexConfig
    base_model_prefix = "model"
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    @classmethod
    def _gen_aoa_config(cls, config: IndexConfig):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""
        statements = [
            f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
            f"model.norm.weight -> {model_prefix}norm.weight",
            f"model.layers.$LAYER_ID.input_layernorm.weight -> {model_prefix}layers.$LAYER_ID.input_layernorm.weight",
            f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
        ]
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            statements.append(
                f"model.layers.$LAYER_ID.self_attn.{name}.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.{name}.weight"
            )
        for name in ["gate_proj", "up_proj", "down_proj"]:
            statements.append(
                f"model.layers.$LAYER_ID.mlp.{name}.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.{name}.weight"
            )
        if cls != cls.base_model_class:
            lm_head_statement = "lm_head.weight -> lm_head.weight"
            if not config.norm_head:
                lm_head_statement = "lm_head.weight^T -> lm_head.weight"
            statements.append(lm_head_statement)
        return {"aoa_statements": statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config: IndexConfig):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""
        statements = [
            f"{model_prefix}embed_tokens.weight -> model.embed_tokens.weight",
            f"{model_prefix}norm.weight -> model.norm.weight",
            f"{model_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.layers.$LAYER_ID.input_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.layers.$LAYER_ID.post_attention_layernorm.weight",
        ]
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            statements.append(
                f"{model_prefix}layers.$LAYER_ID.self_attn.{name}.weight^T -> model.layers.$LAYER_ID.self_attn.{name}.weight"
            )
        for name in ["gate_proj", "up_proj", "down_proj"]:
            statements.append(
                f"{model_prefix}layers.$LAYER_ID.mlp.{name}.weight^T -> model.layers.$LAYER_ID.mlp.{name}.weight"
            )
        if cls != cls.base_model_class:
            lm_head_statement = "lm_head.weight -> lm_head.weight"
            if not config.norm_head:
                lm_head_statement = "lm_head.weight^T -> lm_head.weight"
            statements.append(lm_head_statement)
        return {"aoa_statements": statements}


@register_base_model
class IndexModel(LlamaModel, IndexPretrainedModel):
    config_class = IndexConfig

    def forward(self, *args, use_cache=None, output_hidden_states=None, return_dict=None, **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        outputs = super().forward(
            *args,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        if return_dict:
            return outputs

        output = (outputs.last_hidden_state,)
        if use_cache:
            output += (outputs.past_key_values,)
        if output_hidden_states:
            output += (outputs.hidden_states,)
        return output


class IndexForCausalLM(IndexPretrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: IndexConfig):
        super().__init__(config)
        self.config = config
        self.model = IndexModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = (
            NormHead(config.hidden_size, config.vocab_size)
            if config.norm_head
            else nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)
        )
        self.tie_weights()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def forward(
        self,
        input_ids: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        attention_mask: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        inputs_embeds: paddle.Tensor | None = None,
        labels: paddle.Tensor | None = None,
        loss_mask: paddle.Tensor | None = None,
        use_cache: bool | None = None,
        past_key_values: Cache | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        logits = self.lm_head(outputs[0])
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            valid = shift_labels != -100
            if loss_mask is not None:
                valid = paddle.logical_and(valid, loss_mask[:, 1:].astype("bool"))
            token_loss = paddle.nn.functional.cross_entropy(
                shift_logits.reshape([-1, self.vocab_size]),
                shift_labels.reshape([-1]),
                ignore_index=-100,
                reduction="none",
            )
            valid = valid.reshape([-1]).astype(token_loss.dtype)
            valid_count = paddle.sum(valid)
            loss = paddle.sum(token_loss * valid) / paddle.clip(valid_count, min=1.0)
        if not return_dict:
            output = (logits,) + tuple(v for v in outputs[1:] if v is not None)
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        position_ids = kwargs.get("position_ids")
        if attention_mask is not None and position_ids is None:
            position_ids = paddle.cumsum(attention_mask.astype("int64"), axis=-1) - 1
            position_ids = paddle.where(attention_mask == 0, paddle.ones_like(position_ids), position_ids)
            if past_key_values is not None:
                position_ids = position_ids[:, -1:]
        model_inputs = (
            {"inputs_embeds": inputs_embeds}
            if inputs_embeds is not None and past_key_values is None
            else {"input_ids": input_ids}
        )
        model_inputs.update(
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=kwargs.get("use_cache"),
            attention_mask=attention_mask,
        )
        return model_inputs


__all__ = ["IndexPretrainedModel", "IndexModel", "IndexForCausalLM", "NormHead"]
