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
"""Paddle MiMo model."""
from __future__ import annotations

from typing import Optional, Tuple, Union

import paddle
from paddle import nn

from ...nn.criterion.interface import CriterionLayer
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP as Qwen2MLP
from ...nn.norm import Norm as GeneralNorm
from ..cache_utils import Cache
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..qwen2.modeling import Qwen2Attention, Qwen2ForCausalLMPipe, Qwen2Model, Qwen2PretrainedModel
from .configuration import MiMoConfig


class MiMoMTPLayer(nn.Layer):
    def __init__(self, config: MiMoConfig):
        super().__init__()
        self.input_layernorm = GeneralNorm.create(
            config=config, norm_type="rms_norm", hidden_size=config.hidden_size, norm_eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = GeneralNorm.create(
            config=config, norm_type="rms_norm", hidden_size=config.hidden_size, norm_eps=config.rms_norm_eps
        )
        self.token_layernorm = GeneralNorm.create(
            config=config, norm_type="rms_norm", hidden_size=config.hidden_size, norm_eps=config.rms_norm_eps
        )
        self.hidden_layernorm = GeneralNorm.create(
            config=config, norm_type="rms_norm", hidden_size=config.hidden_size, norm_eps=config.rms_norm_eps
        )
        self.input_proj = GeneralLinear.create(
            config.hidden_size * 2, config.hidden_size, has_bias=False, config=config, tp_plan="colwise"
        )
        self.final_layernorm = GeneralNorm.create(
            config=config, norm_type="rms_norm", hidden_size=config.hidden_size, norm_eps=config.rms_norm_eps
        )
        self.self_attn = Qwen2Attention(config, layer_idx=0)
        self.mlp = Qwen2MLP(config, fuse_up_gate=True)

    def forward(
        self,
        input_embeds: paddle.Tensor,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> paddle.Tensor:
        input_embeds = self.token_layernorm(input_embeds)
        previous_hidden_states = self.hidden_layernorm(hidden_states)
        hidden_states = self.input_proj(paddle.concat([previous_hidden_states, input_embeds], axis=-1))

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            batch_size=batch_size,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return self.final_layernorm(hidden_states)


class MiMoPretrainedModel(Qwen2PretrainedModel):
    config_class = MiMoConfig
    base_model_prefix = "model"

    @classmethod
    def _gen_aoa_config(cls, config: MiMoConfig):
        aoa_config = super()._gen_aoa_config(config)
        model_prefix = "" if cls == cls.base_model_class else "model."
        if not config.tie_word_embeddings and not getattr(cls, "is_fleet", False):
            aoa_config["aoa_statements"] += ["lm_head.weight -> lm_head.weight"]

        for layer_id in range(config.num_nextn_predict_layers):
            mtp_prefix = f"model.mtp_layers.{layer_id}"
            paddle_prefix = f"{model_prefix}mtp_layers.{layer_id}"
            aoa_config["aoa_statements"] += [
                f"{mtp_prefix}.input_layernorm.weight -> {paddle_prefix}.input_layernorm.weight",
                f"{mtp_prefix}.post_attention_layernorm.weight -> {paddle_prefix}.post_attention_layernorm.weight",
                f"{mtp_prefix}.token_layernorm.weight -> {paddle_prefix}.token_layernorm.weight",
                f"{mtp_prefix}.hidden_layernorm.weight -> {paddle_prefix}.hidden_layernorm.weight",
                f"{mtp_prefix}.final_layernorm.weight -> {paddle_prefix}.final_layernorm.weight",
                f"{mtp_prefix}.input_proj.weight^T -> {paddle_prefix}.input_proj.weight",
                f"{mtp_prefix}.self_attn.o_proj.weight^T -> {paddle_prefix}.self_attn.o_proj.weight",
                f"{mtp_prefix}.mlp.down_proj.weight^T -> {paddle_prefix}.mlp.down_proj.weight",
                (
                    f"{mtp_prefix}.self_attn.q_proj.weight^T, {mtp_prefix}.self_attn.k_proj.weight^T, "
                    f"{mtp_prefix}.self_attn.v_proj.weight^T -> {paddle_prefix}.self_attn.qkv_proj.weight, "
                    f"fused_qkv, num_heads={config.num_attention_heads}, "
                    f"num_key_value_groups={config.num_key_value_heads}"
                ),
                (
                    f"{mtp_prefix}.self_attn.q_proj.bias, {mtp_prefix}.self_attn.k_proj.bias, "
                    f"{mtp_prefix}.self_attn.v_proj.bias -> {paddle_prefix}.self_attn.qkv_proj.bias, "
                    f"fused_qkv, num_heads={config.num_attention_heads}, "
                    f"num_key_value_groups={config.num_key_value_heads}, axis=0"
                ),
                (
                    f"{mtp_prefix}.mlp.gate_proj.weight^T, {mtp_prefix}.mlp.up_proj.weight^T -> "
                    f"{paddle_prefix}.mlp.up_gate_proj.weight, fused_ffn"
                ),
            ]
        return aoa_config

    @classmethod
    def _gen_inv_aoa_config(cls, config: MiMoConfig):
        aoa_config = super()._gen_inv_aoa_config(config)
        model_prefix = "" if cls == cls.base_model_class else "model."
        if not config.tie_word_embeddings and not getattr(cls, "is_fleet", False):
            aoa_config["aoa_statements"] += ["lm_head.weight -> lm_head.weight"]

        for layer_id in range(config.num_nextn_predict_layers):
            mtp_prefix = f"model.mtp_layers.{layer_id}"
            paddle_prefix = f"{model_prefix}mtp_layers.{layer_id}"
            aoa_config["aoa_statements"] += [
                f"{paddle_prefix}.input_layernorm.weight -> {mtp_prefix}.input_layernorm.weight",
                f"{paddle_prefix}.post_attention_layernorm.weight -> {mtp_prefix}.post_attention_layernorm.weight",
                f"{paddle_prefix}.token_layernorm.weight -> {mtp_prefix}.token_layernorm.weight",
                f"{paddle_prefix}.hidden_layernorm.weight -> {mtp_prefix}.hidden_layernorm.weight",
                f"{paddle_prefix}.final_layernorm.weight -> {mtp_prefix}.final_layernorm.weight",
                f"{paddle_prefix}.input_proj.weight^T -> {mtp_prefix}.input_proj.weight",
                f"{paddle_prefix}.self_attn.o_proj.weight^T -> {mtp_prefix}.self_attn.o_proj.weight",
                f"{paddle_prefix}.mlp.down_proj.weight^T -> {mtp_prefix}.mlp.down_proj.weight",
                (
                    f"{paddle_prefix}.self_attn.qkv_proj.weight -> {mtp_prefix}.self_attn.q_proj.weight, "
                    f"{mtp_prefix}.self_attn.k_proj.weight, {mtp_prefix}.self_attn.v_proj.weight, "
                    f"fused_qkv, num_heads={config.num_attention_heads}, "
                    f"num_key_value_groups={config.num_key_value_heads}"
                ),
                (
                    f"{paddle_prefix}.self_attn.qkv_proj.bias -> {mtp_prefix}.self_attn.q_proj.bias, "
                    f"{mtp_prefix}.self_attn.k_proj.bias, {mtp_prefix}.self_attn.v_proj.bias, "
                    f"fused_qkv, num_heads={config.num_attention_heads}, "
                    f"num_key_value_groups={config.num_key_value_heads}, axis=0"
                ),
                (
                    f"{paddle_prefix}.mlp.up_gate_proj.weight -> {mtp_prefix}.mlp.gate_proj.weight, "
                    f"{mtp_prefix}.mlp.up_proj.weight, fused_ffn"
                ),
            ]
            for proj in ("q", "k", "v"):
                aoa_config["aoa_statements"] += [
                    f"{mtp_prefix}.self_attn.{proj}_proj.weight^T -> {mtp_prefix}.self_attn.{proj}_proj.weight"
                ]
            aoa_config["aoa_statements"] += [
                f"{mtp_prefix}.mlp.gate_proj.weight^T -> {mtp_prefix}.mlp.gate_proj.weight",
                f"{mtp_prefix}.mlp.up_proj.weight^T -> {mtp_prefix}.mlp.up_proj.weight",
            ]
        return aoa_config


class MiMoModel(Qwen2Model):
    config_class = MiMoConfig

    def __init__(self, config: MiMoConfig):
        super().__init__(config)
        self.mtp_layers = nn.LayerList([MiMoMTPLayer(config) for _ in range(config.num_nextn_predict_layers)])

    def forward(self, *args, **kwargs) -> Union[Tuple, BaseModelOutputWithPast]:
        return super().forward(*args, **kwargs)


class MiMoForCausalLMPipe(Qwen2ForCausalLMPipe):
    config_class = MiMoConfig


MiMoPretrainedModel.base_model_class = MiMoModel


class MiMoForCausalLM(MiMoPretrainedModel):
    enable_to_static_method = True
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: MiMoConfig):
        super().__init__(config)
        self.model = MiMoModel(config)
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        self.tie_weights()

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        loss_mask: Optional[paddle.Tensor] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )
        logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            loss, _ = self.criterion(logits, labels)

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


class MiMoForCausalLMDeprecated(MiMoForCausalLM):
    pass


__all__ = [
    "MiMoConfig",
    "MiMoModel",
    "MiMoPretrainedModel",
    "MiMoMTPLayer",
    "MiMoForCausalLM",
    "MiMoForCausalLMPipe",
    "MiMoForCausalLMDeprecated",
]
