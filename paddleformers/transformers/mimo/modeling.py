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
from paddle import Tensor, nn
from paddle.distributed.fleet.meta_parallel import LayerDesc
from paddle.distributed.fleet.recompute.recompute import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp

from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP as Qwen2MLP
from ...nn.norm import Norm as GeneralNorm
from ...nn.pp_model import GeneralModelForCausalLMPipe
from ...utils.log import logger
from ..cache_utils import Cache, DynamicCache
from ..masking_utils import (
    create_causal_mask_and_row_indices,
    create_sliding_window_causal_mask_and_row_indices,
)
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_utils import register_base_model
from ..qwen2.modeling import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2PretrainedModel,
    Qwen2RotaryEmbedding,
)
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


class MiMoMTPWeights(nn.LayerList):
    """Frozen checkpoint tensors; no auxiliary objective or token shift.

    Storage uses config.num_mtp_layers. Generic training and PP continue to
    use num_nextn_predict_layers=0. This identity stage preserves the tensors
    in pipeline checkpoints without changing the backbone activations.
    """

    def __init__(self, config):
        super().__init__([MiMoMTPLayer(config) for _ in range(config.num_mtp_layers)])
        for parameter in self.parameters():
            parameter.stop_gradient = True

    def forward(self, hidden_states):
        return hidden_states


class MiMoPretrainedModel(Qwen2PretrainedModel):
    config_class = MiMoConfig
    base_model_prefix = "model"

    @classmethod
    def _gen_aoa_config(cls, config: MiMoConfig):
        aoa_config = super()._gen_aoa_config(config)
        model_prefix = "" if cls == cls.base_model_class else "model."

        for layer_id in range(config.num_mtp_layers):
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

        for layer_id in range(config.num_mtp_layers):
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
        # Materialize macro outputs before serialization. The macros use
        # intermediate names for grouped K/V and fused FFN projections.
        for layer_id in range(config.num_hidden_layers):
            for projection in ("k", "v"):
                key = f"model.layers.{layer_id}.self_attn.{projection}_proj.bias"
                aoa_config["aoa_statements"].append(f"{key} -> {key}")
        for layer_id in range(config.num_mtp_layers):
            prefix = f"model.mtp_layers.{layer_id}"
            for projection in ("q", "k", "v"):
                key = f"{prefix}.self_attn.{projection}_proj.weight"
                aoa_config["aoa_statements"].append(f"{key}^T -> {key}")
            for projection in ("k", "v"):
                key = f"{prefix}.self_attn.{projection}_proj.bias"
                aoa_config["aoa_statements"].append(f"{key} -> {key}")
            for projection in ("gate", "up"):
                key = f"{prefix}.mlp.{projection}_proj.weight"
                aoa_config["aoa_statements"].append(f"{key}^T -> {key}")
        return aoa_config


@register_base_model
class MiMoModel(MiMoPretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: MiMoConfig
    """

    def __init__(self, config: MiMoConfig):
        super().__init__(config)
        self.embed_tokens = GeneralEmbedding.create(
            config=config, num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = nn.LayerList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.mtp_layers = MiMoMTPWeights(config)
        self.has_sliding_layers = getattr(
            self.config, "sliding_window", None
        ) is not None and "sliding_attention" in getattr(self.config, "layer_types", [])

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: Tensor,
        attention_mask: Tensor,
        past_key_values: Cache,
        use_cache: bool,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attn_mask_startend_row_indices=None,
        batch_size: int = None,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            past_key_values,
            use_cache,
            position_embeddings,
            attn_mask_startend_row_indices,
            batch_size,
        )

        return hidden_states

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            # [bs, seq_len, dim]
            inputs_embeds = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        cache_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = paddle.arange(cache_length, cache_length + seq_length, dtype="int64").expand(
                (batch_size, seq_length)
            )

        if self.config.sequence_parallel:
            # [bs, seq_len, num_head * head_dim] -> [bs * seq_len, num_head * head_dim]
            inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
            # [seq_len * bs / n, num_head * head_dim] (n is mp parallelism)
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        # Prepare mask arguments
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "cache_length": cache_length,
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "prepare_decoder_attention_mask": self._prepare_decoder_attention_mask,
        }
        # Create the causal mask and row indices
        full_mask, full_indices = create_causal_mask_and_row_indices(**mask_kwargs)

        causal_mask_mapping = {"full_attention": full_mask}
        attn_mask_startend_row_indices_mapping = {"full_attention": full_indices}

        # if model has sliding layer
        if self.has_sliding_layers:
            (
                causal_mask_mapping["sliding_attention"],
                attn_mask_startend_row_indices_mapping["sliding_attention"],
            ) = create_sliding_window_causal_mask_and_row_indices(**mask_kwargs)
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for idx, (decoder_layer) in enumerate(self.layers):
            has_gradient = not hidden_states.stop_gradient
            if (
                self.config.recompute_granularity == "full"
                and self.config.recompute_method == "uniform"
                and self.config.recompute_num_layers == 1
                and has_gradient
            ):
                hidden_states = self.recompute_training_full(
                    decoder_layer,
                    hidden_states,
                    causal_mask_mapping[decoder_layer.attention_type],
                    past_key_values,
                    use_cache,
                    position_embeddings,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices_mapping[
                        decoder_layer.attention_type
                    ],
                    batch_size=batch_size,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    causal_mask_mapping[decoder_layer.attention_type],
                    past_key_values,
                    use_cache,
                    position_embeddings,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices_mapping[
                        decoder_layer.attention_type
                    ],
                    batch_size=batch_size,
                )

        hidden_states = self.norm(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


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

        if attn_mask_startend_row_indices is not None and attention_mask is not None:
            logger.warning("Both masks were supplied; using attn_mask_startend_row_indices.")
            attention_mask = None

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
            loss, _ = self.criterion(logits, labels, loss_mask=loss_mask)

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


class MiMoForCausalLMPipe(GeneralModelForCausalLMPipe):
    config_class = MiMoConfig
    _decoder_layer_cls = Qwen2DecoderLayer
    _init_weights = MiMoModel._init_weights
    _keep_in_fp32_modules = MiMoModel._keep_in_fp32_modules
    _rotary_emb_cls = Qwen2RotaryEmbedding
    transpose_weight_keys = MiMoModel.transpose_weight_keys
    _gen_aoa_config = MiMoForCausalLM._gen_aoa_config
    _gen_inv_aoa_config = MiMoForCausalLM._gen_inv_aoa_config

    def add_sequential_layer(self, layer_desc, name_prefix=""):
        # GeneralModelForCausalLMPipe registers the final projection as
        # "lm_head". Insert the frozen MTP checkpoint stage immediately before
        # it; keep this hook aligned if the parent changes its name or order.
        if name_prefix == "lm_head" and self.config.num_mtp_layers:
            super().add_sequential_layer(LayerDesc(MiMoMTPWeights, config=self.config), "model.mtp_layers")
        super().add_sequential_layer(layer_desc, name_prefix)


__all__ = [
    "MiMoModel",
    "MiMoPretrainedModel",
    "MiMoMTPLayer",
    "MiMoForCausalLM",
    "MiMoForCausalLMPipe",
]
