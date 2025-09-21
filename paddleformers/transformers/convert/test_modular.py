# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Paddle Qwen2 model inheriting from Llama2."""

from __future__ import annotations

import math
import warnings
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import paddle
import paddle.distributed as dist
import paddle.distributed.fleet.meta_parallel as mpu
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.fleet.recompute.recompute import recompute

from ..utils import get_env_device
from . import linear_utils
from ..activations import ACT2FN
from ..contrastive_loss import SimpleContrastiveLoss
from ..conversion_utils import StateDictNameMapping, init_name_mappings
from ..embedding_utils import dist_gather_tensor_with_gradient
from ..linear_utils import Linear
from ..llama import fusion_ops
from ..llama.modeling import get_use_casual_mask
from ..model_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ..model_utils import PretrainedModel, register_base_model
from ..refined_recompute import (
    RRColumnParallelLinear,
    RRColumnSequenceParallelLinear,
    RRRowParallelLinear,
    RRRowSequenceParallelLinear,
    get_skip_recompute_ops,
)
from ..refined_recompute import recompute as rr_recompute
from ..utils import caculate_llm_per_token_flops, logger
from .configuration import Qwen2Config

try:
    from paddle.incubate.nn.functional import fused_rotary_position_embedding
except ImportError:
    fused_rotary_position_embedding = None

try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        GatherOp,
        ScatterOp,
        mark_as_sequence_parallel_parameter,
    )
except:
    pass

try:
    from paddle.nn.functional.flash_attention import flash_attention
except:
    flash_attention = None

from ..llama.modeling import (
    LlamaPretrainedModel,
    LlamaModel,
    LlamaForCausalLM,
    LlamaAttention,
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)


class Qwen2RMSNorm(LlamaRMSNorm):
    """Qwen2的RMSNorm，继承自LlamaRMSNorm"""
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        # Qwen2使用不同的epsilon值
        self.variance_epsilon = config.rms_norm_eps
class Qwen2RotaryEmbedding(LlamaRotaryEmbedding):
    pass

class Qwen2MLP(LlamaMLP):
    """Qwen2的MLP，继承自LlamaMLP"""
    def __init__(self, config: Qwen2Config,is_shared=False, skip_recompute_ops=None):
        super().__init__(config)
        if config.hidden_act == "silu":
            self.act_fn = fusion_ops.swiglu
            self.fuse_swiglu = True
        else:
            self.act_fn = ACT2FN[config.hidden_act]
            self.fuse_swiglu = False
        # Qwen2的MLP结构与Llama相同，但使用不同的配置


class Qwen2Attention(LlamaAttention):
    """Qwen2的注意力机制，继承自LlamaAttention并增加模块化支持"""
    def __init__(self, config: Qwen2Config, layer_idx: int, layerwise_recompute: bool = True, skip_recompute_ops=None):
        super().__init__(config, layer_idx)
        
        # 模块化注意力配置
        self.attention_type = config.layer_types[layer_idx]
        self.sliding_window = config.sliding_window if self.attention_type == "sliding_attention" else None
        
        # Qwen2使用带偏置的线性层
        self.q_proj = nn.Linear(
            self.hidden_size, 
            self.num_heads * self.head_dim, 
            bias_attr=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size, 
            self.num_key_value_heads * self.head_dim, 
            bias_attr=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, 
            self.num_key_value_heads * self.head_dim, 
            bias_attr=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, 
            self.hidden_size, 
            bias_attr=False
        )

    def forward(
        self,
        hidden_states,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # [bs, seq_len, num_head * head_dim] -> [seq_len / n, bs, num_head * head_dim] (n is model parallelism)

        if self.fuse_attention_qkv:
            mix_layer = self.qkv_proj(hidden_states)
            if self.sequence_parallel:
                target_shape = [
                    batch_size,
                    -1,
                    self.num_key_value_heads,
                    (self.num_key_value_groups + 2) * self.head_dim,
                ]
            else:
                target_shape = [0, 0, self.num_key_value_heads, (self.num_key_value_groups + 2) * self.head_dim]
            mix_layer = paddle.reshape_(mix_layer, target_shape)
            query_states, key_states, value_states = paddle.split(
                mix_layer,
                num_or_sections=[self.num_key_value_groups * self.head_dim, self.head_dim, self.head_dim],
                axis=-1,
            )
            if self.gqa_or_mqa:
                query_states = paddle.reshape_(query_states, [0, 0, self.num_heads, self.head_dim])
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            if self.sequence_parallel:
                target_query_shape = [batch_size, -1, self.num_heads, self.head_dim]
                target_key_value_shape = [batch_size, -1, self.num_key_value_heads, self.head_dim]
            else:
                target_query_shape = [0, 0, self.num_heads, self.head_dim]
                target_key_value_shape = [0, 0, self.num_key_value_heads, self.head_dim]
            query_states = query_states.reshape(shape=target_query_shape)
            key_states = key_states.reshape(shape=target_key_value_shape)
            value_states = value_states.reshape(shape=target_key_value_shape)

        if position_ids is not None and not self.use_fused_rope:
            kv_seq_len = position_ids.max().item() + 1
        else:
            kv_seq_len = key_states.shape[-3]
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[-3]
        if self.use_fused_rope:
            assert past_key_value is None, "fuse rotary not support cache kv for now"
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states, _ = fused_rotary_position_embedding(
                query_states,
                key_states,
                v=None,
                sin=sin,
                cos=cos,
                position_ids=position_ids,
                use_neox_rotary_style=False,
            )
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # [bs, seq_len, num_head, head_dim]
        if past_key_value is not None:
            key_states = paddle.concat([past_key_value[0], key_states], axis=1)
            value_states = paddle.concat([past_key_value[1], value_states], axis=1)
        past_key_value = (key_states, value_states) if use_cache else None

        # TODO(wj-Mcat): use broadcast strategy when n_kv_heads = 1
        # repeat k/v heads if n_kv_heads < n_heads
        paddle_version = float(paddle.__version__[:3])
        if not self.config.use_flash_attention or ((paddle_version != 0.0) and (paddle_version <= 2.6)):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        has_gradient = not (query_states.stop_gradient and key_states.stop_gradient and value_states.stop_gradient)
        if (
            self.enable_recompute
            and self.layerwise_recompute
            and has_gradient
            and self.recompute_granularity == "core_attn"
        ):
            recompute_fn = rr_recompute if any(self.skip_recompute_ops.values()) else recompute
            outputs = recompute_fn(
                self.attn_func,
                query_states,
                self.config,
                key_states,
                value_states,
                attention_mask,
                output_attentions,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                training=self.training,
                sequence_parallel=self.sequence_parallel,
                use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            outputs = self.attn_func(
                query_states,
                self.config,
                key_states,
                value_states,
                attention_mask,
                output_attentions,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                training=self.training,
                sequence_parallel=self.sequence_parallel,
            )
        if output_attentions:
            attn_output, attn_weights = outputs
        else:
            attn_output = outputs

        # if sequence_parallel is true, out shape are [q_len / n, bs, num_head * head_dim]
        # else their shape are [bs, q_len, num_head * head_dim], n is mp parallelism.
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        outputs = (attn_output,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    def _create_sliding_window_mask(self, attention_mask: paddle.Tensor, window_size: int) -> paddle.Tensor:
        """创建滑动窗口注意力掩码"""
        seq_length = attention_mask.shape[-1]
        
        # 创建滑动窗口掩码
        sliding_mask = paddle.full(
            (seq_length, seq_length), 
            paddle.finfo(attention_mask.dtype).min,
            dtype=attention_mask.dtype
        )
        
        for i in range(seq_length):
            start = max(0, i - window_size + 1)
            sliding_mask[i, start:i+1] = 0
        
        # 将滑动窗口掩码添加到原始掩码上
        return attention_mask + sliding_mask.unsqueeze(0).unsqueeze(0)


class Qwen2DecoderLayer(LlamaDecoderLayer):
    """Qwen2的解码器层，继承自LlamaDecoderLayer"""
    def __init__(self, config: Qwen2Config, layer_idx: int,layerwise_recompute: bool = False, skip_recompute_ops=None):
        super().__init__(config)
        
        # 使用Qwen2特定的注意力机制
        self.self_attn = Qwen2Attention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config)
        self.post_attention_layernorm = Qwen2RMSNorm(config)
        
        # 记录注意力类型
        self.attention_type = config.layer_types[layer_idx]


class Qwen2PretrainedModel(LlamaPretrainedModel):
    """Qwen2预训练模型基类，继承自LlamaPretrainedModel"""
    config_class = Qwen2Config
    base_model_prefix = "qwen2"


@register_base_model
class Qwen2Model(LlamaModel):
    """Qwen2模型，继承自LlamaModel"""
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        
        # 替换为Qwen2的解码器层
        self.layers = nn.LayerList([
            Qwen2DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # 替换为Qwen2的RMSNorm
        self.norm = Qwen2RMSNorm(config)
        
        # 检查是否有滑动窗口层
        self.has_sliding_layers = any(
            layer.attention_type == "sliding_attention" 
            for layer in self.layers
        )

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        # 根据不同的注意力类型创建不同的掩码
        if self.has_sliding_layers and attention_mask is not None:
            attention_mask = self._create_modular_masks(attention_mask)
        
        # 调用父类的forward方法
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def _create_modular_masks(self, attention_mask: paddle.Tensor) -> dict:
        """为不同的注意力类型创建不同的掩码"""
        mask_dict = {}
        
        # 全注意力掩码
        mask_dict["full_attention"] = attention_mask
        
        # 滑动窗口注意力掩码
        if self.has_sliding_layers:
            sliding_window = self.config.sliding_window
            if sliding_window is not None:
                mask_dict["sliding_attention"] = self._create_sliding_window_mask(
                    attention_mask, sliding_window
                )
        
        return mask_dict


class Qwen2ForCausalLM(LlamaForCausalLM):
    """用于因果语言建模的Qwen2模型，继承自LlamaForCausalLM"""
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.qwen2 = Qwen2Model(config)


class Qwen2ForSequenceClassification(Qwen2PretrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.qwen2 = Qwen2Model(config)
        self.score = Linear(config.hidden_size, self.num_labels, bias_attr=False)

    def get_input_embeddings(self):
        return self.qwen2.embed_tokens

    def set_input_embeddings(self, value):
        self.qwen2.embed_tokens = value

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`paddle.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.qwen2(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = paddle.equal(input_ids, self.config.pad_token_id).astype("int32").argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths
            else:
                sequence_lengths = -1

        # pooled_logits = logits[paddle.arange(batch_size), sequence_lengths]
        pooled_logits = logits.gather_nd(paddle.stack([paddle.arange(logits.shape[0]), sequence_lengths], axis=-1))

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == paddle.int64 or labels.dtype == paddle.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(pooled_logits.reshape([-1, self.num_labels]), labels.reshape([-1]))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


# Copied from transformers.models.llama.modeling_llama.LlamaForTokenClassification with Llama->Qwen2, LLAMA->QWEN2
class Qwen2ForTokenClassification(Qwen2PretrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.qwen2 = Qwen2Model(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = Linear(config.hidden_size, config.num_labels)

    def get_input_embeddings(self):
        return self.qwen2.embed_tokens

    def set_input_embeddings(self, value):
        self.qwen2.embed_tokens = value

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
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`paddle.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.qwen2(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape([-1, self.num_labels]), labels.reshape([-1]))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Qwen2SentenceEmbedding(Qwen2PretrainedModel):
    def __init__(
        self,
        config: Qwen2Config,
        embedding_temperature: float = 0.02,
    ):
        """Qwen2SentenceEmbedding
        For getting larger batch_size, we use tensor parallel to get larger batch_size.

        Args:
            config (Qwen2Config): _description_
            model (Qwen2Model): _description_
            embedding_temperature (float, optional): _description_. Defaults to 0.02.
        """
        super(Qwen2SentenceEmbedding, self).__init__(config)
        self.config = config
        self.qwen2 = Qwen2Model(config)
        self.in_batch_negative_loss = SimpleContrastiveLoss(embedding_temperature)
        self.world_size = dist.get_world_size()
        self.process_rank = dist.get_rank()
        self.embedding_negatives_cross_device = config.embedding_negatives_cross_device
        if self.world_size <= 1:
            self.embedding_negatives_cross_device = False

    def forward(
        self,
        query: Optional[Dict[str, paddle.Tensor]] = None,
        passages: Optional[Dict[str, paddle.Tensor]] = None,
        return_encode=False,
    ):
        """forward"""
        q_reps = self.encode(**query)
        p_reps = self.encode(**passages)

        q_reps = nn.functional.normalize(q_reps, axis=-1)
        p_reps = nn.functional.normalize(p_reps, axis=-1)

        if return_encode:
            return q_reps, p_reps

        if self.embedding_negatives_cross_device:
            q_reps = dist_gather_tensor_with_gradient(q_reps)
            p_reps = dist_gather_tensor_with_gradient(p_reps)

        loss = self.in_batch_negative_loss(q_reps, p_reps)
        return loss

    def encode(
        self,
        input_ids,
        position_ids=None,
        embedding_indices=None,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        **kwargs,
    ):
        """encode"""
        input_type = type(input_ids)
        outputs = self.qwen2(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        if isinstance(outputs, input_type):
            hidden_states = outputs
        else:
            hidden_states = outputs[0]
        last_hidden_states = hidden_states.gather_nd(embedding_indices)
        return last_hidden_states


# 模型注册
__all__ = [
    "Qwen2PretrainedModel",
    "Qwen2Model",
    "Qwen2ForCausalLM",
    "Qwen2ForSequenceClassification",
    "Qwen2ForTokenClassification",
]