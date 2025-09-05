# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from functools import partial
from typing import List, Optional, Tuple, Union

import paddle
from paddle import Tensor, nn
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp
from paddle.nn import functional as F

from ...nn.attention.utils import repeat_kv
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP as Glm4MoeMLP
from ..conversion_utils import StateDictNameMapping, init_name_mappings
from ..llama.modeling import get_use_casual_mask
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from ..qwen2.modeling import is_casual_mask
from .configuration import Glm4MoeConfig


def eager_attention_forward(
    module: nn.Layer,
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    attention_mask: Optional[paddle.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    perm = [0, 2, 1, 3]  # b l h d -> b h l d
    query = paddle.transpose(x=query, perm=perm)
    key = paddle.transpose(x=key, perm=perm)
    value = paddle.transpose(x=value, perm=perm)

    attn_weights = paddle.matmul(query, key.transpose([0, 1, 3, 2])) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=paddle.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = paddle.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)
    attn_output = paddle.reshape(x=attn_output, shape=[0, 0, attn_output.shape[2] * attn_output.shape[3]])

    return attn_output, attn_weights


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat((-x2, x1), axis=-1)


def _apply_rotary_emb(
    x: paddle.Tensor,
    cos: paddle.Tensor,
    sin: paddle.Tensor,
) -> paddle.Tensor:
    x = x.transpose([0, 2, 1, 3])
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed.transpose([0, 2, 1, 3])


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = _apply_rotary_emb(q_rot, cos, sin)
    k_embed = _apply_rotary_emb(k_rot, cos, sin)

    # Concatenate back to full shape
    q_embed = paddle.concat([q_embed, q_pass], axis=-1)
    k_embed = paddle.concat([k_embed, k_pass], axis=-1)

    return q_embed, k_embed


class Glm4MoeAttention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Glm4MoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.rope_scaling = config.rope_scaling
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.sequence_parallel = config.sequence_parallel
        self.attention_bias = config.attention_bias
        self.fuse_attention_qkv = config.fuse_attention_qkv
        self.attn_implementation = config._attn_implementation

        if config.tensor_parallel_degree > 1:
            assert (
                self.num_heads % config.tensor_parallel_degree == 0
            ), f"num_heads: {self.num_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_heads = self.num_heads // config.tensor_parallel_degree
            assert (
                self.num_key_value_heads % config.tensor_parallel_degree == 0
            ), f"num_key_value_heads: {self.num_key_value_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_key_value_heads = self.num_key_value_heads // config.tensor_parallel_degree

        kv_hidden_size = self.config.num_key_value_heads * self.head_dim
        q_hidden_size = self.num_attention_heads * self.head_dim

        self.q_proj = GeneralLinear.create(
            self.hidden_size,
            q_hidden_size,
            has_bias=self.attention_bias,
            config=config,
            tp_plan="colwise",
        )
        self.k_proj = GeneralLinear.create(
            self.hidden_size,
            kv_hidden_size,
            has_bias=self.attention_bias,
            config=config,
            tp_plan="colwise",
        )
        self.v_proj = GeneralLinear.create(
            self.hidden_size,
            kv_hidden_size,
            has_bias=self.attention_bias,
            config=config,
            tp_plan="colwise",
        )
        self.o_proj = GeneralLinear.create(
            q_hidden_size,
            self.hidden_size,
            has_bias=False,
            config=config,
            tp_plan="rowwise",
        )

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = Glm4MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Glm4MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_start_row_indices: Optional[paddle.Tensor] = None,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embedding: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.sequence_parallel:
            target_query_shape = [batch_size, -1, self.num_heads, self.head_dim]
            target_key_value_shape = [batch_size, -1, self.num_key_value_heads, self.head_dim]
        else:
            target_query_shape = [0, 0, self.num_heads, self.head_dim]
            target_key_value_shape = [0, 0, self.num_key_value_heads, self.head_dim]
        query_states = query_states.reshape(target_query_shape)
        key_states = key_states.reshape(target_key_value_shape)
        value_states = value_states.reshape(target_key_value_shape)

        if self.use_qk_norm:  # main diff from Llama
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        cos, sin = position_embedding
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states = paddle.concat([past_key_value[0], key_states], axis=1)
            value_states = paddle.concat([past_key_value[1], value_states], axis=1)

        attention_interface = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            attn_mask_start_row_indices=attn_mask_start_row_indices,
            dropout=self.config.get("attention_dropout", 0.0) if self.training else 0.0,
            scaling=self.scaling,
        )

        # if sequence_parallel is true, out shape are [q_len / n, bs, num_head * head_dim]
        # else their shape are [bs, q_len, num_head * head_dim], n is mp parallelism.
        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])

        # print(self.o_proj)
        # print("attn_output: ", attn_output)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class Glm4MoeTopkRouter(nn.Layer):
    def __init__(self, config: Glm4MoeConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = paddle.create_parameter(
            shape=[self.n_routed_experts, config.hidden_size],
            dtype="bfloat16",
            default_initializer=paddle.nn.initializer.Uniform(),
        )

        self.register_buffer("e_score_correction_bias", paddle.zeros((self.n_routed_experts,), dtype=paddle.float32))

    @paddle.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.reshape([-1, self.n_routed_experts]) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.reshape([-1, self.n_group, self.n_routed_experts // self.n_group])
            .topk(2, axis=-1)[0]
            .sum(axis=-1)
        )
        group_idx = paddle.topk(group_scores, k=self.topk_group, axis=-1, sorted=False)[1]
        group_mask = paddle.zeros_like(group_scores)
        group_mask = paddle.put_along_axis(group_mask, group_idx, 1, axis=1, broadcast=False)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand([-1, self.n_group, self.n_routed_experts // self.n_group])
            .reshape([-1, self.n_routed_experts])
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.cast("bool"), 0.0)
        topk_indices = paddle.topk(scores_for_choice, k=self.top_k, axis=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape([-1, self.config.hidden_size])
        router_logits = F.linear(hidden_states.cast("float32"), self.weight.cast("float32").t())
        scores = router_logits.sigmoid()
        topk_indices = self.get_topk_indices(scores)
        topk_weights = paddle.take_along_axis(scores, topk_indices, axis=1, broadcast=False)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class Glm4MoeRMSNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Glm4MoeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = self.create_parameter(
            shape=[hidden_size], default_initializer=paddle.nn.initializer.Constant(1.0)
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(paddle.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Glm4MoeMoE(nn.Layer):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.LayerList(
            [
                Glm4MoeMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = Glm4MoeTopkRouter(config)
        self.shared_experts = Glm4MoeMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )

    def moe(self, hidden_states: paddle.Tensor, topk_indices: paddle.Tensor, topk_weights: paddle.Tensor):
        r"""
        CALL FOR CONTRIBUTION! I don't have time to optimise this right now, but expert weights need to be fused
        to not have to do a loop here (deepseek has 256 experts soooo yeah).
        """
        final_hidden_states = paddle.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = paddle.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = paddle.transpose(expert_mask, perm=[2, 0, 1])

        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = paddle.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(index=token_indices, axis=0, value=weighted_output)

        # in original deepseek, the output of the experts are gathered once we leave this module
        # thus the moe module is itelsf an IsolatedParallel module
        # and all expert are "local" meaning we shard but we don't gather
        return final_hidden_states.cast(hidden_states.dtype)

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.reshape((-1, hidden_states.shape[-1]))
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights)
        hidden_states = paddle.reshape(hidden_states, orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class Glm4MoeDecoderLayer(nn.Layer):
    def __init__(self, config: Glm4MoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Glm4MoeAttention(config=config, layer_idx=layer_idx)

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = Glm4MoeMoE(config)
        else:
            self.mlp = Glm4MoeMLP(config)

        self.input_layernorm = Glm4MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Glm4MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.sequence_parallel:
            self.post_attention_layernorm.enable_sequence_parallel()
            if not hasattr(config, "disable_ffn_model_parallel"):
                self.input_layernorm.enable_sequence_parallel()

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        # output_router_logits: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        position_embedding: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attn_mask_start_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> paddle.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            attn_mask_start_row_indices=attn_mask_start_row_indices,
            position_ids=position_ids,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embedding=position_embedding,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        # else:
        #     router_logits = None
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        # if output_router_logits:
        #     outputs += (router_logits,)
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]
        return outputs


class Glm4MoePreTrainedModel(PretrainedModel):
    config: Glm4MoeConfig
    config_class = Glm4MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    transpose_weight_keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]

    # def _init_weights(self, module):
    #     super()._init_weights(module)
    #     if isinstance(module, Glm4MoeTopkRouter):
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    @classmethod
    def _get_name_mappings(cls, config: Glm4MoeConfig) -> list[StateDictNameMapping]:
        mappings: list[StateDictNameMapping] = []
        model_mappings = [
            ["embed_tokens.weight"],
            ["norm.weight"],
        ]
        for layer_index in range(config.num_hidden_layers):
            layer_mappings = [
                [f"layers.{layer_index}.self_attn.q_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.k_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.v_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.o_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.q_proj.bias", None],
                [f"layers.{layer_index}.self_attn.k_proj.bias", None],
                [f"layers.{layer_index}.self_attn.v_proj.bias", None],
                [f"layers.{layer_index}.self_attn.rotary_emb.inv_freq"],
                [f"layers.{layer_index}.input_layernorm.weight"],
                [f"layers.{layer_index}.post_attention_layernorm.weight"],
                [f"layers.{layer_index}.self_attn.q_norm.weight"],
                [f"layers.{layer_index}.self_attn.k_norm.weight"],
            ]
            model_mappings.extend(layer_mappings)

            for expert_idx in range(config.num_experts):
                expert_mappings = [
                    [f"layers.{layer_index}.mlp.experts.{expert_idx}.gate_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.mlp.experts.{expert_idx}.down_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.mlp.experts.{expert_idx}.up_proj.weight", None, "transpose"],
                ]
                model_mappings.extend(expert_mappings)
            model_mappings.append([f"layers.{layer_index}.mlp.gate.weight", None, "transpose"])

        init_name_mappings(mappings=model_mappings)
        # base-model prefix "GPTOssModel"
        if "Glm4MoeModel" not in config.architectures:
            for mapping in model_mappings:
                mapping[0] = "model." + mapping[0]
                mapping[1] = "model." + mapping[1]
            model_mappings.append(["lm_head.weight", "lm_head.weight", "transpose"])

        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(model_mappings)]
        return mappings

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: Glm4MoeConfig, is_split=True):
        from ..conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(num_layers, num_experts):
            final_actions = {}

            base_actions = {
                "lm_head.weight": partial(fn, is_column=True),
                # Row Linear
                "embed_tokens.weight": partial(fn, is_column=False),
                "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
            }

            if not config.vocab_size % config.tensor_parallel_degree == 0:
                base_actions.pop("lm_head.weight")
                base_actions.pop("embed_tokens.weight")
            # Column Linear
            base_actions["layers.0.self_attn.q_proj.weight"] = partial(fn, is_column=True)

            # if we have enough num_key_value_heads to split, then split it.
            if config.num_key_value_heads % config.tensor_parallel_degree == 0:
                base_actions["layers.0.self_attn.k_proj.weight"] = partial(fn, is_column=True)
                base_actions["layers.0.self_attn.v_proj.weight"] = partial(fn, is_column=True)

            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers, config.num_experts)

        return mappings

    @classmethod
    def _get_fuse_or_split_param_mappings(cls, config: Glm4MoeConfig, is_fuse=False):
        # return parameter fuse utils
        from ..conversion_utils import split_or_fuse_func

        fn = split_or_fuse_func(is_fuse=is_fuse)

        # last key is fused key, other keys are to be fused.
        fuse_qkv_keys = [
            (
                "layers.0.self_attn.q_proj.weight",
                "layers.0.self_attn.k_proj.weight",
                "layers.0.self_attn.v_proj.weight",
                "layers.0.self_attn.qkv_proj.weight",
            ),
        ]

        fuse_gate_up_keys = (
            "layers.0.mlp.gate_proj.weight",
            "layers.0.mlp.up_proj.weight",
            "layers.0.mlp.gate_up_fused_proj.weight",
        )
        num_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)
        fuse_attention_qkv = getattr(config, "fuse_attention_qkv", False)
        fuse_attention_ffn = getattr(config, "fuse_attention_ffn", False)

        final_actions = {}
        if is_fuse:
            if fuse_attention_qkv:
                for i in range(config.num_hidden_layers):
                    for fuse_keys in fuse_qkv_keys:
                        keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in fuse_keys])
                        final_actions[keys] = partial(
                            fn, is_qkv=True, num_heads=num_heads, num_key_value_heads=num_key_value_heads
                        )
            if fuse_attention_ffn:
                for i in range(config.num_hidden_layers):
                    keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in fuse_gate_up_keys])
                    final_actions[keys] = fn
        else:
            if not fuse_attention_qkv:
                for i in range(config.num_hidden_layers):
                    for fuse_keys in fuse_qkv_keys:
                        keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in fuse_keys])
                        final_actions[keys] = partial(
                            fn, split_nums=3, is_qkv=True, num_heads=num_heads, num_key_value_heads=num_key_value_heads
                        )
            if not fuse_attention_ffn:
                for i in range(config.num_hidden_layers):
                    keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in fuse_gate_up_keys])
                    final_actions[keys] = partial(fn, split_nums=2)
        return final_actions


class Glm4MoeRotaryEmbedding(nn.Layer):
    def __init__(self, config: Glm4MoeConfig, device=None):
        super().__init__()
        self.config = config
        # self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype(dtype=paddle.float32) / dim))
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.original_inv_freq = self.inv_freq

    @paddle.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.place)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = paddle.matmul(inv_freq_expanded, position_ids_expanded).transpose([0, 2, 1])
        emb = paddle.concat((freqs, freqs), axis=-1)
        cos = paddle.cos(emb) * self.attention_scaling
        sin = paddle.sin(emb) * self.attention_scaling

        return cos.cast(dtype=x.dtype), sin.cast(dtype=x.dtype)


@register_base_model
class Glm4MoeModel(Glm4MoePreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.92.*", r"model\.layers\.46.*"]

    def __init__(self, config: Glm4MoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.sequence_parallel = config.sequence_parallel
        self.recompute_granularity = config.recompute_granularity
        self.no_recompute_layers = config.no_recompute_layers if config.no_recompute_layers is not None else []

        self.embed_tokens = GeneralEmbedding.create(
            config=config, num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = nn.LayerList(
            [Glm4MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Glm4MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Glm4MoeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        if config.sequence_parallel:
            self.norm.enable_sequence_parallel()

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: Tensor,
        position_ids: Optional[Tensor],
        attention_mask: Tensor,
        output_attentions: bool,
        # output_router_logits: bool,
        past_key_value: Tensor,
        use_cache: bool,
        position_embedding: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attn_mask_start_row_indices=None,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            position_ids,
            attention_mask,
            output_attentions,
            # output_router_logits,
            past_key_value,
            use_cache,
            position_embedding,
            attn_mask_start_row_indices,
            # use_reentrant=self.config.recompute_use_reentrant,
        )

        return hidden_states

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_start_row_indices=None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # output_router_logits = (
        #     output_router_logits if output_router_logits is not None else self.config.output_router_logits
        # )

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        cache_length = 0
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
        else:
            cache_length = past_key_values[0][0].shape[1]
            seq_length_with_past += cache_length

        if inputs_embeds is None:
            # [bs, seq_len, dim]
            inputs_embeds = self.embed_tokens(input_ids)

        if self.sequence_parallel:
            # [bs, seq_len, num_head * head_dim] -> [bs * seq_len, num_head * head_dim]
            bs, seq_len, hidden_size = inputs_embeds.shape
            inputs_embeds = paddle.reshape_(inputs_embeds, [bs * seq_len, hidden_size])
            # [seq_len * bs / n, num_head * head_dim] (n is mp parallelism)
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        if attn_mask_start_row_indices is not None or get_use_casual_mask():
            attention_mask = None
        else:
            # [bs, seq_len]
            attention_mask = (
                paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)
                if attention_mask is None
                else attention_mask
            )

            causal_mask = self._prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                input_shape=(batch_size, seq_length),
                past_key_values_length=cache_length,
                dtype=inputs_embeds.dtype,
            )  # [bs, 1, seq_len, seq_len]
            if self.config.use_flash_attention:
                causal_mask = None if is_casual_mask(causal_mask) else causal_mask

        # if cache_position is None:
        #     past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        #     cache_position: torch.Tensor = torch.arange(
        #         past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        #     )

        # if position_ids is None:
        #     position_ids = cache_position.unsqueeze(0)

        if position_ids is None:
            position_ids = paddle.arange(seq_length, dtype="int64").expand((batch_size, seq_length))

        hidden_states = inputs_embeds
        position_embedding = self.rotary_emb(hidden_states, position_ids)

        # causal_mask = create_causal_mask(
        #     config=self.config,
        #     input_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     cache_position=cache_position,
        #     past_key_values=past_key_values,
        #     position_ids=position_ids,
        # )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        # all_router_logits = () if output_router_logits else None
        next_decoder_cache = () if use_cache else None

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            has_gradient = not hidden_states.stop_gradient
            if self.config.recompute and self.config.recompute_granularity == "full" and has_gradient:
                layer_outputs = self.recompute_training_full(
                    layer_module=decoder_layer,
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    attn_mask_start_row_indices=attn_mask_start_row_indices,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    # output_router_logits=output_router_logits,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    position_embedding=position_embedding,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    attn_mask_start_row_indices=attn_mask_start_row_indices,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    # output_router_logits=output_router_logits,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    position_embedding=position_embedding,
                )

            # # NOTE: clear outdate cache after it has been used for memory saving
            # past_key_value = past_key_values[idx] = None
            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            # if output_router_logits:
            #     all_router_logits += (layer_outputs[-1],)
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Glm4MoeForCausalLM(Glm4MoePreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = Glm4MoeModel(self.config)
        self.vocab_size = config.vocab_size
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def prepare_inputs_for_generation(
        self,
        input_ids,
        use_cache=False,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        # output_router_logits=False,
        **kwargs,
    ):
        batch_size, seq_length = input_ids.shape
        position_ids = kwargs.get("position_ids", paddle.arange(seq_length).expand((batch_size, seq_length)))
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(axis=-1)
            position_ids = position_ids[:, -1].unsqueeze(-1)
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                # "output_router_logits": output_router_logits,
            }
        )
        return model_inputs

    def _get_model_inputs_spec(self, dtype: str):
        return {
            "input_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            "attention_mask": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            "position_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
        }

    @staticmethod
    def update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False):
        # update cache
        if isinstance(outputs, tuple) and len(outputs) > 1 and not isinstance(outputs[1], paddle.Tensor):
            model_kwargs["past_key_values"] = outputs[1]
        if isinstance(outputs, CausalLMOutputWithPast) and "past_key_values" in outputs:
            model_kwargs["past_key_values"] = outputs.past_key_values
        # update position_ids
        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.concat([position_ids, position_ids[..., -1:] + 1], axis=-1)
        if not is_encoder_decoder and "attention_mask" in model_kwargs:
            # TODO: support attention mask for other models
            attention_mask = model_kwargs["attention_mask"]
            if len(attention_mask.shape) == 2:
                model_kwargs["attention_mask"] = paddle.concat(
                    [attention_mask, paddle.ones([attention_mask.shape[0], 1], dtype=attention_mask.dtype)],
                    axis=-1,
                )
            elif len(attention_mask.shape) == 4:
                model_kwargs["attention_mask"] = paddle.concat(
                    [attention_mask, paddle.ones([*attention_mask.shape[:3], 1], dtype=attention_mask.dtype)],
                    axis=-1,
                )[:, :, -1:, :]
        return model_kwargs

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
        logits_to_keep: Union[int, paddle.Tensor] = 0,
    ):

        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # if attn_mask_startend_row_indices is not None and attention_mask is not None:
        #     logger.warning(
        #         "You have provided both attn_mask_startend_row_indices and attention_mask. "
        #         "The attn_mask_startend_row_indices will be used."
        #     )
        #     attention_mask = None

        outputs = self.model(
            input_ids=input_ids,  # [bs, seq_len]
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # [bs, seq_len, dim]
        # if labels is None，means we need full output, instead of tensor_parallel_output
        # tensor_parallel_output is together with ParallelCrossEntropy
        tensor_parallel_output = self.config.tensor_parallel_output and self.config.tensor_parallel_degree > 1
        if labels is not None and self.config.use_fused_linear_cross_entropy:
            from paddlenlp_kernel.triton.cut_cross_entropy import linear_cross_entropy

            assert (
                self.config.tensor_parallel_degree <= 1
            ), "The argument `use_fused_linear_cross_entropy` is imcompatiable with tensor parallel "
            # todo :hidden_states[:, slice_indices, :]
            masked_lm_loss = linear_cross_entropy(hidden_states, self.lm_head.weight, targets=labels)
            binary_sequence = paddle.where(
                masked_lm_loss > 0, paddle.ones_like(masked_lm_loss), paddle.zeros_like(masked_lm_loss)
            )
            count = paddle.sum(binary_sequence)
            if count == 0:
                loss = paddle.sum(masked_lm_loss * binary_sequence)
            else:
                loss = paddle.sum(masked_lm_loss * binary_sequence) / count
            logits = None
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :], tensor_parallel_output=tensor_parallel_output)
            loss = None
            if labels is not None:
                loss, _ = self.criterion(logits, labels)
        # aux_loss = None
        # if output_router_logits:
        #     aux_loss = load_balancing_loss_func(
        #         outputs.router_logits if return_dict else outputs[-1],
        #         self.num_experts,
        #         self.num_experts_per_tok,
        #         attention_mask,
        #     )
        #     if labels is not None:
        #         loss += self.router_aux_loss_coef * aux_loss
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # loss = None
        # if labels is not None:
        #     loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["Glm4MoePreTrainedModel", "Glm4MoeModel", "Glm4MoeForCausalLM"]
