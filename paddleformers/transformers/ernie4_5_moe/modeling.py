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

"""Paddle Ernie4_5_Moe model"""

import math
from copy import deepcopy
from typing import Optional, Tuple

import paddle
import paddle.distributed as dist
import paddle.distributed.communication.group
import paddle.nn.functional as F
from paddle import nn
from paddle.autograd import PyLayer
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    GatherOp,
    ScatterOp,
    mark_as_sequence_parallel_parameter,
)
from paddle.nn.functional import swiglu as fused_swiglu

from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP as Ernie4_5MLP
from ...nn.moe.moe_allgather_layer import MOEAllGatherLayerV2
from ...nn.moe.moe_block import MoEStatics
from ...nn.moe.topk_gate import TopKGate
from ...nn.moe.utils import _parse_moe_group
from ...nn.norm import Norm as GeneralNorm
from ...nn.pp_model import GeneralModelForCausalLMPipe
from ...utils.log import logger
from ..cache_utils import Cache, DynamicCache
from ..ernie4_5.modeling import Ernie4_5Attention
from ..masking_utils import create_causal_mask_and_row_indices
from ..model_outputs import MoECausalLMOutputWithPast, MoECausalLMOutputWithPastAndMTP
from ..model_utils import PretrainedModel, register_base_model
from ..modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ..tensor_parallel_utils import model_parallel_dropout
from .configuration import Ernie4_5_MoeConfig

# Note: ProcessGroupNCCL do not support deepcopy protocol, we made modifications here.
paddle.distributed.communication.group.Group.__deepcopy__ = lambda self, _: self
paddle.distributed.communication.group.Group.to_json = lambda self: repr(self)


def mtp_hidden_states_set_zero(hidden_states, inbatch_pack_offset):
    # inbatch_pack_offset: [batch_size, seqlen]
    # hidden_states: [batch_size, seqlen, d_model]
    if len(hidden_states.shape) == 3:
        batch_size, seqlen, d_model = hidden_states.shape
        valid_indices = paddle.where(inbatch_pack_offset[0] > 0)[0]
        mask = paddle.ones_like(hidden_states[0])
        assert batch_size == 1, "only support batch_size=1 in inbatch sft training"

        if len(valid_indices) > 0:
            zeros = paddle.zeros([len(valid_indices), d_model], dtype=hidden_states.dtype)
            mask = paddle.scatter(mask, valid_indices.reshape([-1, 1]), zeros, overwrite=True)
        mask.stop_gradient = True
        hidden_states = hidden_states * mask.unsqueeze(0)

    elif len(hidden_states.shape) == 2:
        seqlen, d_model = hidden_states.shape
        valid_indices = paddle.where(inbatch_pack_offset > 0)[0]
        mask = paddle.ones_like(hidden_states)

        if len(valid_indices) > 0:
            zeros = paddle.zeros([len(valid_indices), d_model], dtype=hidden_states.dtype)
            mask = paddle.scatter(mask, valid_indices, zeros, overwrite=True)

        mask.stop_gradient = True
        hidden_states = hidden_states * mask
    return hidden_states


class Ernie4_5_MoeRotaryEmbedding(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.head_dim = config.head_dim
        self.base = config.rope_theta
        rope_parameters = config.rope_parameters
        self.rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
        rope_init_fn = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config)

        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[Ernie4_5_MoeConfig] = None,
        seq_len: Optional[int] = None,
    ) -> tuple["paddle.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`PreTrainedConfig`]):
                The model configuration.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`paddle.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype(dtype=paddle.float32) / dim))
        return inv_freq, attention_factor

    @dynamic_rope_update
    def forward(self, x, position_ids):
        """
        Compute rotary position embeddings for given sequence length.

        Args:
            seq_length (int): Maximum sequence length
            position_ids (Tensor): Position ids of shape [batch_size, seq_length]

        Returns:
            Tensor: Rotary position embeddings of shape [1, 1, seq_length, head_dim]
        """
        with paddle.amp.auto_cast(enable=False):
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand([position_ids.shape[0], -1, 1])

            position_ids_expanded = position_ids[:, None, :].float()

            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

            emb = paddle.concat((freqs, freqs), axis=-1)

            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Ernie4_5_MoeMLP(Ernie4_5MLP):
    """Mixture of Experts (MoE) variant of ERNIE's MLP layer."""

    def __init__(self, config, hidden_size, moe_intermediate_size, layer_idx=0, **kwargs):
        """Initialize the MoE MLP layer.

        Args:
            config (Ernie4_5_MoeConfig): Configuration for MoE architecture.
            layer_idx (int): Index of current layer in transformer stack
        """

        if getattr(config, "disable_ffn_model_parallel", False):
            config = deepcopy(config)
            config.tensor_model_parallel_size = 1

        super().__init__(
            config, hidden_size=hidden_size, intermediate_size=moe_intermediate_size, layer_idx=layer_idx, **kwargs
        )
        self.moe_dropout_prob = config.moe_dropout_prob
        self.fuse_swiglu = config.fuse_swiglu
        if self.fuse_swiglu:
            assert fused_swiglu is not None, "fused_swiglu operator is not found."

    def forward(self, x):
        """Forward pass through MoE MLP layer.

        Args:
            x (paddle.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]
                              or [seq_len, hidden_size]

        Returns:
            paddle.Tensor: Output tensor with same shape as input
        """
        if self.fuse_up_gate:
            if self.fuse_swiglu:
                x = self.up_gate_proj(x)
                x = fused_swiglu(x)
            else:
                gate, x = self.up_gate_proj(x).chunk(2, axis=-1)
                x = self.act_fn(gate) * x
        else:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            x = self.act_fn(gate) * up

        if self.moe_dropout_prob > 0:
            with get_rng_state_tracker().rng_state("local_seed"):
                x = F.dropout(x=x, p=self.moe_dropout_prob)
        ret = self.down_proj(x)
        return ret


class FakeMoERouterLoss(PyLayer):
    """A gradient trick layer for MoE router loss computation.

    This layer artificially injects router loss gradients during backpropagation
    while passing through the original tensor during forward pass.
    """

    @staticmethod
    def forward(ctx, x, router_loss, num_acc_steps, enable_delay_scale_loss):
        """Forward pass that preserves input tensor while storing router loss context.

        Args:
            x (paddle.Tensor): The input hidden states tensor
            router_loss (paddle.Tensor): Computed router loss value
            num_acc_steps (int): Gradient accumulation steps
            enable_delay_scale_loss (bool): Whether to scale loss by accumulation steps

        Returns:
            paddle.Tensor: The unchanged input tensor x
        """
        ctx.num_acc_steps = num_acc_steps
        ctx.loss_shape = router_loss.shape
        ctx.loss_dtype = router_loss.dtype
        ctx.enable_delay_scale_loss = enable_delay_scale_loss
        return x

    @staticmethod
    def backward(ctx, out_grad):
        """Backward pass that injects router loss gradients.

        Args:
            out_grad (paddle.Tensor): Gradient from downstream layers

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]:
                - The original downstream gradient
                - Artificial router loss gradient
        """
        if ctx.enable_delay_scale_loss:
            router_loss_grad_value = 1.0
        else:
            router_loss_grad_value = 1.0 / ctx.num_acc_steps

        return out_grad, paddle.full(ctx.loss_shape, router_loss_grad_value, dtype=ctx.loss_dtype)


class Ernie4_5_MoeSparseMoeBlock(MOEAllGatherLayerV2):
    def __init__(self, config, layer_idx):
        # correction bias (yes it seems to be a typo with statics <> statistics)
        self.config = config
        moe_num_experts = config.moe_num_experts
        config.moe_world_size = dist.get_world_size(config.moe_group)
        self.use_multimodel_experts = False
        assert (
            moe_num_experts >= config.moe_world_size
        ), f"expert moe_num_experts={moe_num_experts} >= moe_world_size={config.moe_world_size}"
        assert (
            moe_num_experts % config.moe_world_size == 0
        ), f"expert moe_num_experts={moe_num_experts} % moe_world_size={config.moe_world_size} == 0"

        moe_num_experts_per_device = moe_num_experts // config.moe_world_size
        logger.debug(
            f"using moe-world-size: {config.moe_world_size} expert-per-device:{moe_num_experts_per_device}, moe_group={config.moe_group}"
        )

        moe_statics = MoEStatics(config, layer_idx) if config.moe_use_aux_free else None
        experts = nn.LayerList([])
        moe_rank = paddle.distributed.get_rank(config.moe_group)

        if moe_rank < 0:
            moe_rank = 0
        for i in range(moe_num_experts):
            if i // moe_num_experts_per_device == moe_rank:
                config.disable_ffn_model_parallel = True  # no-split expert
                experts.append(
                    Ernie4_5_MoeMLP(
                        deepcopy(config),
                        config.hidden_size,
                        config.moe_intermediate_size,
                        layer_idx,
                        fuse_up_gate=config.fuse_attention_ffn,
                    )
                )
            else:
                experts.append(None)
        assert (
            len(experts) == moe_num_experts  # including None
        ), f"experts.len={len(experts)} != moe_num_experts={moe_num_experts}"

        gate = TopKGate(config, layer_idx, group=config.moe_group)
        # (optional) shared experts for all forwards
        shared_experts = None
        if config.moe_num_shared_experts > 0:
            config.disable_ffn_model_parallel = False  # split shared epxert
            shared_experts = Ernie4_5_MoeMLP(
                deepcopy(config),
                config.hidden_size,
                config.moe_intermediate_size * config.moe_num_shared_experts,
                fuse_up_gate=config.fuse_attention_ffn,
            )
        use_expert_out_alltoall = use_expert_out_alltoall = "alltoall" in config.moe_multimodal_dispatch_use_allgather
        use_padding = "unpad" not in config.moe_multimodal_dispatch_use_allgather
        super().__init__(
            gate=gate,
            experts=experts,
            layer_idx=layer_idx,
            shared_experts=shared_experts,
            group=config.moe_group,
            recompute=bool(
                self.config.recompute_granularity == "selective"
                and self.config.recompute_modules is not None
                and "moe" in self.config.recompute_modules
            ),
            k=config.moe_k,
            all_to_all_dropout=config.moe_all_to_all_dropout,
            group_experts=config.moe_group_experts,
            moe_statics=moe_statics,
            moe_num_experts=config.moe_num_experts,
            use_expert_out_alltoall=use_expert_out_alltoall,
            use_padding=use_padding,
            dense_token_type=3,
        )
        self.norm_min = config.moe_norm_min
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_k


class Ernie4_5_MoeDecoderLayer(nn.Layer):
    """A single transformer decoder layer in ERNIE-MoE model.

    Contains self-attention and feed-forward components with optional MoE (Mixture of Experts)
    support, residual connections, and layer normalization.
    """

    def __init__(self, config, layer_idx):
        """Initialize the decoder layer.

        Args:
            config (Ernie4_5_MoeConfig): Model configuration.
            layer_idx (int): Index of this layer in the transformer stack
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.config = config
        self.self_attn = Ernie4_5Attention(config, layer_idx)
        if (
            ((layer_idx + 1) % config.moe_layer_interval == 0)
            and layer_idx >= config.moe_layer_start_index
            and layer_idx <= config.moe_layer_end_index
        ):
            self.mlp = Ernie4_5_MoeSparseMoeBlock(config, layer_idx)
        else:
            self.mlp = Ernie4_5MLP(
                config,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                fuse_up_gate=config.fuse_attention_ffn,
            )

        if config.sequence_parallel and isinstance(
            self.mlp, Ernie4_5_MoeSparseMoeBlock
        ):  # Under `mp-moe`, gate is effective in attn and is in the synchronization zone.
            for p in self.mlp.gate.parameters():
                mark_as_sequence_parallel_parameter(p)

        self.input_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=config.use_bias,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.post_attention_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=config.use_bias,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )

        self.hidden_dropout = nn.Dropout(p=config.hidden_dropout_prob, mode="upscale_in_train")

        if config.sequence_parallel:
            # There is no Column/RowLinear in bias and expert in mp-moe. No hook is needed.
            if not hasattr(config, "disable_ffn_model_parallel"):
                self.input_layernorm.enable_sequence_parallel()
                if config.use_bias:  # false
                    self.input_layernorm.enable_sequence_parallel()
                    if isinstance(self.mlp, Ernie4_5_MoeSparseMoeBlock):
                        for m in self.mlp.experts:
                            mark_as_sequence_parallel_parameter(m.down_proj.bias)
                    else:
                        mark_as_sequence_parallel_parameter(self.mlp.down_proj.bias)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_gate_logits=False,  # PP model should not output gate logits,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """Forward pass through the decoder layer.

        Args:
            hidden_states (paddle.Tensor): Input tensor [batch_size, seq_len, hidden_size]
            attention_mask (Optional[paddle.Tensor]): Attention mask tensor
            attn_mask_startend_row_indices (Optional[paddle.Tensor]): Indices for variable length attention
            position_ids (Optional[paddle.Tensor]): Position indices for rotary embeddings
            output_attentions (Optional[bool]): Whether to return attention weights
            past_key_values (Optional[Cache]): Cached key/value states
            use_cache (Optional[bool]): Whether to cache key/value states
            output_gate_logits (bool): Whether to return MoE gate logits

        Returns:
            Union: Various output combinations depending on arguments:
                - Base case: Hidden states tensor
                - With attention: Tuple of (hidden_states, attention_weights)
                - With cache: Tuple of (hidden_states, cached_key_value)
                - With MoE: May include gate logits in output tuple
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        (hidden_states, self_attn_weights, *router_loss_attn) = self.self_attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        with model_parallel_dropout(self.config):
            hidden_states = self.hidden_dropout(hidden_states) + residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if isinstance(self.mlp, Ernie4_5_MoeSparseMoeBlock):
            hidden_states, _, router_loss, gate_logits = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
            gate_logits, router_loss = None, None

        with model_parallel_dropout(self.config):
            hidden_states = self.hidden_dropout(hidden_states) + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        # Non-empty only if `use_moe`
        if router_loss_attn:
            router_loss_attn = router_loss_attn[0]
            router_loss = router_loss + router_loss_attn

        # When use_moe is enabled, an additional return value will be added regardless of whether this layer has a moe layer or not
        if router_loss is not None:
            hidden_states = FakeMoERouterLoss.apply(
                hidden_states,
                router_loss,
                self.config.num_acc_steps,
                self.config.enable_delay_scale_loss,
            )
        if self.training:
            hidden_states.stop_gradient = False

        outputs = (hidden_states,) + outputs[1:]

        if output_gate_logits:
            outputs += (gate_logits,)

        # remove empty tuple for pipeline parallel
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]
        return outputs


class Ernie4_5_MoePretrainedModel(PretrainedModel):
    """Base class for ERNIE-Moe pretrained models."""

    config_class = Ernie4_5_MoeConfig
    base_model_prefix = "model"
    _keep_in_fp32_modules = ["mlp.gate.weight", "e_score_correction_bias"]
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "gate",
        "mtp_linear_proj\.\d+",
    ]

    @classmethod
    def _gen_aoa_config(cls, config: Ernie4_5_MoeConfig):
        model_prefix = "" if cls == cls.base_model_class else "model."
        aoa_config = {
            "aoa_statements": [
                f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
                f"model.norm.weight -> {model_prefix}norm.weight",
                f"model.layers.$LAYER_ID.input_layernorm.weight -> {model_prefix}layers.$LAYER_ID.input_layernorm.weight",
                f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
                f"model.layers.$LAYER_ID.mlp.moe_statics.e_score_correction_bias -> {model_prefix}layers.$LAYER_ID.mlp.moe_statics.e_score_correction_bias, dtype='float32'",
                f"model.layers.$LAYER_ID.mlp.gate.weight -> {model_prefix}layers.$LAYER_ID.mlp.gate.weight, dtype='float32'",
                f"model.layers.$LAYER_ID.mlp.down_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.down_proj.weight",
                f"model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
                f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight",
                f"model.layers.$LAYER_ID.mlp.shared_experts.down_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.shared_experts.down_proj.weight",
                f"model.mtp_block.$LAYER_ID.input_layernorm.weight -> {model_prefix}mtp_block.$LAYER_ID.input_layernorm.weight",
                f"model.mtp_block.$LAYER_ID.post_attention_layernorm.weight -> {model_prefix}mtp_block.$LAYER_ID.post_attention_layernorm.weight",
                f"model.mtp_block.$LAYER_ID.mlp.down_proj.weight^T -> {model_prefix}mtp_block.$LAYER_ID.mlp.down_proj.weight",
                f"model.mtp_block.$LAYER_ID.self_attn.o_proj.weight^T -> {model_prefix}mtp_block.$LAYER_ID.self_attn.o_proj.weight",
                f"model.mtp_emb_norm.$LAYER_ID.weight -> {model_prefix}mtp_emb_norm.$LAYER_ID.weight",
                f"model.mtp_hidden_norm.$LAYER_ID.weight -> {model_prefix}mtp_hidden_norm.$LAYER_ID.weight",
                f"model.mtp_linear_proj.$LAYER_ID.weight^T -> {model_prefix}mtp_linear_proj.$LAYER_ID.weight",
            ]
        }

        # attention qkv
        if not config.fuse_attention_qkv:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight"
                for x in ("q", "k", "v")
            ]
            aoa_config["aoa_statements"] += [
                f"model.mtp_block.$LAYER_ID.self_attn.{x}_proj.weight^T -> {model_prefix}mtp_block.$LAYER_ID.self_attn.{x}_proj.weight"
                for x in ("q", "k", "v")
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.self_attn.q_proj.weight^T, model.layers.$LAYER_ID.self_attn.k_proj.weight^T, model.layers.$LAYER_ID.self_attn.v_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}",
                f"model.mtp_block.$LAYER_ID.self_attn.q_proj.weight^T, model.mtp_block.$LAYER_ID.self_attn.k_proj.weight^T, model.mtp_block.$LAYER_ID.self_attn.v_proj.weight^T -> {model_prefix}mtp_block.$LAYER_ID.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}",
            ]
            if config.use_bias:
                aoa_config["aoa_statements"] += [
                    f"model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias -> {model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
                    f"model.mtp_block.$LAYER_ID.self_attn.q_proj.bias, model.mtp_block.$LAYER_ID.self_attn.k_proj.bias, model.mtp_block.$LAYER_ID.self_attn.v_proj.bias -> {model_prefix}mtp_block.$LAYER_ID.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
                ]

        # FFN
        if not config.fuse_attention_ffn:
            aoa_config["aoa_statements"] += (
                [
                    f"model.layers.$LAYER_ID.mlp.{p}_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.{p}_proj.weight"
                    for p in ("gate", "up")
                ]
                + [
                    f"model.layers.$LAYER_ID.mlp.shared_experts.{p}_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.shared_experts.{p}_proj.weight"
                    for p in ("gate", "up")
                ]
                + [
                    f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{p}_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{p}_proj.weight"
                    for p in ("gate", "up")
                ]
                + [
                    f"model.mtp_block.$LAYER_ID.mlp.{p}_proj.weight^T -> {model_prefix}mtp_block.$LAYER_ID.mlp.{p}_proj.weight"
                    for p in ("gate", "up")
                ]
            )
        else:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.mlp.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn",
                f"model.layers.$LAYER_ID.mlp.shared_experts.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.shared_experts.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.shared_experts.up_gate_proj.weight, fused_ffn",
                f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_gate_proj.weight, fused_ffn",
                f"model.mtp_block.$LAYER_ID.mlp.gate_proj.weight^T, model.mtp_block.$LAYER_ID.mlp.up_proj.weight^T -> {model_prefix}mtp_block.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn",
            ]

        if config.tie_word_embeddings:
            aoa_config["aoa_statements"] += ["model.embed_tokens.weight -> lm_head.weight"]

        return aoa_config

    @classmethod
    def _gen_inv_aoa_config(cls, config: Ernie4_5_MoeConfig):
        model_prefix = "" if cls == cls.base_model_class else "model."
        aoa_statements = [
            f"{model_prefix}embed_tokens.weight -> model.embed_tokens.weight",
            f"{model_prefix}norm.weight -> model.norm.weight",
            f"{model_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.layers.$LAYER_ID.input_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.layers.$LAYER_ID.post_attention_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.mlp.moe_statics.e_score_correction_bias -> model.layers.$LAYER_ID.mlp.moe_statics.e_score_correction_bias",
            f"{model_prefix}layers.$LAYER_ID.mlp.gate.weight -> model.layers.$LAYER_ID.mlp.gate.weight",
            f"{model_prefix}layers.$LAYER_ID.mlp.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.down_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.layers.$LAYER_ID.self_attn.o_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.mlp.shared_experts.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.shared_experts.down_proj.weight",
            f"{model_prefix}mtp_block.$LAYER_ID.input_layernorm.weight -> model.mtp_block.$LAYER_ID.input_layernorm.weight",
            f"{model_prefix}mtp_block.$LAYER_ID.post_attention_layernorm.weight -> model.mtp_block.$LAYER_ID.post_attention_layernorm.weight",
            f"{model_prefix}mtp_block.$LAYER_ID.mlp.down_proj.weight^T -> model.mtp_block.$LAYER_ID.mlp.down_proj.weight",
            f"{model_prefix}mtp_block.$LAYER_ID.self_attn.o_proj.weight^T -> model.mtp_block.$LAYER_ID.self_attn.o_proj.weight",
            f"{model_prefix}mtp_emb_norm.$LAYER_ID.weight -> model.mtp_emb_norm.$LAYER_ID.weight",
            f"{model_prefix}mtp_hidden_norm.$LAYER_ID.weight -> model.mtp_hidden_norm.$LAYER_ID.weight",
            f"{model_prefix}mtp_linear_proj.$LAYER_ID.weight^T -> model.mtp_linear_proj.$LAYER_ID.weight",
        ]

        if not config.fuse_attention_qkv:
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> model.layers.$LAYER_ID.self_attn.{x}_proj.weight"
                for x in ("q", "k", "v")
            ]
            aoa_statements += [
                f"{model_prefix}mtp_block.$LAYER_ID.self_attn.{x}_proj.weight^T -> model.mtp_block.$LAYER_ID.self_attn.{x}_proj.weight"
                for x in ("q", "k", "v")
            ]
        else:
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight -> model.layers.$LAYER_ID.self_attn.q_proj.weight, model.layers.$LAYER_ID.self_attn.k_proj.weight, model.layers.$LAYER_ID.self_attn.v_proj.weight, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups = {config.num_key_value_heads}",
                f"{model_prefix}mtp_block.$LAYER_ID.self_attn.qkv_proj.weight -> model.mtp_block.$LAYER_ID.self_attn.q_proj.weight, model.mtp_block.$LAYER_ID.self_attn.k_proj.weight, model.mtp_block.$LAYER_ID.self_attn.v_proj.weight, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups = {config.num_key_value_heads}",
            ]
            for x in ("q", "k", "v"):
                for layer_id in range(config.num_hidden_layers):
                    aoa_statements += [
                        f"model.layers.{layer_id}.self_attn.{x}_proj.weight^T -> model.layers.{layer_id}.self_attn.{x}_proj.weight",
                    ]
                for layer_id in range(config.num_nextn_predict_layers):
                    aoa_statements += [
                        f"model.mtp_block.{layer_id}.self_attn.{x}_proj.weight^T -> model.mtp_block.{layer_id}.self_attn.{x}_proj.weight",
                    ]
            if config.use_bias:
                aoa_statements += [
                    f"{model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias -> model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
                    f"{model_prefix}mtp_block.$LAYER_ID.self_attn.qkv_proj.bias -> model.mtp_block.$LAYER_ID.self_attn.q_proj.bias, model.mtp_block.$LAYER_ID.self_attn.k_proj.bias, model.mtp_block.$LAYER_ID.self_attn.v_proj.bias, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
                ]

        if not config.fuse_attention_ffn:
            aoa_statements += (
                [
                    f"{model_prefix}layers.$LAYER_ID.mlp.{y}_proj.weight^T -> model.layers.$LAYER_ID.mlp.{y}_proj.weight"
                    for y in ("gate", "up")
                ]
                + [
                    f"{model_prefix}layers.$LAYER_ID.mlp.shared_experts.{y}_proj.weight^T -> model.layers.$LAYER_ID.mlp.shared_experts.{y}_proj.weight"
                    for y in ("gate", "up")
                ]
                + [
                    f"{model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{y}_proj.weight^T -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{y}_proj.weight"
                    for y in ("gate", "up")
                ]
                + [
                    f"{model_prefix}mtp_block.$LAYER_ID.mlp.{y}_proj.weight^T -> model.mtp_block.$LAYER_ID.mlp.{y}_proj.weight"
                    for y in ("gate", "up")
                ]
            )
        else:
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.mlp.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.gate_proj.weight, model.layers.$LAYER_ID.mlp.up_proj.weight, fused_ffn",
                f"{model_prefix}layers.$LAYER_ID.mlp.shared_experts.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.shared_experts.gate_proj.weight, model.layers.$LAYER_ID.mlp.shared_experts.up_proj.weight, fused_ffn",
                f"{model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight, model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight, fused_ffn",
                f"{model_prefix}mtp_block.$LAYER_ID.mlp.up_gate_proj.weight -> model.mtp_block.$LAYER_ID.mlp.gate_proj.weight, model.mtp_block.$LAYER_ID.mlp.up_proj.weight, fused_ffn",
            ]
            # mlp
            for layer_id in range(config.moe_layer_start_index):
                for y in ("gate", "up"):
                    aoa_statements += [
                        f"model.layers.{layer_id}.mlp.{y}_proj.weight^T -> model.layers.{layer_id}.mlp.{y}_proj.weight",
                    ]
            # experts
            for layer_id in range(config.moe_layer_start_index, config.num_hidden_layers):
                for y in ("gate", "up"):
                    aoa_statements += [
                        f"model.layers.{layer_id}.mlp.shared_experts.{y}_proj.weight^T -> model.layers.{layer_id}.mlp.shared_experts.{y}_proj.weight"
                    ]
                    for expert_id in range(config.moe_num_experts):
                        aoa_statements += [
                            f"model.layers.{layer_id}.mlp.experts.{expert_id}.{y}_proj.weight^T -> model.layers.{layer_id}.mlp.experts.{expert_id}.{y}_proj.weight"
                        ]
            # mtp
            for layer_id in range(config.num_nextn_predict_layers):
                for y in ("gate", "up"):
                    aoa_statements += [
                        f"model.mtp_block.{layer_id}.mlp.{y}_proj.weight^T -> model.mtp_block.{layer_id}.mlp.{y}_proj.weight"
                    ]

        if config.tie_word_embeddings:
            aoa_statements += ["lm_head.weight -> _"]

        aoa_config = {"aoa_statements": aoa_statements}
        return aoa_config


@register_base_model
class Ernie4_5_MoeModel(Ernie4_5_MoePretrainedModel):
    """The core ERNIE transformer model with MoE (Mixture of Experts) support."""

    def __init__(self, config: Ernie4_5_MoeConfig):
        """Initialize the ERNIE model architecture.

        Args:
            config (Ernie4_5_MoeConfig): Model configuration.
        """
        if config.moe_group in {"mp", "model", "tp"} and config.tensor_model_parallel_size > 1:
            logger.info(f"disable FFN tensor model parallel, moe-group={config.moe_group}")
            config.disable_ffn_model_parallel = True
        config.moe_group_origin = config.moe_group
        if isinstance(config.moe_group, str):
            config.moe_group = _parse_moe_group(config.moe_group)

        config.moe_world_size = dist.get_world_size(config.moe_group)
        if config.moe_world_size < 0:
            config.moe_world_size = 1
        config.moe_rank = dist.get_rank(config.moe_group)
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.config = config

        self.embed_tokens = GeneralEmbedding.create(
            config=config, num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )

        self.layers = nn.LayerList([Ernie4_5_MoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=config.use_bias,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )

        self.rotary_emb = Ernie4_5_MoeRotaryEmbedding(config)

        if self.config.num_nextn_predict_layers > 0:
            self.mtp_block = paddle.nn.LayerList(
                [
                    Ernie4_5_MoeDecoderLayer(config, layer_idx)
                    for layer_idx in range(self.config.num_nextn_predict_layers)
                ]
            )

            self.mtp_hidden_norm = paddle.nn.LayerList(
                [
                    GeneralNorm.create(
                        config=config,
                        norm_type="rms_norm",
                        hidden_size=config.hidden_size,
                        has_bias=config.use_bias,
                        norm_eps=self.config.rms_norm_eps,
                        input_is_parallel=config.sequence_parallel,
                    )
                    for _ in range(self.config.num_nextn_predict_layers)
                ]
            )
            self.mtp_emb_norm = paddle.nn.LayerList(
                [
                    GeneralNorm.create(
                        config=config,
                        norm_type="rms_norm",
                        hidden_size=config.hidden_size,
                        has_bias=config.use_bias,
                        norm_eps=self.config.rms_norm_eps,
                        input_is_parallel=config.sequence_parallel,
                    )
                    for _ in range(self.config.num_nextn_predict_layers)
                ]
            )

            self.mtp_linear_proj = paddle.nn.LayerList(
                [
                    GeneralLinear.create(
                        config.hidden_size * 2,
                        config.hidden_size,
                        has_bias=config.use_bias,
                        config=config,
                        linear_type="default",
                    )
                    for _ in range(config.num_nextn_predict_layers)
                ]
            )
            if config.sequence_parallel:
                logger.info("enable sequence parallel for mtp_linear")
                for mtp_linear in self.mtp_linear_proj:
                    mark_as_sequence_parallel_parameter(mtp_linear.weight)
                    if config.use_bias:
                        mark_as_sequence_parallel_parameter(mtp_linear.bias)

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        layer_module,
        hidden_states,
        attention_mask,
        attn_mask_startend_row_indices,
        position_ids,
        position_embeddings,
        output_attentions,
        past_key_values,
        use_cache,
    ):
        """Perform gradient checkpointing for memory-efficient training.

        Args:
            layer_module (nn.Layer): Transformer layer to recompute
            hidden_states (paddle.Tensor): Input hidden states
            attention_mask (paddle.Tensor): Attention mask
            attn_mask_startend_row_indices (paddle.Tensor): Variable length indices
            position_ids (paddle.Tensor): Position indices
            output_attentions (bool): Whether to output attention weights
            past_key_values (Optional[Cache]): Cached key/value states
            use_cache (bool): Whether to cache key/value states

        Returns:
            paddle.Tensor: Output hidden states after recomputation
        """

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs, output_gate_logits=False)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            attn_mask_startend_row_indices,
            position_ids,
            position_embeddings,
            output_attentions,
            past_key_values,
            use_cache,
        )
        return hidden_states

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        attn_mask_startend_row_indices=None,
        inputs_embeds=None,
        use_cache=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
        **kwargs,
    ):
        """Forward pass through the ERNIE model.

        Args:
            input_ids (Optional[paddle.Tensor]): Input token IDs
            position_ids (Optional[paddle.Tensor]): Position indices
            attention_mask (Optional[paddle.Tensor]): Attention mask
            attn_mask_startend_row_indices (Optional[paddle.Tensor]): Variable length attention indices
            inputs_embeds (Optional[paddle.Tensor]): Precomputed embeddings
            use_cache (Optional[bool]): Whether to cache key/value states
            past_key_values (Optional[Cache]): Cached key/value states
            output_attentions (Optional[bool]): Whether to output attention weights
            output_hidden_states (Optional[bool]): Whether to output all hidden states
            return_dict (Optional[bool]): Whether to return dict or tuple

        Returns:
            Union[Tuple, MoECausalLMOutputWithPast]:
                Various outputs depending on configuration, including:
                - last_hidden_state: Final layer hidden states
                - past_key_values: Cached key/value states if use_cache=True
                - hidden_states: All hidden states if output_hidden_states=True
                - attentions: Attention weights if output_attentions=True
                - router_loss: MoE router loss if use_moe=True
                - gate_logits: MoE gate logits if use_moe=True
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            bsz, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            bsz, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        full_seq_length = seq_length

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        kv_seq_len = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = paddle.arange(kv_seq_len, seq_length).unsqueeze(0).tile((bsz, 1))

        seq_length -= self.config.num_nextn_predict_layers

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "batch_size": bsz,
            "seq_length": full_seq_length,
            "cache_length": kv_seq_len,
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "prepare_decoder_attention_mask": self._prepare_decoder_attention_mask,
        }

        attention_mask, attn_mask_startend_row_indices = create_causal_mask_and_row_indices(**mask_kwargs)

        if self.training and self.config.num_nextn_predict_layers > 0:
            inputs_embeds_extra = inputs_embeds[:, -self.config.num_nextn_predict_layers :, :]
            inputs_embeds = inputs_embeds[:, : -self.config.num_nextn_predict_layers, :]
            inputs_embeds_ori = inputs_embeds

            if position_ids is not None:
                position_ids_extra = position_ids[:, -self.config.num_nextn_predict_layers :]
                position_ids = position_ids[:, : -self.config.num_nextn_predict_layers]
                position_ids_ori = position_ids

            if attention_mask is not None:
                attention_mask_full = attention_mask
                attention_mask = attention_mask[
                    :,
                    :,
                    : -self.config.num_nextn_predict_layers,
                    : -self.config.num_nextn_predict_layers,
                ]

            if attn_mask_startend_row_indices is not None:
                attn_mask_startend_row_indices_extra = attn_mask_startend_row_indices[
                    :, :, -self.config.num_nextn_predict_layers :
                ]
                attn_mask_startend_row_indices = attn_mask_startend_row_indices[
                    :, :, : -self.config.num_nextn_predict_layers
                ]
                attn_mask_startend_row_indices_ori = attn_mask_startend_row_indices

            nbatch_pack_offset = kwargs.get("nbatch_pack_offset", None)
            if nbatch_pack_offset is None:
                raise ValueError("nbatch_pack_offset is required in mtp train")

            nbatch_pack_offset_extra = nbatch_pack_offset[:, -self.config.num_nextn_predict_layers :]
            nbatch_pack_offset = nbatch_pack_offset[:, -self.config.num_nextn_predict_layers :]
            nbatch_pack_offset_ori = nbatch_pack_offset

        if self.config.sequence_parallel:
            inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        hidden_states = inputs_embeds

        if self.config.apply_rope_fusion:
            position_embeddings = None
        else:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)  # cos and sin

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_loss = 0.0
        all_gate_logits = ()
        mtp_outputs = []

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            has_gradient = not hidden_states.stop_gradient
            if (
                self.config.recompute_granularity == "full"
                and self.config.recompute_method == "uniform"
                and self.config.recompute_num_layers == 1
                and has_gradient
            ):
                layer_outputs = self.recompute_training(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    attn_mask_startend_row_indices,
                    position_ids,
                    position_embeddings,
                    output_attentions,
                    past_key_values,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask,
                    attn_mask_startend_row_indices,
                    position_ids,
                    position_embeddings,
                    output_attentions,
                    past_key_values,
                    use_cache,
                )

            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if not (
                self.config.recompute_granularity == "full"
                and self.config.recompute_method == "uniform"
                and self.config.recompute_num_layers == 1
                and has_gradient
            ):
                layer_outputs, gate_logits = layer_outputs[:-1], layer_outputs[-1]
                all_gate_logits = all_gate_logits + (gate_logits,)

        # Multi Token Prediction
        if self.training and self.config.num_nextn_predict_layers > 0:
            mtp_outputs.append(hidden_states)

            for depth in range(self.config.num_nextn_predict_layers):
                if self.config.sequence_parallel:
                    hidden_states = GatherOp.apply(hidden_states)
                    hidden_states = hidden_states.reshape([-1, seq_length, hidden_states.shape[-1]])

                inputs_embeds_cur_depth = paddle.cat(
                    [
                        inputs_embeds_ori[:, (depth + 1) :, :],
                        inputs_embeds_extra[:, : (depth + 1), :],
                    ],
                    axis=1,
                )

                if attention_mask is not None:
                    b, h, seqlen, seqlen = attention_mask.shape
                    attention_mask = attention_mask_full[
                        :,
                        :,
                        (depth + 1) : (seqlen + depth + 1),
                        (depth + 1) : (seqlen + depth + 1),
                    ]

                if attn_mask_startend_row_indices is not None:
                    attn_mask_startend_row_indices = paddle.cat(
                        [
                            attn_mask_startend_row_indices_ori[:, :, (depth + 1) :],
                            attn_mask_startend_row_indices_extra[:, :, : (depth + 1)],
                        ],
                        axis=-1,
                    )
                if position_ids is not None:
                    position_ids = paddle.cat(
                        [
                            position_ids_ori[:, (depth + 1) :],
                            position_ids_extra[:, : (depth + 1)],
                        ],
                        axis=1,
                    )

                nbatch_pack_offset_cur_depth = paddle.cat(
                    [
                        nbatch_pack_offset_ori[:, (depth + 1) :],
                        nbatch_pack_offset_extra[:, : (depth + 1)],
                    ],
                    axis=1,
                )
                hidden_states = mtp_hidden_states_set_zero(hidden_states, nbatch_pack_offset_cur_depth)

                # Norm&Concat
                inputs_embeds_cur_depth_norm = self.mtp_emb_norm[depth](inputs_embeds_cur_depth)
                hidden_states_norm = self.mtp_hidden_norm[depth](hidden_states)

                inputs_embeds_cur_depth = self.mtp_linear_proj[depth](
                    paddle.cat([inputs_embeds_cur_depth_norm, hidden_states_norm], axis=-1)
                )

                if self.config.sequence_parallel:
                    inputs_embeds_cur_depth = inputs_embeds_cur_depth.reshape([-1, inputs_embeds_cur_depth.shape[-1]])
                    inputs_embeds_cur_depth = ScatterOp.apply(inputs_embeds_cur_depth)

                decoder_layer = self.mtp_block[depth]
                past_key_values = None
                layer_outputs = decoder_layer(
                    inputs_embeds_cur_depth,
                    attention_mask,
                    attn_mask_startend_row_indices,
                    position_ids,
                    output_attentions,
                    past_key_values,
                    use_cache,
                )
                if isinstance(layer_outputs, (tuple, list)):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs

                if not (self.config.recompute_granularity is not None and has_gradient):
                    layer_outputs, gate_logits = (
                        layer_outputs[:-1],
                        layer_outputs[-1],
                    )
                    all_gate_logits = all_gate_logits + (gate_logits,)

                mtp_outputs.append(hidden_states)
            mtp_outputs = [self.norm(hidden_states) for hidden_states in mtp_outputs]
            hidden_states, mtp_outputs = mtp_outputs[0], mtp_outputs[1:]
        else:
            hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_self_attns,
                    all_router_loss,
                    all_gate_logits,
                    mtp_outputs,
                ]
                if v is not None
            )

        # assert all_router_loss is None, f'moe not support `return-dict`'
        return MoECausalLMOutputWithPastAndMTP(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_loss=all_router_loss,
            gate_logits=all_gate_logits,
            mtp_outputs=mtp_outputs,
        )


class Ernie4_5_MoeForCausalLM(Ernie4_5_MoePretrainedModel):
    """ERNIE Mixture of Experts (MoE) model for causal language modeling."""

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        """
        Initializes the ERNIE MoE model for causal language modeling.

        Args:
            config (dict): Model configuration.
        """
        super().__init__(config)

        # initialize-trick for big model,
        # see https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/README.md#std-init
        new_initializer_range = math.sqrt(0.3333 / config.hidden_size)
        logger.info(f"change initializer-range from {config.initializer_range} to {new_initializer_range}")
        config.initializer_range = new_initializer_range
        self.config = config
        self.model = Ernie4_5_MoeModel(config)
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        self.tie_weights()  # maybe weight share

    def prepare_attention_mask_for_generation(self, input_ids, pad_token_id, eos_token_id):
        """Avoid using attention_mask with flash_attn on generation."""
        if self.config._attn_implementation == "sdpa":
            return None
        return super().prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id)

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        attn_mask_startend_row_indices=None,
        inputs_embeds=None,
        labels=None,
        loss_mask=None,
        use_cache=False,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,  # true when decode, false when pretrain & eval
        **kwargs,
    ):
        """
        Forward pass for causal language modeling.

        Args:
            input_ids (paddle.Tensor): Input token IDs.
            position_ids (paddle.Tensor): Position IDs.
            attention_mask (paddle.Tensor): Attention mask.
            attn_mask_startend_row_indices (paddle.Tensor): Attention mask start indices.
            inputs_embeds (paddle.Tensor): Optional embedded inputs.
            labels (paddle.Tensor): Target labels.
            loss_mask (paddle.Tensor): Loss mask.
            use_cache (bool): Whether to use cached hidden states.
            past_key_values (Cache): Pre-computed hidden states.
            output_attentions (bool): Whether to output attentions.
            output_hidden_states (bool): Whether to output hidden states.
            return_dict (bool): Whether to return a dictionary.

        Returns:
            Union[tuple, MoECausalLMOutputWithPast]: Model outputs.
        """
        if kwargs.get("attn_mask_start_row_indices", None) is not None and attn_mask_startend_row_indices is None:
            attn_mask_startend_row_indices = kwargs["attn_mask_start_row_indices"]

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            nbatch_pack_offset=kwargs.get("nbatch_pack_offset", None),
        )

        hidden_states = outputs.last_hidden_state
        mtp_outputs = outputs.mtp_outputs

        if self.criterion.loss_type == "dpo":
            logits = self.lm_head(hidden_states)
            chosen_labels = kwargs.get("chosen_labels", None)
            rejected_labels = kwargs.get("rejected_labels", None)
            response_indexs = kwargs.get("response_indexs", None)
            score_deltas = kwargs.get("score_deltas", None)
            reference_chosen_logps = kwargs.get("reference_chosen_logps", None)
            reference_rejected_logps = kwargs.get("reference_rejected_logps", None)
            labels = (
                chosen_labels,
                rejected_labels,
                response_indexs,
                score_deltas,
                reference_chosen_logps,
                reference_rejected_logps,
            )
            return self.criterion(
                logits,
                labels,
            )

        # if labels is None，means we need full output, instead of tensor_parallel_output
        # tensor_parallel_output is togather with ParallelCrossEntropy
        logits = self.lm_head(hidden_states)
        mtp_logits = []
        if len(mtp_outputs) > 0:
            mtp_logits = [self.lm_head(_hidden_states) for _hidden_states in mtp_outputs]

        if return_dict:  # aka Generate Decoding
            if labels is not None:
                loss, _ = self.criterion(logits, labels, loss_mask)
            else:
                loss = None
            return MoECausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                router_loss=outputs.router_loss,
            )
        router_loss = outputs.router_loss

        # Pretrain & Eval must have labels
        assert labels is not None

        return self.criterion(logits, labels, loss_mask, router_loss=router_loss, mtp_logits=mtp_logits)


class Ernie4_5_MoeForCausalLMPipe(GeneralModelForCausalLMPipe):
    config_class = Ernie4_5_MoeConfig
    _decoder_layer_cls = Ernie4_5_MoeDecoderLayer
    _get_tensor_parallel_mappings = Ernie4_5_MoeModel._get_tensor_parallel_mappings
    _init_weights = Ernie4_5_MoeModel._init_weights
    _keep_in_fp32_modules = Ernie4_5_MoeModel._keep_in_fp32_modules
    _tied_weights_keys = ["lm_head.weight"]
    transpose_weight_keys = Ernie4_5_MoeModel.transpose_weight_keys
    _gen_aoa_config = Ernie4_5_MoeForCausalLM._gen_aoa_config
    _gen_inv_aoa_config = Ernie4_5_MoeForCausalLM._gen_inv_aoa_config


__all__ = ["Ernie4_5_MoeModel", "Ernie4_5_MoeForCausalLM", "Ernie4_5_MoeForCausalLMPipe"]
