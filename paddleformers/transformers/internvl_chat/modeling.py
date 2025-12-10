# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 The OpenGVLab Team and The HuggingFace Inc. team. All rights reserved.
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

import warnings
from typing import Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from einops import rearrange
from paddle.distributed.fleet.recompute import recompute

from ...generation.configuration_utils import GenerationConfig
from ...nn.linear import Linear as GeneralLinear
from ...utils import logger
from .. import LlamaForCausalLM, Qwen2ForCausalLM
from ..activations import ACT2FN
from ..model_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from ..model_utils import PretrainedModel
from .configuration import InternVisionConfig, InternVLChatConfig
from .conversation import get_conv_template

try:
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

    has_flash_attn = True
except:
    print("FlashAttention2 is not installed.")
    has_flash_attn = False


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None, max_s=None, need_weights=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
                if unpadded: (nnz, 3, h, d)
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert not need_weights
        assert qkv.dtype in [paddle.float16, paddle.bfloat16]
        assert qkv.is_cuda

        if cu_seqlens is None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            if key_padding_mask is None:
                qkv = rearrange(qkv, "b s ... -> (b s) ...")
                max_s = seqlen
                cu_seqlens = paddle.arange(
                    0, (batch_size + 1) * seqlen, step=seqlen, dtype=paddle.int32, device=qkv.device
                )
                output = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens,
                    max_s,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=causal,
                )
                output = rearrange(output, "(b s) ... -> b s ...", b=batch_size)
            else:
                nheads = qkv.shape[-2]
                x = rearrange(qkv, "b s three h d -> b s (three h d)")
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
                x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
                output_unpad = flash_attn_varlen_qkvpacked_func(
                    x_unpad,
                    cu_seqlens,
                    max_s,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=causal,
                )
                output = rearrange(
                    pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen),
                    "b s (h d) -> b s h d",
                    h=nheads,
                )
        else:
            assert max_s is not None
            output = flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_s,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
            )

        return output, None


class InternRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(paddle.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(paddle.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


try:
    from apex.normalization import FusedRMSNorm

    InternRMSNorm = FusedRMSNorm  # noqa

    logger.info("Discovered apex.normalization.FusedRMSNorm - will use it instead of InternRMSNorm")
except ImportError:
    # using the normal InternRMSNorm
    pass
except Exception:
    logger.warning("discovered apex but it failed to load, falling back to InternRMSNorm")
    pass


NORM2FN = {
    "rms_norm": InternRMSNorm,
    "layer_norm": nn.LayerNorm,
}


class InternVisionEmbeddings(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(
            paddle.randn(1, 1, self.embed_dim),
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(paddle.randn(1, self.num_positions, self.embed_dim))

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = (
            pos_embed.float()
            .reshape(1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1)
            .permute(0, 3, 1, 2)
        )
        pos_embed = (
            F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False)
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
            .to(target_dtype)
        )
        return pos_embed

    def forward(self, pixel_values: paddle.FloatTensor) -> paddle.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = paddle.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = paddle.cat(
            [self.position_embedding[:, :1, :], self._get_pos_embed(self.position_embedding[:, 1:, :], height, width)],
            dim=1,
        )
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings


class InternAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_flash_attn = config.use_flash_attn and has_flash_attn
        if config.use_flash_attn and not has_flash_attn:
            print("Warning: Flash Attention is not available, use_flash_attn is set to False.")
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale = self.head_dim**-0.5
        self.qkv = GeneralLinear.create(
            self.embed_dim, 3 * self.embed_dim, has_bias=config.qkv_bias, linear_type="default"
        )
        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj_drop = nn.Dropout(config.dropout)

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)

        if self.use_flash_attn:
            self.inner_attn = FlashAttention(attention_dropout=config.attention_dropout)
        self.proj = GeneralLinear.create(self.embed_dim, self.embed_dim, linear_type="default")

    def _naive_attn(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _flash_attn(self, x, key_padding_mask=None, need_weights=False):
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads)

        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
            k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = paddle.stack([q, k, v], dim=2)

        context, _ = self.inner_attn(qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=False)
        outs = self.proj(rearrange(context, "b s h d -> b s (h d)"))
        outs = self.proj_drop(outs)
        return outs

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        x = self._naive_attn(hidden_states) if not self.use_flash_attn else self._flash_attn(hidden_states)
        return x


class InternMLP(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.act = ACT2FN[config.hidden_act]
        self.fc1 = GeneralLinear.create(config.hidden_size, config.intermediate_size, linear_type="default")
        self.fc2 = GeneralLinear.create(config.intermediate_size, config.hidden_size, linear_type="default")

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class InternVisionEncoderLayer(nn.Module):
    def __init__(self, config: InternVisionConfig, drop_path_rate: float):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = config.norm_type

        self.attn = InternAttention(config)
        self.mlp = InternMLP(config)
        self.norm1 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)

        self.ls1 = nn.Parameter(config.initializer_factor * paddle.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * paddle.ones(self.embed_dim))
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(
        self,
        hidden_states: paddle.Tensor,
    ) -> Tuple[paddle.FloatTensor, Optional[paddle.FloatTensor], Optional[Tuple[paddle.FloatTensor]]]:
        """
        Args:
            hidden_states (`Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        hidden_states = hidden_states + self.drop_path1(
            self.attn(self.norm1(hidden_states).to(hidden_states.dtype)) * self.ls1
        )

        hidden_states = hidden_states + self.drop_path2(
            self.mlp(self.norm2(hidden_states).to(hidden_states.dtype)) * self.ls2
        )

        return hidden_states


class InternVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`InternEncoderLayer`].

    Args:
        config (`InternConfig`):
            The corresponding vision configuration for the `InternEncoder`.
    """

    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        # stochastic depth decay rule
        dpr = [x.item() for x in paddle.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        self.layers = nn.ModuleList(
            [InternVisionEncoderLayer(config, dpr[idx]) for idx in range(config.num_hidden_layers)]
        )

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        layer_module,
        hidden_states,
    ):
        def create_custorm_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custorm_forward(layer_module),
            hidden_states,
        )

    def forward(
        self,
        inputs_embeds,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if (
                self.config.recompute
                and not hidden_states.stop_gradient
                and self.config.recompute_granularity == "full"
            ):
                hidden_states = self.recompute_training(encoder_layer, hidden_states)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                )
            hidden_states = layer_outputs

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states)


class InternVisionPretrainedModel(PretrainedModel):
    config_class = InternVisionConfig
    base_model_prefix = "vision_model"
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["InternVisionEncoderLayer"]
    transpose_weight_keys = ["fc1", "fc2", "qkv", "proj"]


class InternVisionModel(InternVisionPretrainedModel):
    def __init__(self, config: InternVisionConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config)

    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        pos_emb = self.embeddings.position_embedding
        _, num_positions, embed_dim = pos_emb.shape
        cls_emb = pos_emb[:, :1, :]
        pos_emb = pos_emb[:, 1:, :].reshape(1, old_size // patch_size, old_size // patch_size, -1).permute(0, 3, 1, 2)
        pos_emb = F.interpolate(pos_emb.float(), size=new_size // patch_size, mode="bicubic", align_corners=False)
        pos_emb = pos_emb.to(cls_emb.dtype).reshape(1, embed_dim, -1).permute(0, 2, 1)
        pos_emb = paddle.cat([cls_emb, pos_emb], dim=1)
        self.embeddings.position_embedding = nn.Parameter(pos_emb)
        self.embeddings.image_size = new_size
        logger.info("Resized position embeddings from {} to {}".format(old_size, new_size))

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
        self,
        pixel_values: Optional[paddle.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_embeds: Optional[paddle.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and pixel_embeds is None:
            raise ValueError("You have to specify pixel_values or pixel_embeds")

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            if len(pixel_values.shape) == 4:
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(f"wrong pixel_values size: {pixel_values.shape}")
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class InternVLChatPretrainedModel(PretrainedModel):
    config_class = InternVLChatConfig
    main_input_name = "pixel_values"
    base_model_prefix = "model"
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["InternVisionModel", "LlamaDecoderLayer", "Qwen2DecoderLayer"]
    transpose_weight_keys = [
        "mlp1.1",
        "mlp1.3",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "qkv",
        "fc1",
        "fc2",
        "proj",
        "qkv_proj",
    ]

    @classmethod
    def _gen_aoa_config(cls, config: InternVLChatConfig):
        model_prefix = "" if cls == cls.base_model_prefix else "model."
        aoa_config = {
            "aoa_statements": [
                # Vision model components
                f"vision_model.encoder.layers.$LAYER_ID.attn.qkv.weight^T -> vision_model.encoder.layers.$LAYER_ID.attn.qkv.weight",
                f"vision_model.encoder.layers.$LAYER_ID.attn.proj.weight^T -> vision_model.encoder.layers.$LAYER_ID.attn.proj.weight",
                f"vision_model.encoder.layers.$LAYER_ID.mlp.fc1.weight^T -> vision_model.encoder.layers.$LAYER_ID.mlp.fc1.weight",
                f"vision_model.encoder.layers.$LAYER_ID.mlp.fc2.weight^T -> vision_model.encoder.layers.$LAYER_ID.mlp.fc2.weight",
                f"vision_model.encoder.layers.$LAYER_ID.norm1.weight -> vision_model.encoder.layers.$LAYER_ID.norm1.weight",
                f"vision_model.encoder.layers.$LAYER_ID.norm2.weight -> vision_model.encoder.layers.$LAYER_ID.norm2.weight",
                f"vision_model.embeddings.patch_embedding.weight^T -> vision_model.embeddings.patch_embedding.weight",
                f"vision_model.embeddings.patch_embedding.bias -> vision_model.embeddings.patch_embedding.bias",
                # Connection MLP
                f"mlp1.1.weight^T -> mlp1.1.weight",
                f"mlp1.3.weight^T -> mlp1.3.weight",
            ]
        }

        # Language model components based on architecture
        if config.llm_config.architectures[0] == "LlamaForCausalLM":
            # Llama-specific mappings
            aoa_config["aoa_statements"] += [
                f"language_model.lm_head.weight -> language_model.lm_head.weight",
                f"language_model.{model_prefix}embed_tokens.weight -> embed_tokens.weight",
                f"language_model.{model_prefix}norm.weight -> norm.weight",
            ]
            aoa_config["aoa_statements"] += [
                f"language_model.{model_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> language_model.{model_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight"
                for x in ("q", "k", "v", "o")
            ]
            aoa_config["aoa_statements"] += [
                f"language_model.{model_prefix}layers.$LAYER_ID.mlp.{p}_proj.weight^T -> language_model.{model_prefix}layers.$LAYER_ID.mlp.{p}_proj.weight"
                for p in ("gate", "up", "down")
            ]
            aoa_config["aoa_statements"] += [
                f"language_model.{model_prefix}layers.$LAYER_ID.input_layernorm.weight -> language_model.{model_prefix}layers.$LAYER_ID.input_layernorm.weight",
                f"language_model.{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> language_model.{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
            ]
        elif config.llm_config.architectures[0] == "Qwen2ForCausalLM":
            # Qwen2-specific mappings
            aoa_config["aoa_statements"] += [
                f"language_model.{model_prefix}lm_head.weight -> language_model.lm_head.weight",
                f"language_model.{model_prefix}embed_tokens.weight -> language_model.{model_prefix}embed_tokens.weight",
                f"language_model.{model_prefix}norm.weight -> language_model.{model_prefix}norm.weight",
            ]
            aoa_config["aoa_statements"] += [
                f"language_model.{model_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> language_model.{model_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight"
                for x in ("q", "k", "v", "o")
            ]
            aoa_config["aoa_statements"] += [
                f"language_model.{model_prefix}layers.$LAYER_ID.mlp.{p}_proj.weight^T -> language_model.{model_prefix}layers.$LAYER_ID.mlp.{p}_proj.weight"
                for p in ("gate", "up", "down")
            ]
            aoa_config["aoa_statements"] += [
                f"language_model.{model_prefix}layers.$LAYER_ID.input_layernorm.weight -> language_model.{model_prefix}layers.$LAYER_ID.input_layernorm.weight",
                f"language_model.{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> language_model.{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
            ]

        return aoa_config

    @classmethod
    def _gen_inv_aoa_config(cls, config: InternVLChatConfig):
        """生成反向AOA配置，用于模型保存时的权重转换"""
        model_prefix = "" if cls == cls.base_model_prefix else "model."
        aoa_statements = [
            # Vision model components - reverse mappings
            f"vision_model.encoder.layers.$LAYER_ID.attn.qkv.weight^T -> vision_model.encoder.layers.$LAYER_ID.attn.qkv.weight",
            f"vision_model.encoder.layers.$LAYER_ID.attn.proj.weight^T -> vision_model.encoder.layers.$LAYER_ID.attn.proj.weight",
            f"vision_model.encoder.layers.$LAYER_ID.mlp.fc1.weight^T -> vision_model.encoder.layers.$LAYER_ID.mlp.fc1.weight",
            f"vision_model.encoder.layers.$LAYER_ID.mlp.fc2.weight^T -> vision_model.encoder.layers.$LAYER_ID.mlp.fc2.weight",
            f"vision_model.encoder.layers.$LAYER_ID.norm1.weight -> vision_model.encoder.layers.$LAYER_ID.norm1.weight",
            f"vision_model.encoder.layers.$LAYER_ID.norm2.weight -> vision_model.encoder.layers.$LAYER_ID.norm2.weight",
            f"vision_model.embeddings.patch_embedding.weight^T -> vision_model.embeddings.patch_embedding.weight",
            f"vision_model.embeddings.patch_embedding.bias -> vision_model.embeddings.patch_embedding.bias",
            # Connection MLP - reverse mappings
            f"mlp1.1.weight^T -> mlp1.1.weight",
            f"mlp1.3.weight^T -> mlp1.3.weight",
        ]

        # Language model components based on architecture
        if config.llm_config.architectures[0] == "LlamaForCausalLM":
            # Llama-specific reverse mappings
            aoa_statements += [
                f"language_model.{model_prefix}lm_head.weight -> language_model.{model_prefix}lm_head.weight",
                f"language_model.{model_prefix}embed_tokens.weight -> language_model.{model_prefix}embed_tokens.weight",
                f"language_model.{model_prefix}norm.weight -> language_model.{model_prefix}norm.weight",
            ]
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight"
                for x in ("q", "k", "v", "o")
            ]
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.mlp.{p}_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.{p}_proj.weight"
                for p in ("gate", "up", "down")
            ]
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.input_layernorm.weight -> {model_prefix}layers.$LAYER_ID.input_layernorm.weight",
                f"{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
            ]
        elif config.llm_config.architectures[0] == "Qwen2ForCausalLM":
            # Qwen2-specific reverse mappings
            aoa_statements += [
                f"language_model.lm_head.weight -> language_model.lm_head.weight",
                f"language_model.{model_prefix}embed_tokens.weight -> language_model.{model_prefix}embed_tokens.weight",
                f"language_model.{model_prefix}norm.weight -> language_model.{model_prefix}norm.weight",
            ]
            aoa_statements += [
                f"language_model.{model_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> language_model.{model_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight"
                for x in ("q", "k", "v", "o")
            ]
            aoa_statements += [
                f"language_model.{model_prefix}layers.$LAYER_ID.mlp.{p}_proj.weight^T -> language_model.{model_prefix}layers.$LAYER_ID.mlp.{p}_proj.weight"
                for p in ("gate", "up", "down")
            ]
            aoa_statements += [
                f"language_model.{model_prefix}layers.$LAYER_ID.input_layernorm.weight -> language_model.{model_prefix}layers.$LAYER_ID.input_layernorm.weight",
                f"language_model.{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> language_model.{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
            ]

        return {"aoa_statements": aoa_statements}


class InternVLChatModel(InternVLChatPretrainedModel):
    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio**2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = "flash_attention_2" if use_flash_attn else "eager"

        logger.info(f"num_image_token: {self.num_image_token}")
        logger.info(f"ps_version: {self.ps_version}")
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == "LlamaForCausalLM":
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == "Qwen2ForCausalLM":
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f"{config.llm_config.architectures[0]} is not implemented.")

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            GeneralLinear.create(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size, linear_type="default"
            ),
            nn.GELU(),
            GeneralLinear.create(llm_hidden_size, llm_hidden_size, linear_type="default"),
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

    def forward(
        self,
        pixel_values: paddle.FloatTensor,
        input_ids: paddle.LongTensor = None,
        attention_mask: paddle.Tensor | None = None,
        position_ids: paddle.LongTensor | None = None,
        image_flags: paddle.LongTensor | None = None,
        past_key_values: list[paddle.FloatTensor] | None = None,
        labels: paddle.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if paddle.distributed.is_initialized() and paddle.distributed.get_rank() == 0:
            print(
                f"dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}"
            )

        input_ids = input_ids.reshape(B * N)
        selected = input_ids == self.img_context_token_id
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(
                f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, "
                f"vit_embeds.shape={vit_embeds.shape}"
            )
            n_token = min(selected.sum(), vit_embeds.size(0))
            input_embeds[selected][:n_token] = input_embeds[selected][:n_token] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

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

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor)))
        if self.ps_version == "v1":
            warnings.warn(
                "In ps_version 'v1', the height and width have not been swapped back, "
                "which results in a transposed image."
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(
        self,
        tokenizer,
        pixel_values,
        questions,
        generation_config,
        num_patches_list=None,
        history=None,
        return_history=False,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
        image_counts=None,
    ):
        if history is not None or return_history:
            print("Now multi-turn chat is not supported in batch_chat.")
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print("Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.")

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and "<image>" not in question:
                question = "<image>\n" + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace("<image>", image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = "left"
        model_inputs = tokenizer(queries, return_tensors="pt", padding=True)
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config["eos_token_id"] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(
        self,
        tokenizer,
        pixel_values,
        question,
        generation_config,
        history=None,
        return_history=False,
        num_patches_list=None,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
    ):

        if history is None and pixel_values is not None and "<image>" not in question:
            question = "<image>\n" + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace("<image>", image_tokens, 1)

        model_inputs = tokenizer(query)
        input_ids = paddle.tensor([model_inputs["input_ids"]])
        attention_mask = paddle.tensor([model_inputs["attention_mask"]])
        generation_config["eos_token_id"] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, **generation_config
        )
        response = tokenizer.batch_decode(generation_output[0], skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, "")
            query_to_print = query_to_print.replace(f"{IMG_START_TOKEN}{IMG_END_TOKEN}", "<image>")
            if verbose:
                print(query_to_print, response)
            return response

    @paddle.no_grad()
    def generate(
        self,
        pixel_values: paddle.FloatTensor | None = None,
        input_ids: paddle.FloatTensor | None = None,
        attention_mask: paddle.LongTensor | None = None,
        visual_features: paddle.FloatTensor | None = None,
        generation_config: GenerationConfig | None = None,
        output_hidden_states: bool | None = None,
        **generate_kwargs,
    ) -> paddle.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = input_ids == self.img_context_token_id
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()


__all__ = [
    "InternVisionModel",
    "InternVLChatModel",
]
