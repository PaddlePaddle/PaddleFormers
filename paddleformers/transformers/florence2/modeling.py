# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import math

import paddle
import paddle.nn.functional as F
from paddle import nn

from ...generation import GenerationMixin
from ..activations import ACT2FN
from ..model_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ..model_utils import PretrainedModel
from .configuration import (
    Florence2Config,
    Florence2LanguageConfig,
    Florence2VisionConfig,
)

__all__ = [
    "Florence2ForConditionalGeneration",
    "Florence2LanguageForConditionalGeneration",
    "Florence2VisionModel",
]


def _expand_mask(mask, dtype, target_length=None):
    target_length = target_length or mask.shape[-1]
    expanded = mask[:, None, None, :].expand([mask.shape[0], 1, target_length, mask.shape[-1]]).astype(dtype)
    return paddle.where(expanded > 0, paddle.zeros_like(expanded), paddle.full_like(expanded, paddle.finfo(dtype).min))


def _causal_mask(batch_size, target_length, past_length, dtype):
    rows = paddle.arange(target_length)[:, None] + past_length
    cols = paddle.arange(target_length + past_length)[None, :]
    allowed = cols <= rows
    mask = paddle.where(
        allowed,
        paddle.zeros([target_length, target_length + past_length], dtype=dtype),
        paddle.full([target_length, target_length + past_length], paddle.finfo(dtype).min, dtype=dtype),
    )
    return mask[None, None, :, :].expand([batch_size, 1, target_length, target_length + past_length])


def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
    shifted = paddle.zeros_like(input_ids)
    shifted[:, 1:] = input_ids[:, :-1].clone()
    shifted[:, 0] = decoder_start_token_id
    return paddle.where(shifted == -100, paddle.full_like(shifted, pad_token_id), shifted)


class DropPath(nn.Layer):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
        return x / keep_prob * paddle.floor(random_tensor)


class LearnedAbsolutePositionEmbedding2D(nn.Layer):
    def __init__(self, embedding_dim=256, num_pos=50):
        super().__init__()
        self.row_embeddings = nn.Embedding(num_pos, embedding_dim // 2)
        self.column_embeddings = nn.Embedding(num_pos, embedding_dim - embedding_dim // 2)

    def forward(self, pixel_values):
        height, width = pixel_values.shape[1:3]
        x_emb = self.column_embeddings(paddle.arange(width))
        y_emb = self.row_embeddings(paddle.arange(height))
        pos = paddle.concat(
            [
                x_emb.unsqueeze(0).tile([height, 1, 1]),
                y_emb.unsqueeze(1).tile([1, width, 1]),
            ],
            axis=-1,
        )
        return pos.unsqueeze(0).tile([pixel_values.shape[0], 1, 1, 1])


class PositionalEmbeddingCosine1D(nn.Layer):
    def __init__(self, embed_dim=512, max_seq_len=1024):
        super().__init__()
        denominator = paddle.exp(-math.log(10000) * paddle.arange(0, embed_dim, 2, dtype="float32") / embed_dim)
        frequencies = paddle.arange(max_seq_len, dtype="float32").reshape([max_seq_len, 1]) * denominator
        values = paddle.zeros([max_seq_len, embed_dim])
        values[:, 0::2] = paddle.sin(frequencies)
        values[:, 1::2] = paddle.cos(frequencies)
        self.register_buffer("pos_idx_to_embed", values, persistable=True)

    def forward(self, seq_embeds):
        values = self.pos_idx_to_embed[: seq_embeds.shape[-2]]
        return values.unsqueeze(0) if seq_embeds.ndim == 3 else values


class PreNorm(nn.Layer):
    def __init__(self, norm, fn, drop_path=None):
        super().__init__()
        self.norm = norm
        self.fn = fn
        self.drop_path = drop_path

    def forward(self, x, *args):
        shortcut = x
        x, size = self.fn(self.norm(x) if self.norm is not None else x, *args)
        if self.drop_path is not None:
            x = self.drop_path(x)
        return shortcut + x, size


class MlpNet(nn.Layer):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.net = MlpNet(in_features, hidden_features)

    def forward(self, x, size):
        return self.net(x), size


class DepthWiseConv2d(nn.Layer):
    def __init__(self, dim_in, kernel_size, padding, stride):
        super().__init__()
        self.dw = nn.Conv2D(dim_in, dim_in, kernel_size, stride=stride, padding=padding, groups=dim_in)

    def forward(self, x, size):
        batch_size, _, channels = x.shape
        height, width = size
        x = self.dw(x.transpose([0, 2, 1]).reshape([batch_size, channels, height, width]))
        size = (x.shape[-2], x.shape[-1])
        return x.flatten(2).transpose([0, 2, 1]), size


class ConvEmbed(nn.Layer):
    def __init__(self, patch_size, in_chans, embed_dim, stride, padding, pre_norm):
        super().__init__()
        self.proj = nn.Conv2D(in_chans, embed_dim, patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(in_chans if pre_norm else embed_dim)
        self.pre_norm = pre_norm

    def forward(self, x, size):
        height, width = size
        if x.ndim == 3:
            if self.pre_norm:
                x = self.norm(x)
            x = x.reshape([x.shape[0], height, width, x.shape[-1]]).transpose([0, 3, 1, 2])
        x = self.proj(x)
        height, width = x.shape[-2:]
        x = x.flatten(2).transpose([0, 2, 1])
        if not self.pre_norm:
            x = self.norm(x)
        return x, (height, width)


class ChannelAttention(nn.Layer):
    def __init__(self, dim, groups):
        super().__init__()
        self.groups = groups
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, size):
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape([batch_size, num_tokens, 3, self.groups, channels // self.groups])
        q, k, v = paddle.unbind(qkv.transpose([2, 0, 3, 1, 4]), axis=0)
        attention = paddle.matmul(
            (q * (float(num_tokens) ** -0.5)).transpose([0, 1, 3, 2]),
            k,
        )
        attention = F.softmax(attention, axis=-1)
        x = paddle.matmul(attention, v.transpose([0, 1, 3, 2])).transpose([0, 1, 3, 2])
        x = self.proj(x.transpose([0, 2, 1, 3]).reshape([batch_size, num_tokens, channels]))
        return x, size


def window_partition(x, window_size):
    batch_size, height, width, channels = x.shape
    x = x.reshape([batch_size, height // window_size, window_size, width // window_size, window_size, channels])
    return x.transpose([0, 1, 3, 2, 4, 5]).reshape([-1, window_size, window_size, channels])


def window_reverse(windows, batch_size, window_size, height, width):
    x = windows.reshape(
        [batch_size, height // window_size, width // window_size, window_size, window_size, windows.shape[-1]]
    )
    return x.transpose([0, 1, 3, 2, 4, 5]).reshape([batch_size, height, width, windows.shape[-1]])


class WindowAttention(nn.Layer):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = float(dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, size):
        height, width = size
        batch_size, _, channels = x.shape
        x = x.reshape([batch_size, height, width, channels])
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        if pad_right or pad_bottom:
            x = F.pad(x.transpose([0, 3, 1, 2]), [0, pad_right, 0, pad_bottom]).transpose([0, 2, 3, 1])
        padded_height, padded_width = x.shape[1:3]
        x = window_partition(x, self.window_size).reshape([-1, self.window_size**2, channels])
        qkv = self.qkv(x).reshape([-1, self.window_size**2, 3, self.num_heads, channels // self.num_heads])
        q, k, v = paddle.unbind(qkv.transpose([2, 0, 3, 1, 4]), axis=0)
        attention = F.softmax(paddle.matmul(q * self.scale, k.transpose([0, 1, 3, 2])), axis=-1)
        x = paddle.matmul(attention, v).transpose([0, 2, 1, 3]).reshape([-1, self.window_size**2, channels])
        x = self.proj(x).reshape([-1, self.window_size, self.window_size, channels])
        x = window_reverse(x, batch_size, self.window_size, padded_height, padded_width)
        return x[:, :height, :width].reshape([batch_size, height * width, channels]), size


class SpatialBlock(nn.Layer):
    def __init__(self, dim, num_heads, window_size, drop_path_rate):
        super().__init__()
        drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else None
        self.conv1 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1))
        self.window_attn = PreNorm(nn.LayerNorm(dim), WindowAttention(dim, num_heads, window_size), drop_path)
        self.conv2 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1))
        self.ffn = PreNorm(nn.LayerNorm(dim), Mlp(dim, dim * 4), drop_path)

    def forward(self, x, size):
        x, size = self.conv1(x, size)
        x, size = self.window_attn(x, size)
        x, size = self.conv2(x, size)
        return self.ffn(x, size)


class ChannelBlock(nn.Layer):
    def __init__(self, dim, groups, drop_path_rate):
        super().__init__()
        drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else None
        self.conv1 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1))
        self.channel_attn = PreNorm(nn.LayerNorm(dim), ChannelAttention(dim, groups), drop_path)
        self.conv2 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1))
        self.ffn = PreNorm(nn.LayerNorm(dim), Mlp(dim, dim * 4), drop_path)

    def forward(self, x, size):
        x, size = self.conv1(x, size)
        x, size = self.channel_attn(x, size)
        x, size = self.conv2(x, size)
        return self.ffn(x, size)


class DaViTBlock(nn.Layer):
    def __init__(self, dim, num_heads, groups, window_size, spatial_drop, channel_drop):
        super().__init__()
        self.spatial_block = SpatialBlock(dim, num_heads, window_size, spatial_drop)
        self.channel_block = ChannelBlock(dim, groups, channel_drop)

    def forward(self, x, size):
        x, size = self.spatial_block(x, size)
        return self.channel_block(x, size)


class DaViT(nn.Layer):
    def __init__(self, config):
        super().__init__()
        dpr = paddle.linspace(0, config.drop_path_rate, sum(config.depths) * 2).tolist()
        self.convs = nn.LayerList()
        self.blocks = nn.LayerList()
        offset = 0
        for index, depth in enumerate(config.depths):
            self.convs.append(
                ConvEmbed(
                    config.patch_size[index],
                    3 if index == 0 else config.dim_embed[index - 1],
                    config.dim_embed[index],
                    config.patch_stride[index],
                    config.patch_padding[index],
                    config.patch_prenorm[index],
                )
            )
            self.blocks.append(
                nn.LayerList(
                    [
                        DaViTBlock(
                            config.dim_embed[index],
                            config.num_heads[index],
                            config.num_groups[index],
                            config.window_size,
                            dpr[offset + layer * 2],
                            dpr[offset + layer * 2 + 1],
                        )
                        for layer in range(depth)
                    ]
                )
            )
            offset += depth * 2

    def forward_features_unpool(self, x):
        size = (x.shape[2], x.shape[3])
        for conv, blocks in zip(self.convs, self.blocks):
            x, size = conv(x, size)
            for block in blocks:
                x, size = block(x, size)
        return x


class Florence2Attention(nn.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.0, is_decoder=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _shape(self, tensor, seq_len, batch_size):
        return tensor.reshape([batch_size, seq_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])

    def forward(
        self,
        hidden_states,
        key_value_states=None,
        past_key_value=None,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
    ):
        is_cross_attention = key_value_states is not None
        batch_size, target_length, _ = hidden_states.shape
        query_states = self._shape(self.q_proj(hidden_states) * self.scaling, target_length, batch_size)
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            key_states, value_states = past_key_value
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), key_value_states.shape[1], batch_size)
            value_states = self._shape(self.v_proj(key_value_states), key_value_states.shape[1], batch_size)
        else:
            key_states = self._shape(self.k_proj(hidden_states), target_length, batch_size)
            value_states = self._shape(self.v_proj(hidden_states), target_length, batch_size)
            if past_key_value is not None:
                key_states = paddle.concat([past_key_value[0], key_states], axis=2)
                value_states = paddle.concat([past_key_value[1], value_states], axis=2)
        present = (key_states, value_states) if self.is_decoder else None
        attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2]))
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, axis=-1)
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask.reshape([1, -1, 1, 1])
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        output = paddle.matmul(attn_probs, value_states).transpose([0, 2, 1, 3])
        output = self.out_proj(output.reshape([batch_size, target_length, self.embed_dim]))
        return output, attn_weights if output_attentions else None, present


class Florence2EncoderLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Florence2Attention(config.d_model, config.encoder_attention_heads, config.attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.activation_fn = ACT2FN[config.activation_function]
        self.dropout = config.dropout
        self.activation_dropout = config.activation_dropout

    def forward(self, hidden_states, attention_mask=None, layer_head_mask=None, output_attentions=False):
        residual = hidden_states
        hidden_states, weights, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_layer_norm(
            residual + F.dropout(hidden_states, p=self.dropout, training=self.training)
        )
        residual = hidden_states
        hidden_states = self.fc2(
            F.dropout(self.activation_fn(self.fc1(hidden_states)), p=self.activation_dropout, training=self.training)
        )
        hidden_states = self.final_layer_norm(
            residual + F.dropout(hidden_states, p=self.dropout, training=self.training)
        )
        return (hidden_states, weights) if output_attentions else (hidden_states,)


class Florence2DecoderLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Florence2Attention(
            config.d_model, config.decoder_attention_heads, config.attention_dropout, is_decoder=True
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.encoder_attn = Florence2Attention(
            config.d_model, config.decoder_attention_heads, config.attention_dropout, is_decoder=True
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.activation_fn = ACT2FN[config.activation_function]
        self.dropout = config.dropout
        self.activation_dropout = config.activation_dropout

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=True,
    ):
        residual = hidden_states
        hidden_states, self_weights, present = self.self_attn(
            hidden_states,
            past_key_value=past_key_value[:2] if past_key_value is not None else None,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_layer_norm(
            residual + F.dropout(hidden_states, p=self.dropout, training=self.training)
        )
        cross_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states, cross_weights, cross_present = self.encoder_attn(
                hidden_states,
                key_value_states=encoder_hidden_states,
                past_key_value=past_key_value[-2:] if past_key_value is not None else None,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                output_attentions=output_attentions,
            )
            hidden_states = self.encoder_attn_layer_norm(
                residual + F.dropout(hidden_states, p=self.dropout, training=self.training)
            )
            present = present + cross_present
        residual = hidden_states
        hidden_states = self.fc2(
            F.dropout(self.activation_fn(self.fc1(hidden_states)), p=self.activation_dropout, training=self.training)
        )
        hidden_states = self.final_layer_norm(
            residual + F.dropout(hidden_states, p=self.dropout, training=self.training)
        )
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_weights, cross_weights)
        if use_cache:
            outputs += (present,)
        return outputs


class Florence2LearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids, past_key_values_length=0):
        positions = paddle.arange(past_key_values_length, past_key_values_length + input_ids.shape[1], dtype="int64")
        return super().forward(positions.unsqueeze(0).expand([input_ids.shape[0], -1]) + self.offset)


class Florence2Encoder(nn.Layer):
    def __init__(self, config, embed_tokens):
        super().__init__()
        self.config = config
        self.embed_tokens = embed_tokens
        self.embed_positions = Florence2LearnedPositionalEmbedding(config.max_position_embeddings, config.d_model)
        self.layers = nn.LayerList([Florence2EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.dropout = config.dropout

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        position_source = input_ids if input_ids is not None else paddle.zeros(inputs_embeds.shape[:2], dtype="int64")
        hidden_states = self.layernorm_embedding(inputs_embeds + self.embed_positions(position_source))
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        expanded_mask = _expand_mask(attention_mask, hidden_states.dtype) if attention_mask is not None else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for index, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            outputs = layer(
                hidden_states,
                expanded_mask,
                head_mask[index] if head_mask is not None else None,
                output_attentions,
            )
            hidden_states = outputs[0]
            if output_attentions:
                all_attentions += (outputs[1],)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple(x for x in [hidden_states, all_hidden_states, all_attentions] if x is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class Florence2Decoder(nn.Layer):
    def __init__(self, config, embed_tokens):
        super().__init__()
        self.config = config
        self.embed_tokens = embed_tokens
        self.embed_positions = Florence2LearnedPositionalEmbedding(config.max_position_embeddings, config.d_model)
        self.layers = nn.LayerList([Florence2DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.dropout = config.dropout

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        position_source = input_ids if input_ids is not None else paddle.zeros(inputs_embeds.shape[:2], dtype="int64")
        hidden_states = self.layernorm_embedding(inputs_embeds + self.embed_positions(position_source, past_length))
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        causal_mask = _causal_mask(hidden_states.shape[0], hidden_states.shape[1], past_length, hidden_states.dtype)
        if attention_mask is not None:
            causal_mask = causal_mask + _expand_mask(attention_mask, hidden_states.dtype, hidden_states.shape[1])
        encoder_mask = (
            _expand_mask(encoder_attention_mask, hidden_states.dtype, hidden_states.shape[1])
            if encoder_attention_mask is not None
            else None
        )
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        next_cache = () if use_cache else None
        for index, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            outputs = layer(
                hidden_states,
                causal_mask,
                encoder_hidden_states,
                encoder_mask,
                head_mask[index] if head_mask is not None else None,
                cross_attn_head_mask[index] if cross_attn_head_mask is not None else None,
                past_key_values[index] if past_key_values is not None else None,
                output_attentions,
                use_cache,
            )
            hidden_states = outputs[0]
            if use_cache:
                next_cache += (outputs[3 if output_attentions else 1],)
            if output_attentions:
                all_self_attentions += (outputs[1],)
                all_cross_attentions += (outputs[2],)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple(
                x
                for x in [hidden_states, next_cache, all_hidden_states, all_self_attentions, all_cross_attentions]
                if x is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class Florence2LanguageModel(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.encoder = Florence2Encoder(config, self.shared)
        self.decoder = Florence2Decoder(config, self.shared)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_output=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        **kwargs,
    ):
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids,
                self.config.pad_token_id,
                self.config.decoder_start_token_id,
            )
        encoder_outputs = encoder_outputs if encoder_outputs is not None else encoder_output
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        encoder_hidden_states = (
            encoder_outputs.last_hidden_state if hasattr(encoder_outputs, "last_hidden_state") else encoder_outputs[0]
        )
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if not return_dict:
            return decoder_outputs + encoder_outputs
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_hidden_states,
            encoder_hidden_states=getattr(encoder_outputs, "hidden_states", None),
            encoder_attentions=getattr(encoder_outputs, "attentions", None),
        )


class Florence2LanguagePretrainedModel(PretrainedModel):
    config_class = Florence2LanguageConfig
    base_model_prefix = "model"
    transpose_weight_keys = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


class Florence2LanguageForConditionalGeneration(Florence2LanguagePretrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.is_encoder_decoder = True
        self.model = Florence2LanguageModel(config)
        self.register_buffer("final_logits_bias", paddle.zeros([1, config.vocab_size]), persistable=True)

    def get_encoder(self):
        return self.model.encoder

    def get_decoder(self):
        return self.model.decoder

    def get_input_embeddings(self):
        return self.model.shared

    def set_input_embeddings(self, value):
        self.model.shared = value
        self.model.encoder.embed_tokens = value
        self.model.decoder.embed_tokens = value
        self.config.vocab_size = value.weight.shape[0]
        self.final_logits_bias = paddle.zeros([1, value.weight.shape[0]], dtype=value.weight.dtype)

    def forward(self, labels=None, return_dict=True, **kwargs):
        if labels is not None and kwargs.get("decoder_input_ids") is None:
            kwargs["decoder_input_ids"] = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )
        outputs = self.model(return_dict=return_dict, **kwargs)
        hidden_states = outputs.last_hidden_state if return_dict else outputs[0]
        logits = paddle.matmul(hidden_states, self.model.shared.weight, transpose_y=True) + self.final_logits_bias
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape([-1, self.config.vocab_size]),
                labels.reshape([-1]),
                ignore_index=-100,
            )
        if not return_dict:
            return ((loss, logits) if loss is not None else (logits,)) + outputs[1:]
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_output": kwargs.get("encoder_output"),
            "attention_mask": kwargs.get("attention_mask"),
            "use_cache": kwargs.get("use_cache", True),
            "return_dict": True,
        }

    def update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False):
        model_kwargs = GenerationMixin.update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder
        )
        if isinstance(outputs, Seq2SeqLMOutput) and outputs.past_key_values is not None:
            model_kwargs["past_key_values"] = outputs.past_key_values
        return model_kwargs

    @staticmethod
    def _expand_encoder_output_for_generation(encoder_output, index):
        if isinstance(encoder_output, BaseModelOutput):
            return BaseModelOutput(
                last_hidden_state=paddle.gather(encoder_output.last_hidden_state, index),
                hidden_states=(
                    tuple(paddle.gather(state, index) for state in encoder_output.hidden_states)
                    if encoder_output.hidden_states is not None
                    else None
                ),
                attentions=(
                    tuple(paddle.gather(state, index) for state in encoder_output.attentions)
                    if encoder_output.attentions is not None
                    else None
                ),
            )
        if isinstance(encoder_output, tuple):
            return tuple(paddle.gather(state, index) if state is not None else None for state in encoder_output)
        return paddle.gather(encoder_output, index)

    def expand_inputs_for_generation(self, input_ids, expand_size, attention_mask=None, **model_kwargs):
        encoder_output = model_kwargs.pop("encoder_output", None)
        index = paddle.tile(paddle.arange(input_ids.shape[0], dtype="int64").unsqueeze(-1), [1, expand_size]).reshape(
            [-1]
        )
        input_ids, model_kwargs = super().expand_inputs_for_generation(
            input_ids,
            expand_size,
            attention_mask=attention_mask,
            **model_kwargs,
        )
        if encoder_output is not None:
            model_kwargs["encoder_output"] = self._expand_encoder_output_for_generation(encoder_output, index)
        return input_ids, model_kwargs

    def _reorder_cache(self, past_key_values, beam_idx):
        return tuple(
            tuple(paddle.index_select(state, beam_idx, axis=0) for state in layer) for layer in past_key_values
        )


class Florence2PretrainedModel(PretrainedModel):
    config_class = Florence2Config
    base_model_prefix = ""
    _keys_to_ignore_on_load_missing = [
        r"language_model.model.encoder.embed_tokens.weight",
        r"language_model.model.decoder.embed_tokens.weight",
    ]
    transpose_weight_keys = [
        "qkv",
        "proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2",
    ]


class Florence2VisionModel(Florence2PretrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config: Florence2VisionConfig):
        super().__init__(config)
        self.vision_tower = DaViT(config)

    def forward(self, pixel_values):
        return self.vision_tower.forward_features_unpool(pixel_values)


class Florence2ForConditionalGeneration(Florence2PretrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.is_encoder_decoder = True
        self.vision_tower = DaViT(config.vision_config)
        image_dim = config.vision_config.dim_embed[-1]
        projection_dim = config.vision_config.projection_dim
        self.image_projection = self.create_parameter([image_dim, projection_dim])
        self.image_proj_norm = nn.LayerNorm(projection_dim)
        self.image_pos_embed = LearnedAbsolutePositionEmbedding2D(
            image_dim, config.vision_config.image_pos_embed["max_pos_embeddings"]
        )
        self.visual_temporal_embed = PositionalEmbeddingCosine1D(
            image_dim, config.vision_config.visual_temporal_embedding["max_temporal_embeddings"]
        )
        self.image_feature_source = config.vision_config.image_feature_source
        self.language_model = Florence2LanguageForConditionalGeneration(config.text_config)

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)
        self.config.vocab_size = value.weight.shape[0]
        self.config.text_config.vocab_size = value.weight.shape[0]

    def _encode_image(self, pixel_values):
        batch_size = pixel_values.shape[0]
        x = self.vision_tower.forward_features_unpool(pixel_values)
        num_tokens = x.shape[1]
        height = width = int(num_tokens**0.5)
        x = x.reshape([batch_size, height, width, x.shape[-1]])
        x = x + self.image_pos_embed(x)
        x = x.reshape([batch_size, 1, height * width, x.shape[-1]])
        x = x + self.visual_temporal_embed(x[:, :, 0]).reshape([1, 1, 1, x.shape[-1]])
        features = {
            "spatial_avg_pool": x.mean(axis=2),
            "temporal_avg_pool": x.mean(axis=1),
            "last_frame": x[:, -1],
        }
        x = paddle.concat([features[source] for source in self.image_feature_source], axis=1)
        return self.image_proj_norm(paddle.matmul(x, self.image_projection))

    def _merge_image_features(self, image_features, inputs_embeds=None, attention_mask=None):
        image_mask = paddle.ones(image_features.shape[:2], dtype=image_features.dtype)
        if inputs_embeds is None:
            return image_features, image_mask
        text_mask = (
            attention_mask.astype(inputs_embeds.dtype)
            if attention_mask is not None
            else paddle.ones(inputs_embeds.shape[:2], dtype=inputs_embeds.dtype)
        )
        return paddle.concat([image_features, inputs_embeds], axis=1), paddle.concat([image_mask, text_mask], axis=1)

    def _split_sft_inputs(self, input_ids, labels, attention_mask):
        source_rows, label_rows = [], []
        max_source = 1
        max_target = 1
        for row, label_row in zip(input_ids.tolist(), labels.tolist()):
            target_start = next((index for index, value in enumerate(label_row) if value != -100), len(row))
            # PaddleFormers SFT labels are shifted left once, so the first
            # supervised label predicts the token after this source position.
            source = row[: target_start + 1] or [self.config.bos_token_id]
            target = [value for value in label_row[target_start:] if value != -100]
            source_rows.append(source)
            label_rows.append(target or [self.config.eos_token_id])
            max_source = max(max_source, len(source))
            max_target = max(max_target, len(target))
        source_ids = paddle.full([len(source_rows), max_source], self.config.pad_token_id, dtype=input_ids.dtype)
        source_mask = paddle.zeros([len(source_rows), max_source], dtype="int64")
        decoder_labels = paddle.full([len(label_rows), max_target], -100, dtype=labels.dtype)
        for index, (source, target) in enumerate(zip(source_rows, label_rows)):
            source_ids[index, : len(source)] = paddle.to_tensor(source, dtype=input_ids.dtype)
            source_mask[index, : len(source)] = 1
            decoder_labels[index, : len(target)] = paddle.to_tensor(target, dtype=labels.dtype)
        return source_ids, decoder_labels, source_mask

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_output=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        **kwargs,
    ):
        if labels is not None and input_ids is not None and labels.shape == input_ids.shape:
            input_ids, labels, attention_mask = self._split_sft_inputs(input_ids, labels, attention_mask)
        image_features = None
        if encoder_output is None and encoder_outputs is None and inputs_embeds is None:
            if input_ids is not None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                image_features = self._encode_image(pixel_values)
                inputs_embeds, attention_mask = self._merge_image_features(
                    image_features,
                    inputs_embeds,
                    attention_mask,
                )
        outputs = self.language_model(
            input_ids=None if inputs_embeds is not None else input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_output=encoder_output if encoder_output is not None else encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs

    def generate(self, input_ids=None, pixel_values=None, inputs_embeds=None, attention_mask=None, **kwargs):
        if inputs_embeds is None:
            if input_ids is not None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                inputs_embeds, attention_mask = self._merge_image_features(
                    self._encode_image(pixel_values), inputs_embeds, attention_mask
                )
        return self.language_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)
