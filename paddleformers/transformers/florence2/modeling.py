# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

import math

import paddle
import paddle.nn.functional as F
from paddle import nn

from ..model_outputs import BaseModelOutput, Seq2SeqLMOutput, Seq2SeqModelOutput
from ..model_utils import PretrainedModel
from .configuration import Florence2Config, Florence2VisionConfig

__all__ = [
    "DaViT",
    "Florence2VisionModel",
    "Florence2VisionModelWithProjection",
    "Florence2Model",
    "Florence2ForConditionalGeneration",
]


def _drop_path(x, rate, training):
    if rate == 0.0 or not training:
        return x
    keep = 1.0 - rate
    return x * paddle.floor(keep + paddle.rand([x.shape[0]] + [1] * (x.ndim - 1), dtype=x.dtype)) / keep


def _activation(x, name):
    if name == "gelu":
        return F.gelu(x)
    if name == "relu":
        return F.relu(x)
    if name == "silu":
        return F.silu(x)
    raise ValueError(f"Unsupported activation_function: {name}")


def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
    if pad_token_id is None:
        raise ValueError("pad_token_id has to be defined.")
    shifted_input_ids = paddle.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id
    return paddle.where(
        shifted_input_ids == -100, paddle.full_like(shifted_input_ids, pad_token_id), shifted_input_ids
    )


class Florence2LearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids, past_key_values_length=0):
        positions = (
            paddle.arange(past_key_values_length, past_key_values_length + input_ids.shape[1], dtype="int64")
            .unsqueeze(0)
            .tile([input_ids.shape[0], 1])
        )
        return super().forward(positions + self.offset)


class LearnedAbsolutePositionEmbedding2D(nn.Layer):
    def __init__(self, embedding_dim=256, num_pos=50):
        super().__init__()
        self.row_embeddings = nn.Embedding(num_pos, embedding_dim // 2)
        self.column_embeddings = nn.Embedding(num_pos, embedding_dim - embedding_dim // 2)

    def forward(self, pixel_values):
        h, w = pixel_values.shape[1:3]
        x = self.column_embeddings(paddle.arange(w, dtype="int64"))
        y = self.row_embeddings(paddle.arange(h, dtype="int64"))
        return (
            paddle.concat([x.unsqueeze(0).tile([h, 1, 1]), y.unsqueeze(1).tile([1, w, 1])], axis=-1)
            .unsqueeze(0)
            .tile([pixel_values.shape[0], 1, 1, 1])
        )


class PositionalEmbeddingCosine1D(nn.Layer):
    def __init__(self, embed_dim=512, max_seq_len=1024):
        super().__init__()
        denominator = paddle.exp(-math.log(10000) * paddle.arange(0, embed_dim, 2, dtype="float32") / embed_dim)
        frequencies = paddle.arange(max_seq_len, dtype="float32").reshape([max_seq_len, 1]) * denominator
        values = paddle.zeros([max_seq_len, embed_dim], dtype="float32")
        values[:, 0::2] = paddle.sin(frequencies)
        values[:, 1::2] = paddle.cos(frequencies)
        self.register_buffer("pos_idx_to_embed", values, persistable=True)

    def forward(self, x):
        return (
            self.pos_idx_to_embed[: x.shape[-2]].unsqueeze(0) if x.ndim == 3 else self.pos_idx_to_embed[: x.shape[-2]]
        )


class DepthWiseConv2d(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2D(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x, size):
        b, n, c = x.shape
        h, w = size
        x = self.dw(x.transpose([0, 2, 1]).reshape([b, c, h, w]))
        return x.flatten(2).transpose([0, 2, 1]), (x.shape[-2], x.shape[-1])


class ConvEmbed(nn.Layer):
    def __init__(self, in_chans, embed_dim, patch_size, stride, padding, pre_norm):
        super().__init__()
        self.proj = nn.Conv2D(in_chans, embed_dim, patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(in_chans if pre_norm else embed_dim)
        self.pre_norm = pre_norm

    def forward(self, x, size):
        if x.ndim == 3:
            if self.pre_norm:
                x = self.norm(x)
            x = x.reshape([x.shape[0], size[0], size[1], x.shape[-1]]).transpose([0, 3, 1, 2])
        x = self.proj(x)
        size = (x.shape[-2], x.shape[-1])
        x = x.flatten(2).transpose([0, 2, 1])
        return (x if self.pre_norm else self.norm(x)), size


class Mlp(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Layer()
        self.net.add_sublayer("fc1", nn.Linear(dim, dim * 4))
        self.net.add_sublayer("act", nn.GELU())
        self.net.add_sublayer("fc2", nn.Linear(dim * 4, dim))

    def forward(self, x, size):
        return self.net.fc2(self.net.act(self.net.fc1(x))), size


class WindowAttention(nn.Layer):
    def __init__(self, dim, heads, window_size):
        super().__init__()
        self.num_heads, self.window_size, self.scale = heads, window_size, (dim // heads) ** -0.5
        self.qkv, self.proj = nn.Linear(dim, dim * 3), nn.Linear(dim, dim)

    def forward(self, x, size):
        b, n, c = x.shape
        h, w = size
        x = x.reshape([b, h, w, c])
        ph, pw = (-h) % self.window_size, (-w) % self.window_size
        if ph or pw:
            x = F.pad(x.transpose([0, 3, 1, 2]), [0, pw, 0, ph]).transpose([0, 2, 3, 1])
        hp, wp = x.shape[1:3]
        ws = self.window_size
        x = x.reshape([b, hp // ws, ws, wp // ws, ws, c]).transpose([0, 1, 3, 2, 4, 5]).reshape([-1, ws * ws, c])
        qkv = (
            self.qkv(x)
            .reshape([x.shape[0], x.shape[1], 3, self.num_heads, c // self.num_heads])
            .transpose([2, 0, 3, 1, 4])
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = paddle.matmul(F.softmax(paddle.matmul(q * self.scale, k.transpose([0, 1, 3, 2])), axis=-1), v)
        x = self.proj(x.transpose([0, 2, 1, 3]).reshape([-1, ws * ws, c]))
        x = x.reshape([b, hp // ws, wp // ws, ws, ws, c]).transpose([0, 1, 3, 2, 4, 5]).reshape([b, hp, wp, c])
        return x[:, :h, :w].reshape([b, n, c]), size


class ChannelAttention(nn.Layer):
    def __init__(self, dim, groups):
        super().__init__()
        self.groups = groups
        self.qkv, self.proj = nn.Linear(dim, dim * 3), nn.Linear(dim, dim)

    def forward(self, x, size):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape([b, n, 3, self.groups, c // self.groups]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.softmax(paddle.matmul((q * n**-0.5).transpose([0, 1, 3, 2]), k), axis=-1)
        x = paddle.matmul(attn, v.transpose([0, 1, 3, 2])).transpose([0, 3, 1, 2]).reshape([b, n, c])
        return self.proj(x), size


class PreNorm(nn.Layer):
    def __init__(self, norm, fn, rate=0.0):
        super().__init__()
        self.norm, self.fn, self.rate = norm, fn, rate

    def forward(self, x, size):
        residual = x
        x, size = self.fn(self.norm(x) if self.norm is not None else x, size)
        return residual + _drop_path(x, self.rate, self.training), size


class SpatialBlock(nn.Layer):
    def __init__(self, dim, heads, window, rate):
        super().__init__()
        self.conv1 = PreNorm(None, DepthWiseConv2d(dim))
        self.window_attn = PreNorm(nn.LayerNorm(dim), WindowAttention(dim, heads, window), rate)
        self.conv2 = PreNorm(None, DepthWiseConv2d(dim))
        self.ffn = PreNorm(nn.LayerNorm(dim), Mlp(dim), rate)

    def forward(self, x, size):
        x, size = self.conv1(x, size)
        x, size = self.window_attn(x, size)
        x, size = self.conv2(x, size)
        return self.ffn(x, size)


class ChannelBlock(nn.Layer):
    def __init__(self, dim, groups, rate):
        super().__init__()
        self.conv1 = PreNorm(None, DepthWiseConv2d(dim))
        self.channel_attn = PreNorm(nn.LayerNorm(dim), ChannelAttention(dim, groups), rate)
        self.conv2 = PreNorm(None, DepthWiseConv2d(dim))
        self.ffn = PreNorm(nn.LayerNorm(dim), Mlp(dim), rate)

    def forward(self, x, size):
        x, size = self.conv1(x, size)
        x, size = self.channel_attn(x, size)
        x, size = self.conv2(x, size)
        return self.ffn(x, size)


class DaViT(nn.Layer):
    def __init__(self, config):
        super().__init__()
        rates = paddle.linspace(0, config.drop_path_rate, sum(config.depths) * 2, dtype="float32").tolist()
        self.convs, self.blocks = nn.LayerList(), nn.LayerList()
        offset = 0
        for i, depth in enumerate(config.depths):
            self.convs.append(
                ConvEmbed(
                    3 if i == 0 else config.dim_embed[i - 1],
                    config.dim_embed[i],
                    config.patch_size[i],
                    config.patch_stride[i],
                    config.patch_padding[i],
                    config.patch_prenorm[i],
                )
            )
            layers = nn.LayerList()
            for j in range(depth):
                layer = nn.Layer()
                layer.add_sublayer(
                    "spatial_block",
                    SpatialBlock(config.dim_embed[i], config.num_heads[i], config.window_size, rates[offset + 2 * j]),
                )
                layer.add_sublayer(
                    "channel_block", ChannelBlock(config.dim_embed[i], config.num_groups[i], rates[offset + 2 * j + 1])
                )
                layers.append(layer)
            self.blocks.append(layers)
            offset += 2 * depth
        self.norms = nn.LayerNorm(config.dim_embed[-1])
        self.head = nn.Linear(config.dim_embed[-1], 1000)

    def forward_features_unpool(self, x, return_size=False):
        size = (x.shape[-2], x.shape[-1])
        for conv, layers in zip(self.convs, self.blocks):
            x, size = conv(x, size)
            for layer in layers:
                x, size = layer.spatial_block(x, size)
                x, size = layer.channel_block(x, size)
        return (x, size) if return_size else x


class Florence2Attention(nn.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.0, is_decoder=False, is_causal=False):
        super().__init__()
        self.embed_dim, self.num_heads, self.head_dim = embed_dim, num_heads, embed_dim // num_heads
        self.scaling, self.dropout, self.is_decoder, self.is_causal = (
            self.head_dim**-0.5,
            dropout,
            is_decoder,
            is_causal,
        )
        self.k_proj, self.v_proj, self.q_proj, self.out_proj = (
            nn.Linear(embed_dim, embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

    def _shape(self, x, b):
        return x.reshape([b, -1, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])

    def forward(
        self, hidden_states, key_value_states=None, past_key_value=None, attention_mask=None, output_attentions=False
    ):
        b, tgt, _ = hidden_states.shape
        cross = key_value_states is not None
        q = self._shape(self.q_proj(hidden_states) * self.scaling, b)
        if cross and past_key_value is not None and past_key_value[0].shape[2] == key_value_states.shape[1]:
            k, v = past_key_value
        elif cross:
            k, v = self._shape(self.k_proj(key_value_states), b), self._shape(self.v_proj(key_value_states), b)
        else:
            k, v = self._shape(self.k_proj(hidden_states), b), self._shape(self.v_proj(hidden_states), b)
            if past_key_value is not None:
                k, v = paddle.concat([past_key_value[0], k], axis=2), paddle.concat([past_key_value[1], v], axis=2)
        present = (k, v) if self.is_decoder else None
        weights = paddle.matmul(q, k.transpose([0, 1, 3, 2]))
        if attention_mask is not None:
            weights = weights + attention_mask
        probs = F.softmax(weights, axis=-1)
        out = (
            paddle.matmul(F.dropout(probs, p=self.dropout, training=self.training), v)
            .transpose([0, 2, 1, 3])
            .reshape([b, tgt, self.embed_dim])
        )
        return self.out_proj(out), (probs if output_attentions else None), present


class Florence2EncoderLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.self_attn = Florence2Attention(d, config.encoder_attention_heads, config.attention_dropout)
        self.self_attn_layer_norm, self.fc1, self.fc2, self.final_layer_norm = (
            nn.LayerNorm(d),
            nn.Linear(d, config.encoder_ffn_dim),
            nn.Linear(config.encoder_ffn_dim, d),
            nn.LayerNorm(d),
        )
        self.dropout, self.activation_dropout, self.activation_function = (
            config.dropout,
            config.activation_dropout,
            config.activation_function,
        )

    def forward(self, x, mask=None, output_attentions=False):
        residual = x
        x, attn, _ = self.self_attn(x, attention_mask=mask, output_attentions=output_attentions)
        x = self.self_attn_layer_norm(residual + F.dropout(x, p=self.dropout, training=self.training))
        residual = x
        x = self.fc2(
            F.dropout(
                _activation(self.fc1(x), self.activation_function), p=self.activation_dropout, training=self.training
            )
        )
        return self.final_layer_norm(residual + F.dropout(x, p=self.dropout, training=self.training)), attn


class Florence2DecoderLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.self_attn = Florence2Attention(d, config.decoder_attention_heads, config.attention_dropout, True, True)
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.encoder_attn = Florence2Attention(d, config.decoder_attention_heads, config.attention_dropout, True)
        self.encoder_attn_layer_norm, self.fc1, self.fc2, self.final_layer_norm = (
            nn.LayerNorm(d),
            nn.Linear(d, config.decoder_ffn_dim),
            nn.Linear(config.decoder_ffn_dim, d),
            nn.LayerNorm(d),
        )
        self.dropout, self.activation_dropout, self.activation_function = (
            config.dropout,
            config.activation_dropout,
            config.activation_function,
        )

    def forward(
        self,
        x,
        mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        use_cache=True,
        output_attentions=False,
    ):
        residual = x
        x, self_attn, present = self.self_attn(
            x,
            past_key_value=past_key_value[:2] if past_key_value else None,
            attention_mask=mask,
            output_attentions=output_attentions,
        )
        x = self.self_attn_layer_norm(residual + F.dropout(x, p=self.dropout, training=self.training))
        cross_attn = None
        if encoder_hidden_states is not None:
            residual = x
            x, cross_attn, cross_present = self.encoder_attn(
                x,
                encoder_hidden_states,
                past_key_value[-2:] if past_key_value else None,
                attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            x = self.encoder_attn_layer_norm(residual + F.dropout(x, p=self.dropout, training=self.training))
            present = present + cross_present
        residual = x
        x = self.fc2(
            F.dropout(
                _activation(self.fc1(x), self.activation_function), p=self.activation_dropout, training=self.training
            )
        )
        return (
            self.final_layer_norm(residual + F.dropout(x, p=self.dropout, training=self.training)),
            self_attn,
            cross_attn,
            (present if use_cache else None),
        )


def _expand_mask(mask, dtype, tgt_len=None):
    if mask is None:
        return None
    tgt_len = tgt_len or mask.shape[1]
    expanded = mask.unsqueeze([1, 2]).tile([1, 1, tgt_len, 1]).astype(dtype)
    return (1.0 - expanded) * paddle.finfo(dtype).min


def _causal_mask(mask, bsz, tgt_len, past_len, dtype):
    src_len = tgt_len + past_len
    causal = paddle.full([tgt_len, src_len], paddle.finfo(dtype).min, dtype=dtype)
    causal = paddle.triu(causal, diagonal=past_len + 1).unsqueeze([0, 1]).tile([bsz, 1, 1, 1])
    if mask is not None:
        causal = causal + _expand_mask(mask, dtype, tgt_len)
    return causal


class Florence2Encoder(nn.Layer):
    def __init__(self, config, embed_tokens):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.embed_positions = Florence2LearnedPositionalEmbedding(config.max_position_embeddings, config.d_model)
        self.layers = nn.LayerList([Florence2EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding, self.dropout = nn.LayerNorm(config.d_model), config.dropout

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        position_input = input_ids if input_ids is not None else x
        x = F.dropout(
            self.layernorm_embedding(x + self.embed_positions(position_input)), p=self.dropout, training=self.training
        )
        mask = _expand_mask(attention_mask, x.dtype) if attention_mask is not None else None
        states = (x,) if output_hidden_states else None
        attentions = () if output_attentions else None
        for layer in self.layers:
            x, attn = layer(x, mask, output_attentions)
            if output_hidden_states:
                states += (x,)
            if output_attentions:
                attentions += (attn,)
        return (
            BaseModelOutput(last_hidden_state=x, hidden_states=states, attentions=attentions)
            if return_dict
            else (x, states, attentions)
        )


class Florence2Decoder(nn.Layer):
    def __init__(self, config, embed_tokens):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.embed_positions = Florence2LearnedPositionalEmbedding(config.max_position_embeddings, config.d_model)
        self.layers = nn.LayerList([Florence2DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding, self.dropout = nn.LayerNorm(config.d_model), config.dropout

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        past_len = past_key_values[0][0].shape[2] if past_key_values else 0
        position_input = input_ids if input_ids is not None else x
        x = F.dropout(
            self.layernorm_embedding(x + self.embed_positions(position_input, past_len)),
            p=self.dropout,
            training=self.training,
        )
        mask = _causal_mask(attention_mask, x.shape[0], x.shape[1], past_len, x.dtype)
        enc_mask = (
            _expand_mask(encoder_attention_mask, x.dtype, x.shape[1]) if encoder_attention_mask is not None else None
        )
        states = (x,) if output_hidden_states else None
        self_attns = () if output_attentions else None
        cross_attns = () if output_attentions and encoder_hidden_states is not None else None
        cache = () if use_cache else None
        for i, layer in enumerate(self.layers):
            x, sa, ca, present = layer(
                x,
                mask,
                encoder_hidden_states,
                enc_mask,
                past_key_values[i] if past_key_values else None,
                use_cache,
                output_attentions,
            )
            if output_hidden_states:
                states += (x,)
            if output_attentions:
                self_attns += (sa,)
                cross_attns = cross_attns + (ca,) if cross_attns is not None else None
            if use_cache:
                cache += (present,)
        output = Seq2SeqModelOutput(
            last_hidden_state=x,
            past_key_values=cache,
            decoder_hidden_states=states,
            decoder_attentions=self_attns,
            cross_attentions=cross_attns,
        )
        return output if return_dict else (x, cache, states, self_attns, cross_attns)


class Florence2ScaledWordEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, embed_scale=1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids):
        return super().forward(input_ids) * self.embed_scale


class Florence2TiedLMHead(nn.Layer):
    def __init__(self, embed_tokens):
        super().__init__()
        self.embed_tokens = embed_tokens

    def forward(self, hidden_states):
        return paddle.matmul(hidden_states, self.embed_tokens.weight, transpose_y=True)


class Florence2Model(nn.Layer):
    def __init__(self, config):
        super().__init__()
        tc = config.text_config if isinstance(config, Florence2Config) else config
        self.shared = Florence2ScaledWordEmbedding(
            tc.vocab_size, tc.d_model, tc.pad_token_id, math.sqrt(tc.d_model) if tc.scale_embedding else 1.0
        )
        self.encoder, self.decoder = Florence2Encoder(tc, self.shared), Florence2Decoder(tc, self.shared)
        self.config = tc

    def get_input_embeddings(self):
        return self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        **kwargs
    ):
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no decoder_input_ids or decoder_inputs_embeds are passed, input_ids cannot be None."
                )
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids, attention_mask, inputs_embeds, output_attentions, output_hidden_states, True
            )
        elif not hasattr(encoder_outputs, "last_hidden_state"):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0])
        decoder_outputs = self.decoder(
            decoder_input_ids,
            decoder_attention_mask,
            encoder_outputs.last_hidden_state,
            attention_mask,
            past_key_values,
            decoder_inputs_embeds,
            use_cache,
            output_attentions,
            output_hidden_states,
            True,
        )
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.decoder_hidden_states,
            decoder_attentions=decoder_outputs.decoder_attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class Florence2LanguageForConditionalGeneration(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.model = Florence2Model(config)
        self.register_buffer("final_logits_bias", paddle.zeros([1, config.vocab_size]), persistable=True)
        self.lm_head = Florence2TiedLMHead(self.model.shared)
        self.config = config

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def forward(self, labels=None, return_dict=True, **kwargs):
        if labels is not None:
            kwargs["use_cache"] = False
            if kwargs.get("decoder_input_ids") is None and kwargs.get("decoder_inputs_embeds") is None:
                kwargs["decoder_input_ids"] = self.prepare_decoder_input_ids_from_labels(labels)
        outputs = self.model(return_dict=True, **kwargs)
        logits = self.lm_head(outputs.last_hidden_state) + self.final_logits_bias
        if labels is not None:
            flat_labels = labels.reshape([-1])
            token_loss = F.cross_entropy(
                logits.reshape([-1, logits.shape[-1]]),
                flat_labels,
                ignore_index=-100,
                reduction="none",
            )
            loss = token_loss[flat_labels != -100].mean()
        else:
            loss = None
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

    def prepare_decoder_input_ids_from_labels(self, labels):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return tuple(
            tuple(paddle.index_select(past_state, beam_idx, axis=0) for past_state in layer_past[:2]) + layer_past[2:]
            for layer_past in past_key_values
        )


class Florence2VisionModel(PretrainedModel):
    config_class = Florence2VisionConfig

    def __init__(self, config):
        super().__init__(config)
        self.vision_tower = DaViT(config)

    def forward(self, pixel_values):
        return self.vision_tower.forward_features_unpool(pixel_values)


class Florence2VisionModelWithProjection(Florence2VisionModel):
    def __init__(self, config):
        super().__init__(config)
        self._build_image_projection_layers(config)

    def _build_image_projection_layers(self, config):
        self.image_projection = self.create_parameter([config.dim_embed[-1], config.projection_dim])
        self.image_proj_norm = nn.LayerNorm(config.projection_dim)
        self.image_pos_embed = LearnedAbsolutePositionEmbedding2D(
            config.dim_embed[-1], config.image_pos_embed["max_pos_embeddings"]
        )
        self.visual_temporal_embed = PositionalEmbeddingCosine1D(
            config.dim_embed[-1], config.visual_temporal_embedding["max_temporal_embeddings"]
        )
        self.image_feature_source = config.image_feature_source

    def forward(self, pixel_values):
        x, size = self.vision_tower.forward_features_unpool(pixel_values, return_size=True)
        return self._project_features(x, size)

    def _project_features(self, x, size):
        b, _, c = x.shape
        h, w = size
        if h * w != x.shape[1]:
            raise ValueError("DaViT feature map size does not match its token sequence")
        x = x.reshape([b, h, w, c]) + self.image_pos_embed(x.reshape([b, h, w, c]))
        x = x.reshape([b, 1, h * w, c])
        x = x + self.visual_temporal_embed(x[:, :, 0]).reshape([1, 1, 1, c])
        features = {"spatial_avg_pool": x.mean(axis=2), "temporal_avg_pool": x.mean(axis=1), "last_frame": x[:, -1]}
        return self.image_proj_norm(
            paddle.matmul(
                paddle.concat([features[k] for k in self.image_feature_source], axis=1), self.image_projection
            )
        )


class Florence2ForConditionalGeneration(PretrainedModel):
    config_class = Florence2Config
    base_model_prefix = "florence2"
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2",
        "image_projection",
        "lm_head",
        "qkv",
        "proj",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.vision_tower = DaViT(config.vision_config)
        del self.vision_tower.norms
        del self.vision_tower.head
        self.image_projection = self.create_parameter(
            [config.vision_config.dim_embed[-1], config.vision_config.projection_dim]
        )
        self.image_proj_norm = nn.LayerNorm(config.vision_config.projection_dim)
        self.image_pos_embed = LearnedAbsolutePositionEmbedding2D(
            config.vision_config.dim_embed[-1], config.vision_config.image_pos_embed["max_pos_embeddings"]
        )
        self.visual_temporal_embed = PositionalEmbeddingCosine1D(
            config.vision_config.dim_embed[-1],
            config.vision_config.visual_temporal_embedding["max_temporal_embeddings"],
        )
        self.image_feature_source = config.vision_config.image_feature_source
        self.language_model = Florence2LanguageForConditionalGeneration(config.text_config)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _encode_image(self, pixel_values):
        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must have shape [batch_size, channels, height, width]")
        x, size = self.vision_tower.forward_features_unpool(pixel_values, return_size=True)
        b, _, c = x.shape
        h, w = size
        if h * w != x.shape[1]:
            raise ValueError("DaViT feature map size does not match its token sequence")
        x = x.reshape([b, h, w, c]) + self.image_pos_embed(x.reshape([b, h, w, c]))
        x = x.reshape([b, 1, h * w, c])
        x = x + self.visual_temporal_embed(x[:, :, 0]).reshape([1, 1, 1, c])
        features = {"spatial_avg_pool": x.mean(axis=2), "temporal_avg_pool": x.mean(axis=1), "last_frame": x[:, -1]}
        return self.image_proj_norm(
            paddle.matmul(
                paddle.concat([features[k] for k in self.image_feature_source], axis=1), self.image_projection
            )
        )

    def prepare_encoder_decoder_kwargs_for_generation(self, input_ids, model_kwargs):
        if "encoder_output" not in model_kwargs:
            pixel_values = model_kwargs.pop("pixel_values", None)
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                image_features = self._encode_image(pixel_values)
                inputs_embeds = paddle.concat([image_features, inputs_embeds], axis=1)
                text_mask = model_kwargs.get("attention_mask")
                image_mask = paddle.ones([image_features.shape[0], image_features.shape[1]], dtype="int64")
                if text_mask is None:
                    text_mask = paddle.ones(input_ids.shape, dtype="int64")
                model_kwargs["attention_mask"] = paddle.concat([image_mask.astype(text_mask.dtype), text_mask], axis=1)
            model_kwargs["encoder_output"] = self.get_encoder()(
                inputs_embeds=inputs_embeds,
                attention_mask=model_kwargs.get("attention_mask"),
            )
        return model_kwargs

    def forward(
        self, input_ids=None, pixel_values=None, attention_mask=None, inputs_embeds=None, return_dict=True, **kwargs
    ):
        image_features = None
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids) if input_ids is not None else None
            if pixel_values is not None:
                image_features = self._encode_image(pixel_values)
                inputs_embeds = (
                    image_features if inputs_embeds is None else paddle.concat([image_features, inputs_embeds], axis=1)
                )
                image_mask = paddle.ones(
                    [image_features.shape[0], image_features.shape[1]],
                    dtype=attention_mask.dtype if attention_mask is not None else "int64",
                )
                if attention_mask is None and input_ids is not None:
                    attention_mask = paddle.ones(input_ids.shape, dtype=image_mask.dtype)
                attention_mask = (
                    image_mask if attention_mask is None else paddle.concat([image_mask, attention_mask], axis=1)
                )
        if attention_mask is None and inputs_embeds is not None:
            attention_mask = paddle.ones(inputs_embeds.shape[:2], dtype="int64")
        output = self.language_model(
            attention_mask=attention_mask, inputs_embeds=inputs_embeds, return_dict=True, **kwargs
        )
        output.image_hidden_states = image_features
        return (
            output if return_dict else ((output.loss, output.logits) if output.loss is not None else (output.logits,))
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        pixel_values=None,
        decoder_attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        if encoder_outputs is None:
            encoder_outputs = kwargs.pop("encoder_output", None)
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            remove_prefix_length = (
                past_length if decoder_input_ids.shape[1] > past_length else decoder_input_ids.shape[1] - 1
            )
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]
        return {
            "input_ids": None,
            "decoder_input_ids": decoder_input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "decoder_attention_mask": decoder_attention_mask,
            "use_cache": use_cache,
            **kwargs,
        }

    def prepare_decoder_input_ids_from_labels(self, labels):
        return self.language_model.prepare_decoder_input_ids_from_labels(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        return self.language_model._reorder_cache(past_key_values, beam_idx)

    @classmethod
    def _gen_aoa_config(cls, config):
        statements = [
            "language_model.model.shared.weight -> language_model.model.encoder.embed_tokens.weight",
            "language_model.model.encoder.embed_tokens.weight -> language_model.model.decoder.embed_tokens.weight",
            "language_model.model.encoder.embed_tokens.weight -> language_model.lm_head.embed_tokens.weight",
            "language_model.model.encoder.embed_positions.weight -> language_model.model.encoder.embed_positions.weight",
            "language_model.model.decoder.embed_positions.weight -> language_model.model.decoder.embed_positions.weight",
            "image_pos_embed.row_embeddings.weight -> image_pos_embed.row_embeddings.weight",
            "image_pos_embed.column_embeddings.weight -> image_pos_embed.column_embeddings.weight",
            "visual_temporal_embed.pos_idx_to_embed -> visual_temporal_embed.pos_idx_to_embed",
            "image_projection -> image_projection",
        ]
        for stage, depth in enumerate(config.vision_config.depths):
            conv = f"vision_tower.convs.{stage}"
            statements.extend(
                [
                    f"{conv}.proj.weight -> {conv}.proj.weight",
                    f"{conv}.proj.bias -> {conv}.proj.bias",
                    f"{conv}.norm.weight -> {conv}.norm.weight",
                    f"{conv}.norm.bias -> {conv}.norm.bias",
                ]
            )
            for block in range(depth):
                base = f"vision_tower.blocks.{stage}.{block}"
                for name in [
                    "spatial_block.window_attn.fn.qkv",
                    "spatial_block.window_attn.fn.proj",
                    "spatial_block.ffn.fn.net.fc1",
                    "spatial_block.ffn.fn.net.fc2",
                    "channel_block.channel_attn.fn.qkv",
                    "channel_block.channel_attn.fn.proj",
                    "channel_block.ffn.fn.net.fc1",
                    "channel_block.ffn.fn.net.fc2",
                ]:
                    statements.extend(
                        [
                            f"{base}.{name}.weight^T -> {base}.{name}.weight",
                            f"{base}.{name}.bias -> {base}.{name}.bias",
                        ]
                    )
                for name in [
                    "spatial_block.conv1.fn.dw",
                    "spatial_block.conv2.fn.dw",
                    "channel_block.conv1.fn.dw",
                    "channel_block.conv2.fn.dw",
                ]:
                    statements.extend(
                        [f"{base}.{name}.weight -> {base}.{name}.weight", f"{base}.{name}.bias -> {base}.{name}.bias"]
                    )
                for name in [
                    "spatial_block.window_attn.norm",
                    "spatial_block.ffn.norm",
                    "channel_block.channel_attn.norm",
                    "channel_block.ffn.norm",
                ]:
                    statements.extend(
                        [f"{base}.{name}.weight -> {base}.{name}.weight", f"{base}.{name}.bias -> {base}.{name}.bias"]
                    )
        for side, layers in [
            ("encoder", config.text_config.encoder_layers),
            ("decoder", config.text_config.decoder_layers),
        ]:
            for i in range(layers):
                base = f"language_model.model.{side}.layers.{i}"
                names = [
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.q_proj",
                    "self_attn.out_proj",
                    "fc1",
                    "fc2",
                ]
                if side == "decoder":
                    names += [
                        "encoder_attn.k_proj",
                        "encoder_attn.v_proj",
                        "encoder_attn.q_proj",
                        "encoder_attn.out_proj",
                    ]
                for name in names:
                    statements.extend(
                        [
                            f"{base}.{name}.weight^T -> {base}.{name}.weight",
                            f"{base}.{name}.bias -> {base}.{name}.bias",
                        ]
                    )
                for name in ["self_attn_layer_norm", "final_layer_norm"] + (
                    ["encoder_attn_layer_norm"] if side == "decoder" else []
                ):
                    statements.extend(
                        [f"{base}.{name}.weight -> {base}.{name}.weight", f"{base}.{name}.bias -> {base}.{name}.bias"]
                    )
            prefix = f"language_model.model.{side}"
            statements.extend(
                [
                    f"{prefix}.layernorm_embedding.weight -> {prefix}.layernorm_embedding.weight",
                    f"{prefix}.layernorm_embedding.bias -> {prefix}.layernorm_embedding.bias",
                ]
            )
        statements.extend(
            [
                "image_proj_norm.weight -> image_proj_norm.weight",
                "image_proj_norm.bias -> image_proj_norm.bias",
                "language_model.final_logits_bias -> language_model.final_logits_bias",
            ]
        )
        return {"aoa_statements": statements}
