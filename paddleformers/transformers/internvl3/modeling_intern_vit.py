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

from typing import Optional, Tuple, Union

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet.utils import recompute
from ..activations import ACT2FN
from ..model_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ..model_utils import PretrainedModel
from .configuration_intern_vit import InternVisionConfig


class DropPath(nn.Layer):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return hidden_states
        keep_prob = 1.0 - self.drop_prob
        shape = [hidden_states.shape[0]] + [1] * (hidden_states.ndim - 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=hidden_states.dtype)
        random_tensor = paddle.floor(random_tensor)
        return hidden_states / keep_prob * random_tensor


class InternRMSNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[hidden_size],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype("float32")
        variance = hidden_states.pow(2).mean(axis=-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.astype(input_dtype)


NORM2FN = {
    "rms_norm": InternRMSNorm,
    "layer_norm": nn.LayerNorm,
}


def build_norm(norm_type: str, hidden_size: int, eps: float):
    if norm_type == "rms_norm":
        return InternRMSNorm(hidden_size, eps=eps)
    return nn.LayerNorm(hidden_size, epsilon=eps)


class InternVisionEmbeddings(nn.Layer):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = self.create_parameter(
            shape=[1, 1, self.embed_dim],
            default_initializer=nn.initializer.Normal(std=1.0),
        )
        self.patch_embedding = nn.Conv2D(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = self.create_parameter(
            shape=[1, self.num_positions, self.embed_dim],
            default_initializer=nn.initializer.Normal(std=1.0),
        )

    def _get_pos_embed(self, pos_embed, height, width):
        target_dtype = pos_embed.dtype
        pos_embed = paddle.reshape(
            pos_embed.astype("float32"),
            [1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1],
        )
        pos_embed = pos_embed.transpose([0, 3, 1, 2])
        pos_embed = F.interpolate(pos_embed, size=(height, width), mode="bicubic", align_corners=False)
        pos_embed = pos_embed.reshape([1, -1, height * width]).transpose([0, 2, 1]).astype(target_dtype)
        return pos_embed

    def forward(self, pixel_values: paddle.Tensor) -> paddle.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.astype(target_dtype))
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose([0, 2, 1])
        class_embeds = paddle.expand(self.class_embedding.astype(target_dtype), [batch_size, 1, self.embed_dim])
        embeddings = paddle.concat([class_embeds, patch_embeds], axis=1)
        position_embedding = paddle.concat(
            [
                self.position_embedding[:, :1, :],
                self._get_pos_embed(self.position_embedding[:, 1:, :], height, width),
            ],
            axis=1,
        )
        return embeddings + position_embedding.astype(target_dtype)


class InternAttention(nn.Layer):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got {self.embed_dim} and {self.num_heads})."
            )

        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias_attr=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj_drop = nn.Dropout(config.dropout)
        self.qk_normalization = config.qk_normalization
        if self.qk_normalization:
            self.q_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape([batch_size, seq_len, 3, self.num_heads, hidden_size // self.num_heads]).transpose(
            [2, 0, 3, 1, 4]
        )
        query, key, value = qkv[0], qkv[1], qkv[2]

        if self.qk_normalization:
            bsz, nheads, qlen, dim = query.shape
            query = self.q_norm(query.transpose([0, 2, 1, 3]).reshape([bsz, qlen, nheads * dim]))
            key = self.k_norm(key.transpose([0, 2, 1, 3]).reshape([bsz, qlen, nheads * dim]))
            query = query.reshape([bsz, qlen, nheads, dim]).transpose([0, 2, 1, 3])
            key = key.reshape([bsz, qlen, nheads, dim]).transpose([0, 2, 1, 3])

        attn = paddle.matmul(query * self.scale, key.transpose([0, 1, 3, 2]))
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        hidden_states = paddle.matmul(attn, value).transpose([0, 2, 1, 3]).reshape([batch_size, seq_len, hidden_size])
        hidden_states = self.proj(hidden_states)
        hidden_states = self.proj_drop(hidden_states)
        return hidden_states


class InternMLP(nn.Layer):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.act = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class InternVisionEncoderLayer(nn.Layer):
    def __init__(self, config: InternVisionConfig, drop_path_rate: float):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.norm_type = config.norm_type
        self.attn = InternAttention(config)
        self.mlp = InternMLP(config)
        self.norm1 = build_norm(self.norm_type, self.embed_dim, config.layer_norm_eps)
        self.norm2 = build_norm(self.norm_type, self.embed_dim, config.layer_norm_eps)
        self.ls1 = self.create_parameter(
            shape=[self.embed_dim],
            default_initializer=nn.initializer.Constant(config.initializer_factor),
        )
        self.ls2 = self.create_parameter(
            shape=[self.embed_dim],
            default_initializer=nn.initializer.Constant(config.initializer_factor),
        )
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = hidden_states + self.drop_path1(self.attn(self.norm1(hidden_states).astype(hidden_states.dtype)) * self.ls1)
        hidden_states = hidden_states + self.drop_path2(self.mlp(self.norm2(hidden_states).astype(hidden_states.dtype)) * self.ls2)
        return hidden_states


class InternVisionEncoder(nn.Layer):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        dpr = paddle.linspace(0, config.drop_path_rate, config.num_hidden_layers, dtype="float32").numpy().tolist()
        self.layers = nn.LayerList(
            [InternVisionEncoderLayer(config, float(dpr[idx])) for idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False
        self.config = config

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
    ) -> paddle.Tensor:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
        )
        return hidden_states

    def forward(
        self,
        inputs_embeds,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds
        recompute_num_layers = max(int(getattr(self.config, "recompute_num_layers", 1) or 1), 1)

        for layer_num, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            has_gradient = not hidden_states.stop_gradient
            if (
                self.config.recompute_granularity == "full"
                and self.config.recompute_method == "uniform"
                and has_gradient
                and layer_num % recompute_num_layers == 0
            ):
                hidden_states = self.recompute_training_full(
                    encoder_layer,
                    hidden_states,
                )
            else:
                hidden_states = encoder_layer(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states)


class InternVisionModel(PretrainedModel):
    main_input_name = "pixel_values"
    config_class = InternVisionConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["InternVisionEncoderLayer"]

    def __init__(self, config: InternVisionConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config)


    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        pos_emb = self.embeddings.position_embedding
        _, _, embed_dim = pos_emb.shape
        cls_emb = pos_emb[:, :1, :]
        pos_emb = pos_emb[:, 1:, :].reshape([1, old_size // patch_size, old_size // patch_size, -1]).transpose([0, 3, 1, 2])
        pos_emb = F.interpolate(
            pos_emb.astype("float32"),
            size=(new_size // patch_size, new_size // patch_size),
            mode="bicubic",
            align_corners=False,
        )
        pos_emb = pos_emb.astype(cls_emb.dtype).reshape([1, embed_dim, -1]).transpose([0, 2, 1])
        pos_emb = paddle.concat([cls_emb, pos_emb], axis=1)
        self.embeddings.position_embedding = self.create_parameter(
            shape=pos_emb.shape,
            default_initializer=nn.initializer.Assign(pos_emb),
        )
        self.embeddings.image_size = new_size

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
        self,
        pixel_values: Optional[paddle.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_embeds: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and pixel_embeds is None:
            raise ValueError("You have to specify pixel_values or pixel_embeds")

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            if len(pixel_values.shape) != 4:
                raise ValueError(f"wrong pixel_values size: {pixel_values.shape}")
            hidden_states = self.embeddings(pixel_values)

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


__all__ = ["InternVisionModel"]
