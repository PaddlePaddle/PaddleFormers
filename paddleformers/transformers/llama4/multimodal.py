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

import math
from dataclasses import dataclass

import paddle
import paddle.nn.functional as F
from paddle import nn

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ..model_outputs import BaseModelOutputWithPooling, CausalLMOutputWithPast
from ..model_utils import PretrainedModel
from .configuration import Llama4Config, Llama4VisionConfig
from .modeling import Llama4ForCausalLM


@dataclass
class Llama4CausalLMOutputWithPast(CausalLMOutputWithPast):
    loss: paddle.Tensor | None = None
    logits: paddle.Tensor | None = None
    past_key_values: object | None = None
    hidden_states: tuple[paddle.Tensor, ...] | None = None
    attentions: tuple[paddle.Tensor, ...] | None = None
    image_hidden_states: paddle.Tensor | None = None


def pixel_shuffle(input_tensor, shuffle_ratio):
    batch_size, num_patches, channels = input_tensor.shape
    patch_size = int(math.sqrt(num_patches))
    input_tensor = input_tensor.reshape([batch_size, patch_size, patch_size, channels])
    batch_size, height, width, channels = input_tensor.shape
    hidden_states = input_tensor.reshape(
        [batch_size, height, int(width * shuffle_ratio), int(channels / shuffle_ratio)]
    )
    hidden_states = hidden_states.transpose([0, 2, 1, 3])
    hidden_states = hidden_states.reshape(
        [
            batch_size,
            int(height * shuffle_ratio),
            int(width * shuffle_ratio),
            int(channels / (shuffle_ratio**2)),
        ]
    )
    hidden_states = hidden_states.transpose([0, 2, 1, 3])
    return hidden_states.reshape([batch_size, -1, hidden_states.shape[-1]])


class Llama4VisionMLP2(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.intermediate_size, config.projector_input_dim, bias_attr=False)
        self.fc2 = nn.Linear(config.projector_output_dim, config.projector_output_dim, bias_attr=False)
        self.dropout = config.projector_dropout

    def forward(self, hidden_states):
        hidden_states = F.gelu(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        return F.gelu(self.fc2(hidden_states))


class Llama4VisionPixelShuffleMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.pixel_shuffle_ratio = config.pixel_shuffle_ratio
        self.mlp = Llama4VisionMLP2(config)

    def forward(self, hidden_states):
        return self.mlp(pixel_shuffle(hidden_states, self.pixel_shuffle_ratio))


class Llama4MultiModalProjector(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.vision_config.vision_output_dim,
            config.text_config.hidden_size,
            bias_attr=False,
        )

    def forward(self, image_features):
        return self.linear_1(image_features)


def vision_apply_rotary_emb(query, key, freqs):
    query_float = query.astype("float32").reshape([*query.shape[:-1], -1, 2])
    key_float = key.astype("float32").reshape([*key.shape[:-1], -1, 2])
    cos = paddle.cos(freqs)[None, :, None, :]
    sin = paddle.sin(freqs)[None, :, None, :]

    query_out = paddle.stack(
        [query_float[..., 0] * cos - query_float[..., 1] * sin, query_float[..., 0] * sin + query_float[..., 1] * cos],
        axis=-1,
    ).flatten(-2)
    key_out = paddle.stack(
        [key_float[..., 0] * cos - key_float[..., 1] * sin, key_float[..., 0] * sin + key_float[..., 1] * cos],
        axis=-1,
    ).flatten(-2)
    return query_out.astype(query.dtype), key_out.astype(key.dtype)


class Llama4VisionAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = 1
        self.attention_dropout = config.attention_dropout
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias_attr=True)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias_attr=True)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias_attr=True)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias_attr=True)

    def forward(self, hidden_states, freqs_ci, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        target_shape = [batch_size, sequence_length, self.num_heads, self.head_dim]
        query = self.q_proj(hidden_states).reshape(target_shape)
        key = self.k_proj(hidden_states).reshape(target_shape)
        value = self.v_proj(hidden_states).reshape(target_shape)
        query, key = vision_apply_rotary_emb(query, key, freqs_ci)
        query = query.transpose([0, 2, 1, 3])
        key = key.transpose([0, 2, 1, 3])
        value = value.transpose([0, 2, 1, 3])
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attention_output, attention_weights = attention_interface(
            self,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            is_causal=False,
        )
        return self.o_proj(attention_output), attention_weights


class Llama4VisionMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias_attr=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias_attr=True)

    def forward(self, hidden_states):
        return self.fc2(F.gelu(self.fc1(hidden_states)))


class Llama4VisionEncoderLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Llama4VisionAttention(config)
        self.mlp = Llama4VisionMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.norm_eps)

    def forward(self, hidden_states, freqs_ci, attention_mask=None, output_attentions=False):
        residual = hidden_states
        hidden_states, attention_weights = self.self_attn(
            self.input_layernorm(hidden_states), freqs_ci=freqs_ci, attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return (hidden_states, attention_weights) if output_attentions else (hidden_states,)


class Llama4VisionEncoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.LayerList([Llama4VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        freqs_ci,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for layer in self.layers:
            if output_hidden_states:
                encoder_states += (hidden_states,)
            layer_outputs = layer(
                hidden_states,
                freqs_ci=freqs_ci,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions += (layer_outputs[1],)
        if output_hidden_states:
            encoder_states += (hidden_states,)
        if not return_dict:
            return tuple(x for x in (hidden_states, encoder_states, all_attentions) if x is not None)
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class Llama4UnfoldConvolution(nn.Layer):
    def __init__(self, config):
        super().__init__()
        kernel_size = (
            (config.patch_size, config.patch_size) if isinstance(config.patch_size, int) else config.patch_size
        )
        self.unfold = nn.Unfold(kernel_sizes=kernel_size, strides=kernel_size)
        self.linear = nn.Linear(
            config.num_channels * kernel_size[0] * kernel_size[1], config.hidden_size, bias_attr=False
        )

    def forward(self, hidden_states):
        hidden_states = self.unfold(hidden_states).transpose([0, 2, 1])
        return self.linear(hidden_states)


class Llama4VisionRotaryEmbedding(nn.Layer):
    def __init__(self, config):
        super().__init__()
        grid_size = config.image_size // config.patch_size
        image_index = paddle.arange(grid_size**2, dtype="int32").reshape([-1, 1])
        image_index = paddle.concat([image_index, image_index[:1]], axis=0)
        image_index[-1, -1] = -2
        frequency_x = (image_index % grid_size).astype("float32")
        frequency_y = (image_index // grid_size).astype("float32")
        frequency_dim = config.hidden_size // config.num_attention_heads // 2
        rope_frequency = 1.0 / (
            config.rope_theta
            ** (paddle.arange(0, frequency_dim, 2, dtype="float32")[: frequency_dim // 2] / frequency_dim)
        )
        freqs_x = paddle.repeat_interleave((frequency_x + 1)[..., None] * rope_frequency, 2, axis=-1)
        freqs_y = paddle.repeat_interleave((frequency_y + 1)[..., None] * rope_frequency, 2, axis=-1)
        freqs = paddle.concat([freqs_x, freqs_y], axis=-1).astype("float32")[..., ::2]
        freqs = paddle.where(image_index.reshape([-1, 1, 1]) < 0, paddle.zeros_like(freqs), freqs)
        self.register_buffer("freqs", freqs.squeeze(1), persistable=False)

    def forward(self, hidden_states):
        return self.freqs


class Llama4VisionModel(PretrainedModel):
    config_class = Llama4VisionConfig
    base_model_prefix = "vision_model"
    transpose_weight_keys = ["linear", "q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_patches = (config.image_size // config.patch_size) ** 2 + 1
        scale = config.hidden_size**-0.5
        self.patch_embedding = Llama4UnfoldConvolution(config)
        self.class_embedding = self.create_parameter(
            [config.hidden_size], default_initializer=nn.initializer.Normal(std=scale)
        )
        self.positional_embedding_vlm = self.create_parameter(
            [self.num_patches, config.hidden_size], default_initializer=nn.initializer.Normal(std=scale)
        )
        self.rotary_embedding = Llama4VisionRotaryEmbedding(config)
        self.layernorm_pre = nn.LayerNorm(config.hidden_size, epsilon=config.norm_eps)
        self.layernorm_post = nn.LayerNorm(config.hidden_size, epsilon=config.norm_eps)
        self.model = Llama4VisionEncoder(config)
        self.vision_adapter = Llama4VisionPixelShuffleMLP(config)

    def forward(
        self,
        pixel_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        hidden_states = self.patch_embedding(pixel_values)
        class_embedding = self.class_embedding.reshape([1, 1, -1]).expand([hidden_states.shape[0], 1, -1])
        hidden_states = paddle.concat([hidden_states, class_embedding], axis=1)
        hidden_states = self.layernorm_pre(hidden_states + self.positional_embedding_vlm.astype(hidden_states.dtype))
        outputs = self.model(
            hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            freqs_ci=self.rotary_embedding(pixel_values),
            return_dict=True,
        )
        hidden_states = self.layernorm_post(outputs.last_hidden_state)[:, :-1, :]
        hidden_states = self.vision_adapter(hidden_states)
        if not return_dict:
            return tuple(x for x in (hidden_states, outputs.hidden_states, outputs.attentions) if x is not None)
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Llama4ForConditionalGeneration(PretrainedModel):
    config_class = Llama4Config
    base_model_prefix = "model"
    transpose_weight_keys = [
        "linear",
        "linear_1",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "fc1",
        "fc2",
        "gate_proj",
        "up_proj",
        "down_proj",
        "router.linear",
    ]

    @classmethod
    def _gen_aoa_config(cls, config):
        """Map the nested Transformers Llama4 checkpoint into Paddle layers."""
        text_statements = Llama4ForCausalLM._gen_aoa_config(config.text_config)["aoa_statements"]
        statements = []
        for statement in text_statements:
            source, destination = statement.split(" -> ")
            statements.append(f"language_model.{source} -> language_model.{destination}")

        statements.extend(
            [
                f"vision_model.{name} -> vision_model.{name}"
                for name in [
                    "class_embedding",
                    "positional_embedding_vlm",
                    "layernorm_pre.weight",
                    "layernorm_pre.bias",
                    "layernorm_post.weight",
                    "layernorm_post.bias",
                ]
            ]
        )
        statements.extend(
            [
                "vision_model.patch_embedding.linear.weight^T -> vision_model.patch_embedding.linear.weight",
                "vision_model.vision_adapter.mlp.fc1.weight^T -> vision_model.vision_adapter.mlp.fc1.weight",
                "vision_model.vision_adapter.mlp.fc2.weight^T -> vision_model.vision_adapter.mlp.fc2.weight",
                "multi_modal_projector.linear_1.weight^T -> multi_modal_projector.linear_1.weight",
            ]
        )
        for layer_id in range(config.vision_config.num_hidden_layers):
            prefix = f"vision_model.model.layers.{layer_id}"
            for norm_name in ["input_layernorm", "post_attention_layernorm"]:
                for parameter in ["weight", "bias"]:
                    name = f"{prefix}.{norm_name}.{parameter}"
                    statements.append(f"{name} -> {name}")
            for projection in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                weight = f"{prefix}.self_attn.{projection}.weight"
                bias = f"{prefix}.self_attn.{projection}.bias"
                statements.extend([f"{weight}^T -> {weight}", f"{bias} -> {bias}"])
            for projection in ["fc1", "fc2"]:
                weight = f"{prefix}.mlp.{projection}.weight"
                bias = f"{prefix}.mlp.{projection}.bias"
                statements.extend([f"{weight}^T -> {weight}", f"{bias} -> {bias}"])
        return {"aoa_statements": statements}

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vision_model = Llama4VisionModel(config.vision_config)
        self.multi_modal_projector = Llama4MultiModalProjector(config)
        self.language_model = Llama4ForCausalLM(config.text_config)
        self.vocab_size = config.text_config.vocab_size

    def get_input_embeddings(self):
        return self.language_model.model.embed_tokens

    def get_output_embeddings(self):
        return self.language_model.lm_head

    def get_image_features(self, pixel_values, **kwargs):
        return self.vision_model(pixel_values, **{key: value for key, value in kwargs.items() if value is not None})

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        pixel_values=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            inputs_embeds = None
            # Image embeddings are already represented in the KV cache after
            # prefill. Re-encoding them would also leave no image placeholder
            # in the one-token decode input.
            pixel_values = None
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "use_cache": kwargs.get("use_cache", True),
        }
        if inputs_embeds is not None and past_key_values is None:
            model_inputs["inputs_embeds"] = inputs_embeds
            model_inputs["input_ids"] = None
        return model_inputs

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        vision_feature_select_strategy=None,
        labels=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError("pixel_values and inputs_embeds cannot be specified together")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_hidden_states = None
        if pixel_values is not None:
            image_hidden_states = self.get_image_features(pixel_values, return_dict=True).last_hidden_state
            image_features = self.multi_modal_projector(
                image_hidden_states.reshape([-1, image_hidden_states.shape[-1]])
            ).astype(inputs_embeds.dtype)
            image_mask = input_ids == self.config.image_token_index
            if int(image_mask.astype("int64").sum()) != image_features.shape[0]:
                raise ValueError(
                    f"Image features and image tokens do not match: {image_features.shape[0]} != "
                    f"{int(image_mask.astype('int64').sum())}"
                )
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[image_mask] = image_features

        outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        if not return_dict:
            values = (outputs.logits, outputs.past_key_values, outputs.hidden_states)
            values = tuple(value for value in values if value is not None)
            return (outputs.loss,) + values if outputs.loss is not None else values
        return Llama4CausalLMOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            image_hidden_states=image_hidden_states,
        )


__all__ = [
    "Llama4ForConditionalGeneration",
    "Llama4VisionEncoderLayer",
    "Llama4VisionModel",
]
