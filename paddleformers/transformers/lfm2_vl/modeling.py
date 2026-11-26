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

"""Paddle implementation of LFM2-VL and its SigLIP2 NaFlex vision tower."""

from copy import deepcopy
from dataclasses import dataclass

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet.utils import recompute

from ...nn.criterion.interface import CriterionLayer
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead
from ..activations import ACT2FN
from ..lfm2.modeling import Lfm2Model, Lfm2PreTrainedModel
from ..model_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from ..model_utils import PretrainedModel, register_base_model
from ..siglip_vision_model.modeling import (
    SiglipAttention,
    SiglipEncoderLayer,
    SiglipMLP,
)
from .configuration import Lfm2VlConfig


@dataclass
class Lfm2VlModelOutputWithPast(BaseModelOutputWithPast):
    image_hidden_states: paddle.Tensor | None = None


@dataclass
class Lfm2VlCausalLMOutputWithPast(CausalLMOutputWithPast):
    image_hidden_states: paddle.Tensor | None = None


class Siglip2VisionEmbeddings(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.patch_embedding = GeneralLinear.create(
            config.num_channels * config.patch_size * config.patch_size,
            config.hidden_size,
            has_bias=True,
            linear_type="default",
        )
        self.position_embedding_size = int(config.num_patches**0.5)
        self.position_embedding = nn.Embedding(config.num_patches, config.hidden_size)

    def forward(self, pixel_values, spatial_shapes):
        patch_embeds = self.patch_embedding(pixel_values.astype(self.patch_embedding.weight.dtype))
        position_embeddings = self.position_embedding.weight.reshape(
            [self.position_embedding_size, self.position_embedding_size, -1]
        )
        position_embeddings = position_embeddings.transpose([2, 0, 1]).unsqueeze(0)
        resized_batch = []
        max_length = pixel_values.shape[1]
        for batch_index in range(spatial_shapes.shape[0]):
            height, width = [int(item) for item in spatial_shapes[batch_index].tolist()]
            resized = F.interpolate(
                position_embeddings.astype("float32"),
                size=[height, width],
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            resized = resized.reshape([position_embeddings.shape[1], height * width]).transpose([1, 0])
            resized = resized.astype(patch_embeds.dtype)
            if height * width < max_length:
                resized = paddle.concat([resized, resized[0:1].expand([max_length - height * width, -1])], axis=0)
            resized_batch.append(resized)
        return patch_embeds + paddle.stack(resized_batch)


class Siglip2Attention(SiglipAttention):
    def __init__(self, config):
        config = deepcopy(config)
        if config._attn_implementation == "flashmask":
            # Shared SigLIP's flashmask indices describe a full square, but NaFlex
            # batches have padded patches. Keep their additive padding mask with SDPA.
            config._attn_implementation = "sdpa"
        super().__init__(config)
        self.q_proj = GeneralLinear.create(
            config.hidden_size, config.hidden_size, has_bias=True, linear_type="default"
        )
        self.k_proj = GeneralLinear.create(
            config.hidden_size, config.hidden_size, has_bias=True, linear_type="default"
        )
        self.v_proj = GeneralLinear.create(
            config.hidden_size, config.hidden_size, has_bias=True, linear_type="default"
        )
        self.out_proj = GeneralLinear.create(
            config.hidden_size, config.hidden_size, has_bias=True, linear_type="default"
        )

    def _get_bidirectional_indices(self, batch_size, seq_len, dtype):
        # On GPU, shared SDPA prioritizes row indices over attention_mask.
        # A full square would silently unmask padded NaFlex patches.
        # Keep an assignment so dy2static does not emit an empty function body
        # when removing a literal ``return None``.
        indices = None
        return indices


class Siglip2MLP(SiglipMLP):
    def __init__(self, config):
        super().__init__(config)
        self.fc1 = GeneralLinear.create(
            config.hidden_size, config.intermediate_size, has_bias=True, linear_type="default"
        )
        self.fc2 = GeneralLinear.create(
            config.intermediate_size, config.hidden_size, has_bias=True, linear_type="default"
        )


class Siglip2EncoderLayer(SiglipEncoderLayer):
    def __init__(self, config):
        super().__init__(config)
        self.self_attn = Siglip2Attention(config)
        self.mlp = Siglip2MLP(config)


class Siglip2Encoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.LayerList([Siglip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        def create_custom_forward(module):
            def custom_forward(layer_hidden_states):
                return module(layer_hidden_states, attention_mask)[0]

            return custom_forward

        for layer in self.layers:
            if (
                self.training
                and not hidden_states.stop_gradient
                and getattr(self.config, "recompute_granularity", None) == "full"
            ):
                hidden_states = recompute(
                    create_custom_forward(layer),
                    hidden_states,
                    use_reentrant=getattr(self.config, "recompute_use_reentrant", True),
                )
            else:
                hidden_states = layer(hidden_states, attention_mask)[0]
        return hidden_states


class Siglip2VisionTransformer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = Siglip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def forward(self, pixel_values, pixel_attention_mask, spatial_shapes, return_dict=True, **kwargs):
        hidden_states = self.embeddings(pixel_values, spatial_shapes)
        attention_mask = None
        if pixel_attention_mask is not None:
            minimum = paddle.finfo(hidden_states.dtype).min
            attention_mask = (1 - pixel_attention_mask.astype(hidden_states.dtype))[:, None, None, :] * minimum
        hidden_states = self.encoder(hidden_states, attention_mask)
        hidden_states = self.post_layernorm(hidden_states)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=None)


class Siglip2VisionModel(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.vision_model = Siglip2VisionTransformer(config)

    def forward(self, *args, **kwargs):
        return self.vision_model(*args, **kwargs)


class Lfm2VlMultiModalProjector(nn.Layer):
    def __init__(self, config):
        super().__init__()
        in_features = config.vision_config.hidden_size * config.downsample_factor**2
        self.factor = config.downsample_factor
        self.layer_norm = nn.LayerNorm(in_features) if config.projector_use_layernorm else None
        self.linear_1 = GeneralLinear.create(
            in_features, config.projector_hidden_size, has_bias=config.projector_bias, linear_type="default"
        )
        self.linear_2 = GeneralLinear.create(
            config.projector_hidden_size,
            config.text_config.hidden_size,
            has_bias=config.projector_bias,
            linear_type="default",
        )
        self.activation = ACT2FN[config.projector_hidden_act]

    def forward(self, hidden_states):
        batch_size, height, width, channels = hidden_states.shape
        factor = self.factor
        hidden_states = hidden_states.reshape([batch_size, height, width // factor, channels * factor])
        hidden_states = hidden_states.transpose([0, 2, 1, 3])
        hidden_states = hidden_states.reshape([batch_size, width // factor, height // factor, channels * factor**2])
        hidden_states = hidden_states.transpose([0, 2, 1, 3])
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
        return self.linear_2(self.activation(self.linear_1(hidden_states)))


class Lfm2VlPreTrainedModel(PretrainedModel):
    config_class = Lfm2VlConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    @classmethod
    def _checkpoint_weights(cls, config):
        weights = [
            ("language_model." + name, transpose)
            for name, transpose in Lfm2PreTrainedModel._checkpoint_weights(config.text_config)
        ]
        vision = "vision_tower.vision_model."
        weights += [
            (vision + "embeddings.patch_embedding.weight", True),
            (vision + "embeddings.patch_embedding.bias", False),
            (vision + "embeddings.position_embedding.weight", False),
            (vision + "post_layernorm.weight", False),
            (vision + "post_layernorm.bias", False),
        ]
        for index in range(config.vision_config.num_hidden_layers):
            prefix = vision + f"encoder.layers.{index}."
            for name in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.out_proj",
                "mlp.fc1",
                "mlp.fc2",
                "layer_norm1",
                "layer_norm2",
            ):
                weights.append((prefix + name + ".weight", not name.startswith("layer_norm")))
                weights.append((prefix + name + ".bias", False))
        for name in ("linear_1", "linear_2"):
            weights.append(("multi_modal_projector." + name + ".weight", True))
            if config.projector_bias:
                weights.append(("multi_modal_projector." + name + ".bias", False))
        if config.projector_use_layernorm:
            weights.extend(("multi_modal_projector.layer_norm." + name, False) for name in ("weight", "bias"))
        return weights

    @classmethod
    def _gen_aoa_config(cls, config):
        prefix = "" if cls == cls.base_model_class else "model."
        statements = Lfm2PreTrainedModel._weight_statements(cls._checkpoint_weights(config), prefix)
        if prefix:
            source = "model.language_model.embed_tokens.weight" if config.tie_word_embeddings else "lm_head.weight"
            statements.append(f"{source} -> lm_head.weight")
        return {"aoa_statements": statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config):
        prefix = "" if cls == cls.base_model_class else "model."
        statements = Lfm2PreTrainedModel._weight_statements(cls._checkpoint_weights(config), prefix, inverse=True)
        if prefix:
            target = "_" if config.tie_word_embeddings else "lm_head.weight"
            statements.append(f"lm_head.weight -> {target}")
        return {"aoa_statements": statements}


@register_base_model
class Lfm2VlModel(Lfm2VlPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.vision_tower = Siglip2VisionModel(config.vision_config)
        self.multi_modal_projector = Lfm2VlMultiModalProjector(config)
        self.language_model = Lfm2Model(config.text_config)

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def get_image_features(self, pixel_values, spatial_shapes, pixel_attention_mask):
        outputs = self.vision_tower(pixel_values, pixel_attention_mask, spatial_shapes)
        feature_lengths = pixel_attention_mask.astype("int64").sum(axis=1)
        image_features = []
        for image_index in range(outputs.last_hidden_state.shape[0]):
            feature_length = int(feature_lengths[image_index])
            height, width = [int(item) for item in spatial_shapes[image_index].tolist()]
            features = outputs.last_hidden_state[image_index, :feature_length]
            features = features.reshape([1, height, width, -1])
            features = self.multi_modal_projector(features).reshape([-1, self.config.text_config.hidden_size])
            image_features.append(features)
        outputs.pooler_output = image_features
        return outputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        pixel_values=None,
        spatial_shapes=None,
        pixel_attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        image_features = None
        if pixel_values is not None:
            if input_ids is None:
                raise ValueError("input_ids must be provided with pixel_values to locate LFM2-VL image tokens")
            image_features_list = self.get_image_features(
                pixel_values, spatial_shapes, pixel_attention_mask
            ).pooler_output
            image_features = paddle.concat(image_features_list, axis=0).astype(inputs_embeds.dtype)
            image_mask = input_ids == self.config.image_token_id
            if int(image_mask.astype("int64").sum()) != image_features.shape[0]:
                raise ValueError(
                    f"Image tokens and features do not match: {int(image_mask.sum())} versus {image_features.shape[0]}"
                )
            flat_embeddings = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
            flat_mask = image_mask.flatten()
            flat_embeddings[flat_mask] = image_features
            inputs_embeds = flat_embeddings.reshape(inputs_embeds.shape)
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return Lfm2VlModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            image_hidden_states=image_features,
        )


class Lfm2VlForConditionalGeneration(Lfm2VlPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = Lfm2VlModel(config)
        self.lm_head = LMHead(config.text_config)
        self.criterion = CriterionLayer(config.text_config)
        self.tie_weights()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, **kwargs)
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            for key in ("pixel_values", "pixel_attention_mask", "spatial_shapes"):
                inputs.pop(key, None)
        return inputs

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, labels=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, **kwargs)
        logits = self.lm_head(outputs.last_hidden_state)
        loss = self.criterion(logits, labels)[0] if labels is not None else None
        return Lfm2VlCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            image_hidden_states=outputs.image_hidden_states,
        )


__all__ = [
    "Lfm2VlForConditionalGeneration",
    "Lfm2VlModel",
    "Lfm2VlPreTrainedModel",
    "Siglip2VisionModel",
]
