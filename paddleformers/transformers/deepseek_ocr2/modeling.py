# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""Native Transformers DeepSeek-OCR-2: SAM, hybrid Qwen2 vision and MHA/MoE text."""

from dataclasses import dataclass
from functools import partial

import paddle
import paddle.nn.functional as F
from paddle import nn

from ...nn.activation import ACT2FN
from ...nn.criterion.interface import CriterionLayer
from ...nn.linear import Linear
from ...nn.lm_head import LMHead
from ...nn.mlp import MLP
from ..llama.modeling import LlamaModel
from ..model_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from ..model_utils import PretrainedModel, register_base_model
from ..qwen2.modeling import Qwen2Model
from .configuration_native import DeepseekOcr2Config
from .modeling_legacy import (
    Attention,
    DeepseekOCR2ForCausalLM,
    DeepseekOCR2ForConditionalGeneration,
    DeepseekOCR2Model,
    ImageEncoderViT,
    _parse_line_result,
    dynamic_preprocess,
    extract_coordinates_and_label,
    load_image,
)


class DeepseekOcr2SamLayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.transpose([0, 2, 3, 1])).transpose([0, 3, 1, 2])


class DeepseekOcr2SamVisionEncoder(ImageEncoderViT):
    """Reuse the SAM blocks; retain native normalization and interpolation semantics."""

    def __init__(self, config):
        super().__init__(
            img_size=config.image_size,
            patch_size=config.patch_size,
            in_chans=config.num_channels,
            embed_dim=config.hidden_size,
            depth=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            mlp_ratio=config.mlp_dim / config.hidden_size,
            out_chans=config.output_channels,
            hidden_dimension=config.downsample_channels[-1],
            qkv_bias=config.qkv_bias,
            norm_layer=partial(nn.LayerNorm, epsilon=config.layer_norm_eps),
            act_layer=lambda: ACT2FN[config.hidden_act],
            use_abs_pos=config.use_abs_pos,
            use_rel_pos=config.use_rel_pos,
            window_size=config.window_size,
            global_attn_indexes=config.global_attn_indexes,
            attn_implementation=config._attn_implementation,
        )
        self.neck[1] = DeepseekOcr2SamLayerNorm(config.output_channels, epsilon=1e-6)
        self.neck[3] = DeepseekOcr2SamLayerNorm(config.output_channels, epsilon=1e-6)
        self.net_2 = nn.Conv2D(
            config.output_channels, config.downsample_channels[0], 3, stride=2, padding=1, bias_attr=False
        )
        self.net_3 = nn.Conv2D(
            config.downsample_channels[0], config.downsample_channels[1], 3, stride=2, padding=1, bias_attr=False
        )
        if config.attention_dropout:
            raise ValueError("Nonzero SAM attention dropout is not supported by the shared SAM blocks")

    def forward(self, pixel_values):
        x = self.patch_embed(pixel_values)
        if self.pos_embed is not None:
            pos = self.pos_embed
            if list(pos.shape[1:3]) != list(x.shape[1:3]):
                pos = F.interpolate(
                    pos.transpose([0, 3, 1, 2]).astype("float32"),
                    size=x.shape[1:3],
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
                pos = pos.transpose([0, 2, 3, 1]).astype(x.dtype)
            x = x + pos
        for block in self.blocks:
            x = block(x)
        return self.net_3(self.net_2(self.neck(x.transpose([0, 3, 1, 2]))))


class DeepseekOcr2VisionEncoder(Qwen2Model):
    def __init__(self, config):
        super().__init__(config)
        del self.embed_tokens

    def forward(self, inputs_embeds, num_patches, **kwargs):
        length = inputs_embeds.shape[1]
        row = paddle.arange(length).unsqueeze(-1)
        col = paddle.arange(length).unsqueeze(0)
        # Image prefix is bidirectional. Queries see all image tokens and preceding queries.
        allowed = (col <= row) | ((row < num_patches) & (col < num_patches))
        mask = allowed.unsqueeze(0).unsqueeze(0).expand([inputs_embeds.shape[0], 1, length, length])
        positions = paddle.arange(length, dtype="int64").unsqueeze(0).expand([inputs_embeds.shape[0], -1])
        return super().forward(
            inputs_embeds=inputs_embeds, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True
        )


class DeepseekOcr2VisionModel(nn.Layer):
    def __init__(self, config):
        super().__init__()
        if config.sam_config.downsample_channels[-1] != config.encoder_config.hidden_size:
            raise ValueError("SAM output channels must equal the vision encoder hidden size")
        self.sam_encoder = DeepseekOcr2SamVisionEncoder(config.sam_config)
        self.vision_encoder = DeepseekOcr2VisionEncoder(config.encoder_config)
        self.query_768_resolution = nn.Embedding(144, config.encoder_config.hidden_size)
        self.query_1024_resolution = nn.Embedding(256, config.encoder_config.hidden_size)

    def forward(self, pixel_values):
        hidden = self.sam_encoder(pixel_values).flatten(2).transpose([0, 2, 1])
        batch, patches, _ = hidden.shape
        query = self.query_768_resolution.weight if patches <= 144 else self.query_1024_resolution.weight
        inputs = paddle.concat([hidden, query.unsqueeze(0).expand([batch, -1, -1])], axis=1)
        return self.vision_encoder(inputs, patches).last_hidden_state[:, patches:]


class DeepseekOcr2TextTopkRouter(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = self.create_parameter([config.n_routed_experts, config.hidden_size])

    def forward(self, hidden):
        config = self.config
        scores = F.softmax(
            paddle.matmul(hidden.astype("float32"), self.weight.astype("float32"), transpose_y=True), axis=-1
        )
        if config.topk_method == "group_limited_greedy":
            groups = scores.reshape([-1, config.n_group, config.n_routed_experts // config.n_group]).max(axis=-1)
            selected = paddle.topk(groups, config.topk_group, axis=-1, sorted=False)[1]
            mask = F.one_hot(selected, config.n_group).sum(axis=-2).astype("bool")
            mask = mask.unsqueeze(-1).expand([-1, -1, config.n_routed_experts // config.n_group]).reshape(scores.shape)
            scores = paddle.where(mask, scores, paddle.zeros_like(scores))
        weights, indices = paddle.topk(scores, config.num_experts_per_tok, axis=-1, sorted=False)
        return weights * config.routed_scaling_factor, indices


class DeepseekOcr2TextExperts(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = self.create_parameter(
            [config.n_routed_experts, 2 * config.moe_intermediate_size, config.hidden_size]
        )
        self.down_proj = self.create_parameter(
            [config.n_routed_experts, config.hidden_size, config.moe_intermediate_size]
        )
        self.act = ACT2FN[config.hidden_act]
        self.num_experts = config.n_routed_experts

    def forward(self, hidden, weights, indices):
        result = paddle.zeros_like(hidden)
        for expert in range(self.num_experts):
            locations = paddle.nonzero(indices == expert)
            if locations.shape[0] == 0:
                continue
            tokens = locations[:, 0]
            selected = paddle.gather(hidden, tokens)
            gate, up = paddle.matmul(selected, self.gate_up_proj[expert], transpose_y=True).chunk(2, axis=-1)
            values = paddle.matmul(self.act(gate) * up, self.down_proj[expert], transpose_y=True)
            values = values * paddle.gather_nd(weights, locations).unsqueeze(-1).astype(hidden.dtype)
            result = paddle.scatter_nd_add(result, tokens.unsqueeze(-1), values)
        return result


class DeepseekOcr2TextMoe(nn.Layer):
    def __init__(self, config):
        super().__init__()
        if config.tensor_model_parallel_size > 1 or config.sequence_parallel:
            raise NotImplementedError("Native packed OCR-2 experts currently require single-device execution")
        self.gate = DeepseekOcr2TextTopkRouter(config)
        self.experts = DeepseekOcr2TextExperts(config)
        self.shared_experts = MLP(config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts)

    def forward(self, hidden):
        flat = hidden.reshape([-1, hidden.shape[-1]])
        weights, indices = self.gate(flat)
        return self.experts(flat, weights, indices).reshape(hidden.shape) + self.shared_experts(hidden)


class DeepseekOcr2TextModel(LlamaModel):
    """Reuse the standard MHA/RoPE/mask/cache path; only sparse MLPs differ."""

    def __init__(self, config):
        super().__init__(config)
        for index, kind in enumerate(config.mlp_layer_types):
            if kind == "sparse":
                self.layers[index].mlp = DeepseekOcr2TextMoe(config)


def _sam_mapping(config):
    pairs = [
        ("patch_embed.projection.weight", "patch_embed.proj.weight", False),
        ("patch_embed.projection.bias", "patch_embed.proj.bias", False),
    ]
    if config.use_abs_pos:
        pairs.append(("pos_embed", "pos_embed", False))
    for index in range(config.num_hidden_layers):
        source, target = f"layers.{index}", f"blocks.{index}"
        for first, second in (("layer_norm1", "norm1"), ("layer_norm2", "norm2")):
            for suffix in ("weight", "bias"):
                pairs.append((f"{source}.{first}.{suffix}", f"{target}.{second}.{suffix}", False))
        for first, second in (
            ("attn.qkv", "attn.qkv"),
            ("attn.proj", "attn.proj"),
            ("mlp.lin1", "mlp.lin1"),
            ("mlp.lin2", "mlp.lin2"),
        ):
            pairs.append((f"{source}.{first}.weight", f"{target}.{second}.weight", True))
            if first != "attn.qkv" or config.qkv_bias:
                pairs.append((f"{source}.{first}.bias", f"{target}.{second}.bias", False))
        if config.use_rel_pos:
            for name in ("rel_pos_h", "rel_pos_w"):
                pairs.append((f"{source}.attn.{name}", f"{target}.attn.{name}", False))
    for first, second in (
        ("neck.conv1", "neck.0"),
        ("neck.conv2", "neck.2"),
        ("neck.layer_norm1", "neck.1"),
        ("neck.layer_norm2", "neck.3"),
        ("proj.conv1", "net_2"),
        ("proj.conv2", "net_3"),
    ):
        pairs.append((first + ".weight", second + ".weight", False))
        if "norm" in first:
            pairs.append((first + ".bias", second + ".bias", False))
    return pairs


class DeepseekOcr2PreTrainedModel(PretrainedModel):
    config_class = DeepseekOcr2Config
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]

    @classmethod
    def _mapping(cls, config, inverse=False):
        is_base = cls == cls.base_model_class
        prefix = "" if is_base else "model."
        statements = []

        def add(source, target=None, transpose=False):
            target = source if target is None else target
            source, target = "model." + source, prefix + target
            if inverse:
                source, target = target, source
            statements.append(f"{source}{'^T' if transpose else ''} -> {target}")

        add("view_separator")
        add("multi_modal_projector.weight", transpose=True)
        add("multi_modal_projector.bias")
        for name in ("query_768_resolution", "query_1024_resolution"):
            add(f"vision_tower.{name}.weight")
        for source, target, transpose in _sam_mapping(config.vision_config.sam_config):
            add("vision_tower.sam_encoder." + source, "vision_tower.sam_encoder." + target, transpose)
        # Reuse Qwen2's fused QKV/FFN mappings, excluding its (absent) token embedding.
        method = Qwen2Model._gen_inv_aoa_config if inverse else Qwen2Model._gen_aoa_config
        for line in method(config.vision_config.encoder_config)["aoa_statements"]:
            if "embed_tokens" in line or "lm_head" in line:
                continue
            left, right = line.split(" -> ")
            if inverse:
                left = (
                    left.replace("model.", "model.vision_tower.vision_encoder.")
                    if left.startswith("model.")
                    else prefix + "vision_tower.vision_encoder." + left
                )
                right = right.replace("model.", "model.vision_tower.vision_encoder.")
            else:
                left = left.replace("model.", "model.vision_tower.vision_encoder.")
                right = prefix + "vision_tower.vision_encoder." + right
            statements.append(left + " -> " + right)
        add("language_model.embed_tokens.weight")
        add("language_model.norm.weight")
        text = config.text_config
        for index, kind in enumerate(text.mlp_layer_types):
            layer = f"language_model.layers.{index}"
            for name in ("input_layernorm", "post_attention_layernorm"):
                add(f"{layer}.{name}.weight")
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                add(f"{layer}.self_attn.{name}.weight", transpose=True)
                if text.attention_bias:
                    add(f"{layer}.self_attn.{name}.bias")
            mlp = f"{layer}.mlp"
            if kind == "sparse":
                for name in ("gate.weight", "experts.gate_up_proj", "experts.down_proj"):
                    add(f"{mlp}.{name}")
                mlp += ".shared_experts"
            for name in ("gate_proj", "up_proj", "down_proj"):
                add(f"{mlp}.{name}.weight", transpose=True)
                if text.mlp_bias:
                    add(f"{mlp}.{name}.bias")
        if not is_base:
            if not config.tie_word_embeddings:
                statements.append("lm_head.weight -> lm_head.weight")
            elif not inverse:
                statements.append("model.language_model.embed_tokens.weight -> lm_head.weight")
        return {"aoa_statements": statements}

    @classmethod
    def _gen_aoa_config(cls, config):
        return cls._mapping(config)

    @classmethod
    def _gen_inv_aoa_config(cls, config):
        return cls._mapping(config, inverse=True)


@dataclass
class DeepseekOcr2ModelOutputWithPast(BaseModelOutputWithPast):
    image_hidden_states: paddle.Tensor = None


@dataclass
class DeepseekOcr2CausalLMOutputWithPast(CausalLMOutputWithPast):
    image_hidden_states: paddle.Tensor = None


@dataclass
class DeepseekOcr2ModelOutputWithPooling(BaseModelOutputWithPooling):
    local_last_hidden_state: paddle.Tensor = None


@register_base_model
class DeepseekOcr2Model(DeepseekOcr2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.vision_tower = DeepseekOcr2VisionModel(config.vision_config)
        self.multi_modal_projector = Linear.create(
            config.vision_config.encoder_config.hidden_size,
            config.text_config.hidden_size,
            has_bias=True,
            linear_type="default",
        )
        self.language_model = DeepseekOcr2TextModel(config.text_config)
        self.view_separator = self.create_parameter([config.text_config.hidden_size])

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def get_image_features(self, pixel_values, pixel_values_local=None, num_local_patches=None, **kwargs):
        global_hidden = self.vision_tower(pixel_values)
        global_features = self.multi_modal_projector(global_hidden)
        batch = global_features.shape[0]
        counts = [0] * batch if num_local_patches is None else [int(x) for x in num_local_patches]
        if len(counts) != batch or any(x < 0 for x in counts):
            raise ValueError("num_local_patches must contain one nonnegative count per image")
        if sum(counts) != (0 if pixel_values_local is None else pixel_values_local.shape[0]):
            raise ValueError("Local crop counts do not match pixel_values_local")
        local_hidden = None if not sum(counts) else self.vision_tower(pixel_values_local)
        local_features = None if local_hidden is None else self.multi_modal_projector(local_hidden)
        features, offset = [], 0
        for image, count in zip(global_features.unbind(0), counts):
            parts = [image, self.view_separator.unsqueeze(0)]
            if count:
                parts.insert(0, local_features[offset : offset + count].flatten(0, 1))
            features.append(paddle.concat(parts, axis=0))
            offset += count
        return DeepseekOcr2ModelOutputWithPooling(
            last_hidden_state=global_hidden, pooler_output=features, local_last_hidden_state=local_hidden
        )

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        pixel_values_local=None,
        num_local_patches=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        return_dict=True,
        attn_mask_startend_row_indices=None,
        output_hidden_states=None,
        **kwargs
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        image_features = None
        if pixel_values is not None:
            features = self.get_image_features(pixel_values, pixel_values_local, num_local_patches).pooler_output
            image_features = paddle.concat(features, axis=0).astype(inputs_embeds.dtype)
            if input_ids is None:
                embedding = self.get_input_embeddings()(paddle.to_tensor(self.config.image_token_id, dtype="int64"))
                mask = paddle.all(inputs_embeds == embedding, axis=-1)
            else:
                mask = input_ids == self.config.image_token_id
            if int(mask.astype("int64").sum()) != image_features.shape[0]:
                raise ValueError("Image features and image tokens do not match")
            inputs_embeds = paddle.masked_scatter(
                inputs_embeds, mask.unsqueeze(-1).expand_as(inputs_embeds), image_features.flatten()
            )
        elif pixel_values_local is not None:
            raise ValueError("Local crops require a global image view")
        output = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
            output_hidden_states=output_hidden_states,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )
        result = DeepseekOcr2ModelOutputWithPast(
            last_hidden_state=output.last_hidden_state,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            image_hidden_states=image_features,
        )
        return result if return_dict else result.to_tuple()


class DeepseekOcr2ForConditionalGeneration(DeepseekOcr2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekOcr2Model(config)
        self.lm_head = LMHead(config.text_config)
        self.criterion = CriterionLayer(config.text_config)
        self.tie_weights()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def get_image_features(self, pixel_values, **kwargs):
        return self.model.get_image_features(pixel_values, **kwargs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        logits_to_keep=0,
        return_dict=True,
        **kwargs
    ):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, **kwargs)
        hidden = output.last_hidden_state
        if labels is None:
            hidden = hidden[:, -logits_to_keep:] if isinstance(logits_to_keep, int) else hidden[:, logits_to_keep]
        logits = self.lm_head(hidden)
        loss = self.criterion(logits, labels)[0] if labels is not None else None
        result = DeepseekOcr2CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            image_hidden_states=output.image_hidden_states,
        )
        return result if return_dict else result.to_tuple()

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, **kwargs)
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            for key in ("pixel_values", "pixel_values_local", "num_local_patches"):
                inputs.pop(key, None)
        return inputs


DeepseekOcr2ForCausalLM = DeepseekOcr2ForConditionalGeneration

__all__ = [
    "DeepseekOCR2Model",
    "DeepseekOCR2ForCausalLM",
    "DeepseekOCR2ForConditionalGeneration",
    "Attention",
    "ImageEncoderViT",
    "_parse_line_result",
    "dynamic_preprocess",
    "extract_coordinates_and_label",
    "load_image",
    "DeepseekOcr2PreTrainedModel",
    "DeepseekOcr2Model",
    "DeepseekOcr2ForConditionalGeneration",
    "DeepseekOcr2ForCausalLM",
]
