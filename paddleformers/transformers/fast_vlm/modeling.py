# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Native FastVLM: fixed image placeholders, composite backbones and shared loading."""

from dataclasses import dataclass
from functools import partial

import paddle
import paddle.nn.functional as F
from paddle import nn

from ...nn.activation import ACT2FN
from ...nn.criterion.interface import CriterionLayer
from ...nn.linear import Linear
from ...nn.lm_head import LMHead
from ..fastvlm.modeling_vision import FastViT, LayerNormChannel, RepCPE
from ..model_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from ..model_utils import PretrainedModel, register_base_model
from ..qwen2.modeling import Qwen2ForCausalLMDeprecated, Qwen2Model
from .configuration import FastVlmConfig


class FastVlmLayerNorm2d(LayerNormChannel):
    def forward(self, x):
        x = x.transpose([0, 2, 3, 1])
        return F.layer_norm(x, [self.weight.shape[0]], self.weight, self.bias, self.eps).transpose([0, 3, 1, 2])


def _vision_args(config):
    args = dict(config.model_args)
    args.pop("inference_mode", None)
    layers = args.pop("layers", [2, 12, 24, 4, 2])
    dims = args.pop("embed_dims", [96, 192, 384, 768, 1536])
    mixers = args.pop("token_mixers", ["repmixer", "repmixer", "repmixer", "attention", "attention"])
    ratios = args.pop("mlp_ratios", [4] * len(layers))
    cls_ratio = args.pop("cls_ratio", 2.0)
    down_patch_size = args.pop("down_patch_size", 7)
    # These native timm defaults are structural, not checkpoint tensors.
    for name, expected in {
        "in_chans": 3,
        "num_classes": 0,
        "stem_use_scale_branch": False,
        "lkc_use_act": True,
    }.items():
        if name in args and args.pop(name) != expected:
            raise ValueError(f"Unsupported FastViT model_args value for {name}")
    if args:
        raise ValueError(f"Unsupported FastViT model_args: {sorted(args)}")
    if len(layers) != 5 or len(dims) != 5 or len(mixers) != 5:
        raise ValueError("fastvit_mci3 requires five stages")
    if int(dims[-1] * cls_ratio) != config.hidden_size:
        raise ValueError("vision hidden_size must match the final FastViT channel count")
    return dict(
        layers=layers,
        embed_dims=dims,
        token_mixers=mixers,
        mlp_ratios=ratios,
        cls_ratio=cls_ratio,
        down_patch_size=down_patch_size,
    )


class FastVlmVisionModel(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.timm_model = FastViT(
            **_vision_args(config),
            downsamples=[True] * 5,
            pos_embs=[None, None, None, partial(RepCPE, spatial_shape=(7, 7)), partial(RepCPE, spatial_shape=(7, 7))],
            norm_layer=FastVlmLayerNorm2d,
            stem_scale_branch=False,
            inference_mode=True,
            num_classes=0,
        )

    def forward(self, pixel_values):
        model = self.timm_model
        return model.conv_exp(model.forward_tokens(model.forward_embeddings(pixel_values)))


class FastVlmMultiModalProjector(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = Linear.create(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            has_bias=config.multimodal_projector_bias,
            linear_type="default",
        )
        self.linear_2 = Linear.create(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            has_bias=config.multimodal_projector_bias,
            linear_type="default",
        )
        self.act = ACT2FN[config.projector_hidden_act]

    def forward(self, x):
        return self.linear_2(self.act(self.linear_1(x)))


def _vision_key_pairs(config):
    """Explicit native timm -> shared FastViT parameter paths; no model allocation."""
    args = _vision_args(config)
    pairs = []

    def add(source, target, suffixes):
        pairs.extend((source + "." + suffix, target + "." + suffix) for suffix in suffixes)

    for i in range(3):
        add(f"stem.{i}.reparam_conv", f"patch_embed.{i}.reparam_conv", ("weight", "bias"))
    block_indices = [0, 2, 4, 7, 10]
    for stage, (depth, mixer) in enumerate(zip(args["layers"], args["token_mixers"])):
        if stage:
            down = [1, 3, 5, 8][stage - 1]
            add(
                f"stages.{stage}.downsample.proj.0.reparam_conv",
                f"network.{down}.proj.0.lkb_reparam",
                ("weight", "bias"),
            )
            add(
                f"stages.{stage}.downsample.proj.1.reparam_conv",
                f"network.{down}.proj.1.reparam_conv",
                ("weight", "bias"),
            )
        if stage >= 3:
            add(
                f"stages.{stage}.pos_emb.reparam_conv",
                f"network.{block_indices[stage] - 1}.reparam_conv",
                ("weight", "bias"),
            )
        for layer in range(depth):
            src, dst = f"stages.{stage}.blocks.{layer}", f"network.{block_indices[stage]}.{layer}"
            for name in ("fc1", "fc2"):
                add(f"{src}.mlp.{name}", f"{dst}.convffn.{name}", ("weight", "bias"))
            add(f"{src}.mlp.conv.conv", f"{dst}.convffn.conv.conv", ("weight",))
            add(f"{src}.mlp.conv.bn", f"{dst}.convffn.conv.bn", ("weight", "bias"))
            for source, target in (("running_mean", "_mean"), ("running_var", "_variance")):
                pairs.append((f"{src}.mlp.conv.bn.{source}", f"{dst}.convffn.conv.bn.{target}"))
            if mixer == "repmixer":
                add(f"{src}.token_mixer.reparam_conv", f"{dst}.token_mixer.reparam_conv", ("weight", "bias"))
                pairs.append((f"{src}.layer_scale.gamma", f"{dst}.layer_scale"))
            elif mixer == "attention":
                add(f"{src}.norm", f"{dst}.norm", ("weight", "bias"))
                add(f"{src}.token_mixer.qkv", f"{dst}.token_mixer.qkv", ("weight",))
                add(f"{src}.token_mixer.proj", f"{dst}.token_mixer.proj", ("weight", "bias"))
                for i in (1, 2):
                    pairs.append((f"{src}.layer_scale_{i}.gamma", f"{dst}.layer_scale_{i}"))
            else:
                raise ValueError(f"Unsupported FastViT mixer: {mixer}")
    add("final_conv.reparam_conv", "conv_exp.reparam_conv", ("weight", "bias"))
    for src, dst in (("fc1", "reduce"), ("fc2", "expand")):
        add(f"final_conv.se.{src}", f"conv_exp.se.{dst}", ("weight", "bias"))
    return pairs


class FastVlmPreTrainedModel(PretrainedModel):
    config_class = FastVlmConfig
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]

    @classmethod
    def _mapping(cls, config, inverse=False):
        is_base = cls == cls.base_model_class
        text_class = Qwen2Model if is_base else Qwen2ForCausalLMDeprecated
        method = text_class._gen_inv_aoa_config if inverse else text_class._gen_aoa_config
        statements = []
        for statement in method(config.text_config)["aoa_statements"]:
            if is_base and "lm_head" in statement:
                continue
            left, right = statement.replace("model.", "model.language_model.").split(" -> ")
            if is_base:
                if inverse and not left.startswith("model."):
                    left = "language_model." + left
                elif not inverse:
                    right = "language_model." + right
            statements.append(left + " -> " + right)
        prefix = "" if is_base else "model."
        for source, target in _vision_key_pairs(config.vision_config):
            source = "model.vision_tower.timm_model." + source
            target = prefix + "vision_tower.timm_model." + target
            statements.append(f"{target} -> {source}" if inverse else f"{source} -> {target}")
        if not inverse:
            # Paddle BatchNorm uses fixed momentum and has no PyTorch step counter.
            statements.append("model.vision_tower.timm_model.stages.$S.blocks.$B.mlp.conv.bn.num_batches_tracked -> _")
        for layer in ("linear_1", "linear_2"):
            source, target = f"model.multi_modal_projector.{layer}", f"{prefix}multi_modal_projector.{layer}"
            if inverse:
                source, target = target, source
            statements.append(f"{source}.weight^T -> {target}.weight")
            if config.multimodal_projector_bias:
                statements.append(f"{source}.bias -> {target}.bias")
        return {"aoa_statements": statements}

    @classmethod
    def _gen_aoa_config(cls, config):
        return cls._mapping(config)

    @classmethod
    def _gen_inv_aoa_config(cls, config):
        return cls._mapping(config, inverse=True)


@dataclass
class FastVlmModelOutputWithPast(BaseModelOutputWithPast):
    image_hidden_states: paddle.Tensor = None


@dataclass
class FastVlmCausalLMOutputWithPast(CausalLMOutputWithPast):
    image_hidden_states: paddle.Tensor = None


@register_base_model
class FastVlmModel(FastVlmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.vision_tower = FastVlmVisionModel(config.vision_config)
        self.multi_modal_projector = FastVlmMultiModalProjector(config)
        self.language_model = Qwen2Model(config.text_config)

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def get_image_features(
        self, pixel_values, vision_feature_layer=None, vision_feature_select_strategy=None, **kwargs
    ):
        if vision_feature_layer not in (None, -1) or vision_feature_select_strategy not in (None, "full"):
            raise ValueError("FastVLM supports only the full final vision feature map")
        features = self.vision_tower(pixel_values)
        projected = self.multi_modal_projector(features.flatten(2).transpose([0, 2, 1]))
        output = BaseModelOutputWithPooling(last_hidden_state=features, pooler_output=list(projected.unbind(0)))
        return output if kwargs.get("return_dict", True) else output.to_tuple()

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_features):
        if input_ids is None:
            token_embedding = self.get_input_embeddings()(paddle.to_tensor(self.config.image_token_id, dtype="int64"))
            mask = paddle.all(inputs_embeds == token_embedding, axis=-1)
        else:
            mask = input_ids == self.config.image_token_id
        if int(mask.astype("int64").sum()) != image_features.shape[0]:
            raise ValueError("Image features and image tokens do not match")
        return mask.unsqueeze(-1).expand_as(inputs_embeds)

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        return_dict=True,
        attn_mask_startend_row_indices=None,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        cache_position=None,
        **kwargs
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values, vision_feature_layer, vision_feature_select_strategy
            ).pooler_output
            image_features = paddle.concat(image_features, axis=0).astype(inputs_embeds.dtype)
            mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_features)
            inputs_embeds = paddle.masked_scatter(
                inputs_embeds,
                mask,
                image_features.flatten(),
            )
        if position_ids is None:
            offset = past_key_values.get_seq_length() if past_key_values is not None else 0
            positions = (
                cache_position
                if cache_position is not None
                else paddle.arange(offset, offset + inputs_embeds.shape[1], dtype="int64")
            )
            position_ids = positions.unsqueeze(0)
            position_ids = position_ids.expand([inputs_embeds.shape[0], -1])
        output = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            return_dict=True,
        )
        result = FastVlmModelOutputWithPast(
            last_hidden_state=output.last_hidden_state,
            past_key_values=output.past_key_values,
            image_hidden_states=image_features,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )
        return result if return_dict else result.to_tuple()


class FastVlmForConditionalGeneration(FastVlmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = FastVlmModel(config)
        self.lm_head = LMHead(config.text_config)
        self.criterion = CriterionLayer(config.text_config)
        self.tie_weights()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

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
        result = FastVlmCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            image_hidden_states=output.image_hidden_states,
        )
        return result if return_dict else result.to_tuple()

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, **kwargs)
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            inputs.pop("pixel_values", None)
        return inputs


FastVlmForCausalLM = FastVlmForConditionalGeneration

__all__ = ["FastVlmPreTrainedModel", "FastVlmModel", "FastVlmForConditionalGeneration", "FastVlmForCausalLM"]
