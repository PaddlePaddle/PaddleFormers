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

from dataclasses import dataclass

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel
from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding
from paddlefleet.models.gpt.lm_head import GPTLMHead
from paddlefleet.models.kimi_k3 import (
    build_kimi_k3_vision_config,
    build_vision_startend_row_indices,
    kimi_k3_vision_builder,
    merge_input_ids_with_image_features,
)
from paddlefleet.transformer.layer import FleetLayer

from ...nn.criterion.interface import CriterionLayer
from ...nn.pp_model import GeneralModelForCausalLMPipe
from ..gpt_provider import GPTModelProvider
from ..model_utils import PretrainedModel
from .configuration import KimiK3Config, KimiK3TextConfig


@dataclass
class KimiK3ModelProvider(GPTModelProvider):
    """Kimi-K3 configuration provider for PaddleFleet GPTModel.

    Derives the mixed KDA and gated MLA attention schedule, block attention
    residuals, and flattened KDA fields required by PaddleFleet.
    """

    # === Kimi-K3 required defaults ===
    multi_latent_attention: bool = True
    gated_attention: bool = True
    use_qk_norm: bool = True
    gated_linear_unit: bool = True
    normalization: str = "RMSNorm"

    # KDA/MLA hybrid attention schedule
    linear_attn_config: dict | None = None

    # General defaults
    share_embeddings_and_output_weights: bool = False

    transform_rules = {
        **GPTModelProvider.transform_rules,
        "dtype": "params_dtype",
        # HF config.json -> Fleet TransformerConfig field mappings
        **KimiK3TextConfig._HF_TO_FLEET_FIELD_MAP,
    }

    def __post_init__(self):
        if not isinstance(self.linear_attn_config, dict):
            raise ValueError("Kimi-K3 requires linear_attn_config.")
        self._build_layer_types()
        self._expand_attn_res_block_size()
        self._flatten_linear_attn_config()
        super().__post_init__()

    def _build_layer_types(self):
        """Turn the one-based KDA/MLA schedule into a per-layer type list."""
        kda_layers = self._layer_numbers("kda_layers")
        full_attn_layers = self._layer_numbers("full_attn_layers")
        overlap = kda_layers & full_attn_layers
        if overlap:
            raise ValueError(
                "Kimi-K3 kda_layers and full_attn_layers must be disjoint; " f"overlap={sorted(overlap)}."
            )
        expected_layers = set(range(1, self.num_hidden_layers + 1))
        actual_layers = kda_layers | full_attn_layers
        if actual_layers != expected_layers:
            raise ValueError(
                "Kimi-K3 attention schedule must cover every decoder layer "
                f"exactly once; missing={sorted(expected_layers - actual_layers)}, "
                f"out_of_range={sorted(actual_layers - expected_layers)}."
            )
        self.layer_types = [
            "kimi_delta_attention" if layer_number in kda_layers else "multi_latent_attention"
            for layer_number in range(1, self.num_hidden_layers + 1)
        ]

    def _layer_numbers(self, name):
        """Parse and validate a list of one-based layer numbers."""
        values = self.linear_attn_config.get(name)
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"Kimi-K3 {name} must be a list of layer numbers.")
        if any(type(value) is not int for value in values):
            raise ValueError(f"Kimi-K3 {name} must contain only integers.")
        if len(values) != len(set(values)):
            raise ValueError(f"Kimi-K3 {name} contains duplicate layer numbers.")
        return set(values)

    def _expand_attn_res_block_size(self):
        """Fleet counts attention and MLP as two residual sublayers; the source
        value counts decoder layers, so double it."""
        if self.attn_res_block_size is None or self.attn_res_block_size <= 0:
            raise ValueError("Kimi-K3 attn_res_block_size must be a positive integer.")
        self.block_attention_residuals = True
        self.attn_res_block_size *= 2

    def _flatten_linear_attn_config(self):
        """Flatten the nested KDA config into Fleet TransformerConfig fields."""
        cfg = self.linear_attn_config
        head_dim = cfg["head_dim"]
        num_heads = cfg["num_heads"]
        self.linear_conv_kernel_dim = cfg["short_conv_kernel_size"]
        self.linear_key_head_dim = head_dim
        self.linear_value_head_dim = head_dim
        self.linear_num_key_heads = num_heads
        self.linear_num_value_heads = num_heads
        self.linear_gate_lora_rank = head_dim
        self.linear_use_full_rank_gate = cfg.get("use_full_rank_gate", False)
        self.linear_gate_lower_bound = cfg.get("gate_lower_bound")


class KimiK3PretrainedModel(PretrainedModel):
    config_class = KimiK3Config
    base_model_prefix = "model"


def _build_text_model(model_class, config):
    text_config = config.get_text_config()

    # Parallelism config safeguards
    text_config.tensor_model_parallel_size = max(getattr(text_config, "tensor_model_parallel_size", 1), 1)
    text_config.context_parallel_size = max(getattr(text_config, "context_parallel_size", 1), 1)
    text_config.pipeline_model_parallel_size = max(getattr(text_config, "pipeline_model_parallel_size", 1), 1)
    text_config.virtual_pipeline_model_parallel_size = max(
        getattr(text_config, "virtual_pipeline_model_parallel_size", 1), 1
    )
    text_config.expert_model_parallel_size = max(getattr(text_config, "expert_model_parallel_size", 1), 1)

    model_provider = KimiK3ModelProvider.from_config(text_config)
    gpt_model = model_provider.provide()
    gpt_model.config_to_save = config
    gpt_model.is_fleet = model_class.is_fleet
    return gpt_model


class KimiK3Model(KimiK3PretrainedModel):
    """AutoModel-compatible alias for the Kimi-K3 text decoder."""

    is_fleet = True

    def __new__(cls, config):
        return _build_text_model(cls, config)


class KimiK3ForCausalLM(KimiK3PretrainedModel):
    """Kimi-K3 text-only causal language model."""

    is_fleet = True

    def __new__(cls, config):
        return _build_text_model(cls, config)


class KimiK3ForCausalLMPipe(KimiK3PretrainedModel, GeneralModelForCausalLMPipe):
    """Pipeline alias for the Kimi-K3 text-only model."""

    is_fleet = True

    def __new__(cls, config):
        return _build_text_model(cls, config)


def build_kimi_k3_vision_tower(vision_config, params_dtype=None):
    """Build the MoonViT3d tower from a :class:`KimiK3VisionConfig`.

    ``params_dtype`` must match the text backbone: the projector output is
    spliced into the text embedding stream, and the patch-embed conv would
    otherwise hit a dtype mismatch against ``pixel_values``.
    """
    overrides = vision_config.to_fleet_vision_overrides()
    if params_dtype is not None:
        overrides["params_dtype"] = params_dtype
    fleet_config = build_kimi_k3_vision_config(**overrides)
    tower = kimi_k3_vision_builder(
        fleet_config,
        seg_method="layer:TransformerLayer|EmptyLayer",
        num_stages=fleet_config.pipeline_model_parallel_size,
    )
    return tower, fleet_config


class KimiK3VLModel(FleetLayer):
    """Vision tower + text backbone with the K3 dynamic-expansion fusion.

    The text stream carries exactly one placeholder token per media and the model
    expands it into the real visual token count, so the sequence grows inside
    ``forward`` and ``attention_mask`` / ``labels`` / ``position_ids`` must be
    rebuilt. Visual tokens then use plain 1-D position ids continuous with the
    text, not a three-axis MRoPE.
    """

    def __init__(
        self,
        config,
        vision_model=None,
        language_model=None,
        media_placeholder_token_id=None,
        pad_token_id=None,
        ignore_index=-100,
    ):
        assert isinstance(vision_model, NoPipelineParallel)
        assert isinstance(language_model, NoPipelineParallel)
        super().__init__(config=config)
        self.visual = vision_model
        self.language_model = language_model
        self.media_placeholder_token_id = media_placeholder_token_id
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

        self.language_embedding = self._find_language_embedding()
        self.language_backbone = self._find_language_backbone()
        self.language_lm_head = self._find_lm_head()

        # ``forward`` embeds ``input_ids`` here and feeds the merged sequence back
        # in as ``decoder_input``, so the embedding is required. The lm head is
        # optional: a non-last pipeline stage has none.
        if self.language_embedding is None:
            raise RuntimeError(
                "no GPTEmbedding found in the Kimi-K3 language backbone; the "
                "multimodal fusion path cannot embed input_ids without it"
            )
        self.language_embedding.embedding.embed_tokens.reduce_scatter_embeddings = False

    def _find_language_embedding(self):
        for layer in self.language_model._layers.run_function:
            if isinstance(layer, GPTEmbedding):
                return layer
        return None

    def _find_language_backbone(self):
        return [
            layer
            for layer in self.language_model._layers.run_function
            if not isinstance(layer, (GPTEmbedding, GPTLMHead))
        ]

    def _find_lm_head(self):
        for layer in self.language_model._layers.run_function:
            if isinstance(layer, GPTLMHead):
                return layer
        return None

    def get_image_features(self, pixel_values, grid_thws):
        """Run the vision tower; returns one ``(tokens_i, hidden)`` per media."""
        dict_input = {
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
            "attn_mask_startend_row_indices": build_vision_startend_row_indices(grid_thws),
        }
        output = self.visual._layers.forward(dict_input)
        features = output["hidden_states"]
        if not isinstance(features, (list, tuple)):
            features = [features]
        return features

    def forward(self, dict_args):
        """Embed, fuse the visual tokens, then run the text backbone.

        Media inputs make the sequence longer, so ``attention_mask`` / ``labels``
        / ``position_ids`` are rewritten in ``dict_args`` for the caller to read
        back after this returns.
        """
        input_ids = dict_args["input_ids"]
        pixel_values = dict_args.get("pixel_values", None)
        grid_thws = dict_args.get("image_grid_thw", None)
        attention_mask = dict_args.get("attention_mask", None)
        labels = dict_args.get("labels", None)

        # Without the grid the vision tower cannot run; fail loudly rather than
        # silently falling back to text-only training.
        if pixel_values is not None and grid_thws is None:
            raise ValueError(
                "pixel_values were provided without `image_grid_thw`; the Kimi-K3 "
                "vision tower needs the per-image [T, H, W] patch grid."
            )

        inputs_embeds = self.language_embedding.embedding.embed_tokens(input_ids)

        if pixel_values is not None:
            image_features = [f.astype(inputs_embeds.dtype) for f in self.get_image_features(pixel_values, grid_thws)]
            if attention_mask is None:
                attention_mask = paddle.ones(input_ids.shape, dtype="int64")
            # One placeholder expands into many visual tokens, so every
            # per-position tensor changes length here.
            inputs_embeds, attention_mask, labels, position_ids = merge_input_ids_with_image_features(
                image_features,
                inputs_embeds,
                input_ids,
                attention_mask,
                image_token_index=self.media_placeholder_token_id,
                pad_token_id=self.pad_token_id,
                ignore_index=self.ignore_index,
                labels=labels,
            )
            dict_args["attention_mask"] = attention_mask
            dict_args["labels"] = labels
            dict_args["position_ids"] = position_ids

        dict_args["input_ids"] = None
        dict_args["decoder_input"] = inputs_embeds

        lm_dict_args = self.language_embedding(dict_args, decoder_input=inputs_embeds)
        for layer in self.language_backbone:
            lm_dict_args = layer(lm_dict_args)

        if self.language_lm_head is not None:
            return self.language_lm_head(lm_dict_args)
        return lm_dict_args


class FleetKimiK3ForConditionalGeneration(FleetLayer, PretrainedModel):
    config_class = None

    def _post_init(self, original_init, *args, **kwargs):
        pass

    def __init__(self, config, model, criterion):
        super().__init__(config)
        self.model = model
        self.criterion = criterion

    def forward(self, dict_args=None, **kwargs):
        """Run the multimodal model and return the scalar training loss.

        Training only: ``generate()`` is unsupported because the fusion rewrites
        the sequence length and this wrapper has no KV-cache contract, so
        ``labels`` is required. ``dict_args`` may also arrive as plain keyword
        arguments, because ``Trainer.compute_loss`` calls ``model(**inputs)`` for
        models it does not recognise as a Fleet ``GPTModel``.
        """
        if dict_args is None:
            dict_args = kwargs
        logits = self.model(dict_args)
        # Read labels only after the inner forward, which rebuilds them at the
        # expanded sequence length.
        labels = dict_args.get("labels", None)
        if labels is None:
            raise ValueError(
                "KimiK3ForConditionalGeneration supports training only and requires "
                "`labels`; generation is not implemented yet."
            )
        # With num_nextn_predict_layers > 0 the lm head emits the main logits
        # plus one per MTP layer.
        if isinstance(logits, list):
            return self.criterion(logits[0], labels, mtp_logits=logits[1:])
        return self.criterion(logits, labels)


def _build_vl_model(config, criterion):
    """Wire the PaddleFleet vision tower and fusion helper
    (``paddlefleet.models.kimi_k3``) to the :class:`KimiK3ModelProvider` backbone.
    """
    text_config = config.get_text_config()
    vision_config = config.vision_config

    for name in (
        "tensor_model_parallel_size",
        "context_parallel_size",
        "pipeline_model_parallel_size",
        "virtual_pipeline_model_parallel_size",
        "expert_model_parallel_size",
    ):
        setattr(text_config, name, max(getattr(text_config, name, 1), 1))

    vision_model, _ = build_kimi_k3_vision_tower(
        vision_config,
        params_dtype=getattr(text_config, "params_dtype", None) or getattr(text_config, "dtype", None),
    )
    language_model = KimiK3ModelProvider.from_config(text_config).provide()

    strategy = fleet.DistributedStrategy()
    model = KimiK3VLModel(
        config=text_config,
        vision_model=NoPipelineParallel(vision_model, strategy),
        language_model=NoPipelineParallel(language_model, strategy),
        media_placeholder_token_id=config.media_placeholder_token_id,
        pad_token_id=getattr(config, "pad_token_id", None),
        ignore_index=getattr(config, "ignore_index", -100),
    )
    model.config_to_save = config
    return FleetKimiK3ForConditionalGeneration(config, model, criterion)


class KimiK3ForConditionalGeneration(KimiK3PretrainedModel):
    """Kimi-K3 multimodal model: MoonViT3d vision tower + KDA/MLA text backbone."""

    is_fleet = True

    def __new__(cls, config, have_criterion=True):
        if getattr(config, "vision_config", None) is None:
            raise ValueError(
                "KimiK3ForConditionalGeneration requires config.vision_config; "
                "use KimiK3ForCausalLM for the text-only model."
            )

        criterion = CriterionLayer(config.get_text_config()) if have_criterion else None
        return _build_vl_model(config, criterion)


__all__ = [
    "KimiK3Model",
    "KimiK3ForCausalLM",
    "KimiK3ForCausalLMPipe",
    "KimiK3ForConditionalGeneration",
    "KimiK3ModelProvider",
]
