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

import itertools
from dataclasses import dataclass

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.distributed import fleet
from paddlefleet.models.common.empty_layer import EmptyLayer
from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding
from paddlefleet.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_spec,
)
from paddlefleet.models.gpt.lm_head import GPTLMHead
from paddlefleet.models.qwen3_5.layer_specs import get_qwen3_5_vision_spec
from paddlefleet.models.qwen3_5.qwen3_5_model import Qwen3_5RMSNorm, Qwen3_5RMSNormPipe
from paddlefleet.pipeline_parallel import NoPipelineParallel
from paddlefleet.spec_utils import LayerSpec, build_layer
from paddlefleet.tensor_parallel.mappings import scatter_to_sequence_parallel_region
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.transformer.paddle_norm import WrappedPaddleNorm, WrappedPaddleNormPipe
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import get_tensor_model_parallel_group_if_none

from ..gpt_provider import GPTModelProvider


@dataclass
class Qwen3_5VisionProvider(TransformerConfig):
    transform_rules = {
        "num_heads": "num_attention_heads",
    }
    patch_size: int = 16
    use_bias: bool = True
    add_qkv_bias: bool = True
    num_position_embeddings: int = 2304
    embed_dim: int = (1152,)
    hidden_size: int = 1152
    out_hidden_size: int = 3584
    in_channels: int = 3
    spatial_merge_size: int = 2
    spatial_patch_size: int = 16
    temporal_patch_size: int = 2
    hidden_dropout_prob: float = 0.0
    attention_dropout: float = 0.0
    intermediate_size: int = 4304
    initializer_range: float = 0.02
    gated_linear_unit: bool = False
    activation_func: object = F.gelu
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False
    persist_layer_norm: bool = True
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization: str = "LayerNorm"
    apply_rope_fusion: bool = True
    rms_norm_eps: float = 1e-6
    model_version: str = "qwen3_5"

    def provide(self):
        spec = get_qwen3_5_vision_spec(self)
        return build_layer(
            spec,
            seg_method="layer:TransformerLayer|EmptyLayer",
            num_stages=self.pipeline_model_parallel_size,
        )


@dataclass
class Qwen3_5TextModelProvider(GPTModelProvider):
    """Provider for Qwen3.5 language (text) model.

    Extends ``GPTModelProvider`` with Qwen3.5-specific defaults and
    ``transform_rules`` that map PaddleFormers config attribute names
    to PaddleFleet attribute names.
    """

    transform_rules = {
        "tensor_parallel_degree": "tensor_model_parallel_size",
        "pipeline_parallel_degree": "pipeline_model_parallel_size",
        "context_parallel_degree": "context_parallel_size",
        "expert_parallel_degree": "expert_model_parallel_size",
        "dtype": "params_dtype",
        "num_experts": "n_routed_experts",
        "num_local_experts": "n_routed_experts",
        "attn_output_gate": "gated_attention",
    }

    gated_linear_unit: bool = True
    bias_activation_fusion: bool = True
    normalization: str = "RMSNorm"
    position_embedding_type: str = "mrope"
    rotary_base: float = 10000000.0
    rotary_percent: float = 0.25
    mrope_section: list = None

    def __post_init__(self):
        super().__post_init__()
        # Qwen3.5 uses multimodal RoPE with 3D position_ids
        self.position_embedding_type = "mrope"
        if self.mrope_section is None:
            rope_params = getattr(self, "rope_parameters", None) or {}
            self.mrope_section = rope_params.get("mrope_section", [11, 11, 10])
        # Fused rope kernel does not support 3D position_ids required by mrope
        self.apply_rope_fusion = False
        # Qwen3_5TextConfig has num_experts=60 as class default even for dense models.
        # For dense models (model_type without "moe"), clear MoE config
        # so fleet creates dense MLP layers instead of MoE layers.
        model_type = getattr(self, "model_type", "")
        if "moe" not in model_type:
            self.n_routed_experts = None
            self.n_shared_experts = 0
            self.moe_shared_expert_gate = False
        # TODO: Qwen3.5 AOA has not been adapted for TP > 1 yet (first loss is abnormal).
        # Block TP > 1 until AOA TP support is implemented.
        tp_size = getattr(self, "tensor_model_parallel_size", 1) or 1
        if tp_size > 1:
            raise ValueError(
                f"Qwen3.5 does not support tensor_model_parallel_size > 1 yet "
                f"(got {tp_size}). AOA weight mapping for TP sharding has not been "
                f"adapted. Please set tensor_model_parallel_size to 1."
            )

    moe_grouped_gemm: bool = True
    moe_router_load_balancing_type: str = "aux_loss"
    moe_router_pre_softmax: bool = False
    moe_permute_fusion: bool = True
    moe_router_dtype: str = "fp32"
    persist_layer_norm: bool = True
    share_embeddings_and_output_weights: bool = False
    apply_rope_fusion: bool = False
    bias_dropout_fusion: bool = True
    use_qk_norm: bool = True
    moe_router_force_load_balancing: bool = False
    n_shared_experts: int = 1
    moe_shared_expert_gate: bool = True
    multimodal_embedding: bool = False


def get_qwen3_5_language_spec(config):
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        layer_types = ["full_attention"] * config.num_hidden_layers

    empty_layer_spec = LayerSpec(layer=EmptyLayer, extra_kwargs={"config": config})
    head_empty_layers = [empty_layer_spec] * config.num_empty_layers_add_in_head
    tail_empty_layers = [empty_layer_spec] * config.num_empty_layers_add_in_tail

    head_offset = getattr(config, "num_empty_layers_add_in_head", 0)

    LAYER_TYPE_MAP = {
        "full_attention": "self_attention",
        "linear_attention": "gated_delta_net",
    }

    transformer_layers_spec = []
    for i, lt in enumerate(layer_types):
        attn_type = LAYER_TYPE_MAP.get(lt)
        if attn_type is None:
            raise ValueError(f"Unknown layer type: {lt!r} at index {i}")
        spec = get_gpt_layer_local_spec(
            config=config,
            normalization=config.normalization,
            layer_number=i + head_offset,
            attention_layer_type=attn_type,
            num_experts=config.n_routed_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
            multi_latent_attention=config.multi_latent_attention,
        )

        sub = spec.sublayers_spec
        if sub.input_layernorm is WrappedPaddleNorm:
            sub.input_layernorm = Qwen3_5RMSNorm
        if sub.post_attention_layernorm is WrappedPaddleNorm:
            sub.post_attention_layernorm = Qwen3_5RMSNorm

        attn_spec = sub.self_attn
        if hasattr(attn_spec, "sublayers_spec"):
            attn_sub = attn_spec.sublayers_spec
            if hasattr(attn_sub, "q_norm") and attn_sub.q_norm is WrappedPaddleNorm:
                attn_sub.q_norm = Qwen3_5RMSNorm
            if hasattr(attn_sub, "k_norm") and attn_sub.k_norm is WrappedPaddleNorm:
                attn_sub.k_norm = Qwen3_5RMSNorm

        transformer_layers_spec.append(spec)

    full_spec = get_gpt_spec(
        config=config,
        transformer_layers_spec=transformer_layers_spec,
        mtp_layers_spec=None,
        vocab_size=config.vocab_size,
        max_sequence_length=config.max_sequence_length,
        head_empty_layers_spec=head_empty_layers,
        tail_empty_layers_spec=tail_empty_layers,
        position_embedding_type=config.position_embedding_type,
        rotary_percent=config.rotary_percent,
        rotary_base=config.rotary_base,
        rope_scaling=config.rope_scaling,
        parallel_output=config.parallel_output,
        tie_word_embeddings=config.tie_word_embeddings,
    )

    final_norm_spec = full_spec.sublayers_spec.layer_norm
    if final_norm_spec.layer is WrappedPaddleNormPipe:
        final_norm_spec.layer = Qwen3_5RMSNormPipe

    return full_spec


def build_qwen3_5_model(config, criterion):
    """Build a Qwen3.5 VL model (vision encoder + language decoder) from config.

    Parameters
    ----------
    config : PretrainedConfig
        Composite config with ``vision_config`` and ``text_config`` sub-configs,
        plus top-level fields such as ``image_token_id``, ``video_token_id``,
        and parallelism sizes (``tensor_model_parallel_size``, etc.).

    Returns
    -------
    Qwen3_5Model
        The composed vision-language model ready for training or inference.
    """
    vision_config = config.vision_config
    text_config = config.text_config

    # --- Build vision model via Qwen3_5VisionProvider ---
    vision_provider = Qwen3_5VisionProvider.from_config(vision_config)
    vision_provider.gated_linear_unit = False
    vision_model = vision_provider.provide()

    # --- Build language model via Qwen3_5TextModelProvider ---
    language_config = Qwen3_5TextModelProvider.from_config(text_config)
    # Propagate parallelism settings
    # language_provider.tensor_model_parallel_size = config.tensor_model_parallel_size
    # language_provider.pipeline_model_parallel_size = config.pipeline_model_parallel_size
    # language_provider.context_parallel_size = config.context_parallel_size
    # language_provider.expert_model_parallel_size = config.expert_model_parallel_size
    # language_provider.virtual_pipeline_model_parallel_size = config.virtual_pipeline_model_parallel_size
    # language_provider.sequence_parallel = getattr(config, "sequence_parallel", False)
    # Propagate multimodal settings
    # language_provider.multimodal_embedding = True
    language_config.image_token_id = config.image_token_id
    language_config.video_token_id = config.video_token_id

    language_spec = get_qwen3_5_language_spec(language_config)
    language_model = build_layer(
        language_spec,
        seg_method="layer:TransformerLayer|EmptyLayer",
        num_stages=1,
    )

    # --- Assemble the composite VL model ---
    strategy = fleet.DistributedStrategy()

    model = Qwen3_5Model(
        config=language_config,
        vision_model=NoPipelineParallel(vision_model, strategy),
        language_model=NoPipelineParallel(language_model, strategy),
        spatial_merge_size=getattr(config, "spatial_merge_size", config.vision_config.spatial_merge_size),
        image_token_id=config.image_token_id,
        video_token_id=config.video_token_id,
    )

    return FleetQwen3_5ForConditionalGeneration(config, model, criterion)


class Qwen3_5Model(FleetLayer):
    def __init__(
        self,
        config,
        vision_model=None,
        language_model=None,
        spatial_merge_size=2,
        image_token_id=None,
        video_token_id=None,
    ):
        assert isinstance(language_model, NoPipelineParallel)
        assert isinstance(vision_model, NoPipelineParallel)
        super().__init__(config=config)
        self.visual = vision_model
        self.language_model = language_model
        self.spatial_merge_size = spatial_merge_size
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.rope_deltas = None

        self.language_embedding = self._find_language_embedding()
        self.language_backbone = self._find_language_backbone()
        self.language_lm_head = self._find_lm_head()

        self.tp_group = get_tensor_model_parallel_group_if_none(None)

        if self.language_embedding is not None:
            embed_tokens = self.language_embedding.embedding.embed_tokens
            embed_tokens.reduce_scatter_embeddings = False

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

    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        dict_input = {
            "pixel_values": pixel_values,
            "grid_thw": image_grid_thw,
        }
        output = self.visual._layers.forward(dict_input)
        if isinstance(output, tuple):
            return output[0]
        return output

    def get_video_features(self, pixel_values_videos, video_grid_thw=None, **kwargs):
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

    def get_placeholder_mask(
        self,
        input_ids,
        inputs_embeds,
        image_features=None,
        video_features=None,
    ):
        if input_ids is None:
            embed_fn = self.get_input_embeddings()
            special_image_mask = (inputs_embeds == embed_fn(paddle.to_tensor(self.image_token_id, dtype="int64"))).all(
                -1
            )
            special_video_mask = (inputs_embeds == embed_fn(paddle.to_tensor(self.video_token_id, dtype="int64"))).all(
                -1
            )
        else:
            special_image_mask = input_ids == self.image_token_id
            special_video_mask = input_ids == self.video_token_id

        # n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        if image_features is not None:
            assert int(inputs_embeds[special_image_mask].numel()) == int(image_features.numel())

        # n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds)
        if video_features is not None:
            assert int(inputs_embeds[special_video_mask].numel()) == int(video_features.numel())

        return special_image_mask, special_video_mask

    def get_vision_position_ids(
        self,
        start_position,
        grid_thw,
        temp_merge_size=1,
        spatial_merge_size=1,
        time_interval=1,
        device=None,
    ):
        if isinstance(grid_thw, Tensor):
            t = int(grid_thw[0].item())
            h = int(grid_thw[1].item())
            w = int(grid_thw[2].item())
        else:
            t, h, w = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])

        llm_t = t // temp_merge_size
        llm_h = h // spatial_merge_size
        llm_w = w // spatial_merge_size
        seq_len = llm_t * llm_h * llm_w

        pos_w = paddle.arange(start_position, start_position + llm_w).tile([llm_h * llm_t])
        pos_h = paddle.arange(start_position, start_position + llm_h).repeat_interleave(llm_w * llm_t)
        pos_t = paddle.full([seq_len], start_position, dtype="int64")
        pos_t = pos_t * time_interval

        return paddle.stack([pos_t, pos_h, pos_w], axis=0)

    def get_rope_index(
        self,
        input_ids,
        mm_token_type_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        **kwargs,
    ):
        spatial_merge_size = self.spatial_merge_size
        mrope_position_deltas = []
        position_ids = paddle.zeros(
            [3, input_ids.shape[0], input_ids.shape[1]],
            dtype=input_ids.dtype,
        )

        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }

        for batch_idx in range(input_ids.shape[0]):
            current_input_ids = input_ids[batch_idx]
            input_token_type = mm_token_type_ids[batch_idx]

            if attention_mask is not None:
                mask = attention_mask[batch_idx].astype("bool")
                current_input_ids = current_input_ids[mask]
                input_token_type = input_token_type[mask]

            input_type_group = []
            for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
                group = list(group)
                input_type_group.append((key, group[0][0], group[-1][0] + 1))

            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(paddle.arange(text_len).reshape([1, -1]).expand([3, -1]) + current_pos)
                    current_pos += text_len
                else:
                    grid_thw = next(grid_iters[modality_type])
                    vision_position_ids = self.get_vision_position_ids(
                        current_pos,
                        grid_thw,
                        1,
                        spatial_merge_size,
                    )
                    llm_pos_ids_list.append(vision_position_ids)
                    h_val = int(grid_thw[1].item()) if isinstance(grid_thw, Tensor) else int(grid_thw[1])
                    w_val = int(grid_thw[2].item()) if isinstance(grid_thw, Tensor) else int(grid_thw[2])
                    current_pos += max(h_val, w_val) // spatial_merge_size

            llm_positions = paddle.concat(llm_pos_ids_list, axis=1).reshape([3, -1])

            if attention_mask is not None:
                mask = attention_mask[batch_idx].astype("bool")
                position_ids[:, batch_idx, mask] = llm_positions
            else:
                position_ids[:, batch_idx] = llm_positions

            mrope_position_deltas.append(int(llm_positions.max().item()) + 1 - len(current_input_ids))

        mrope_position_deltas = paddle.to_tensor(mrope_position_deltas, dtype="int64").unsqueeze(1)

        return position_ids, mrope_position_deltas

    def compute_3d_position_ids(
        self,
        input_ids=None,
        inputs_embeds=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        past_key_values=None,
        mm_token_type_ids=None,
    ):
        past_key_values_length = (
            0
            if past_key_values is None
            else past_key_values.get_seq_length()
            if hasattr(past_key_values, "get_seq_length")
            else 0
        )
        can_compute_mrope = (
            input_ids is not None
            and mm_token_type_ids is not None
            and (image_grid_thw is not None or video_grid_thw is not None)
        )

        if can_compute_mrope and (self.rope_deltas is None or past_key_values_length == 0):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
            return position_ids

        if self.rope_deltas is not None and inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
            if attention_mask is not None:
                position_ids = attention_mask.astype("int64").cumsum(-1) - 1
                position_ids = paddle.where(
                    attention_mask == 0,
                    paddle.zeros_like(position_ids),
                    position_ids,
                )
                position_ids = position_ids.reshape([1, batch_size, -1]).tile([3, 1, 1])
            else:
                position_ids = (
                    paddle.arange(
                        past_key_values_length,
                        past_key_values_length + seq_length,
                    )
                    .reshape([1, 1, -1])
                    .expand([3, batch_size, -1])
                )

            delta = self.rope_deltas
            if delta.shape[0] != batch_size:
                delta = delta.tile([batch_size // delta.shape[0], 1])
            position_ids = position_ids + delta.unsqueeze(0)
            return position_ids

        return None

    def forward(self, dict_args):
        input_ids = dict_args.get("input_ids", None)
        inputs_embeds = dict_args.get("inputs_embeds", None)
        pixel_values = dict_args.get("pixel_values", None)
        pixel_values_videos = dict_args.get("pixel_values_videos", None)
        image_grid_thw = dict_args.get("image_grid_thw", None)
        video_grid_thw = dict_args.get("video_grid_thw", None)
        attention_mask = dict_args.get("attention_mask", None)
        position_ids = dict_args.get("position_ids", None)
        mm_token_type_ids = dict_args.get("mm_token_type_ids", None)
        past_key_values = dict_args.get("past_key_values", None)

        if inputs_embeds is None and input_ids is not None and self.language_model is not None:
            inputs_embeds = self.language_embedding.embedding.embed_tokens(input_ids)

        if pixel_values is not None and self.visual is not None:
            image_features = self.get_image_features(pixel_values, image_grid_thw)
            image_features = image_features.astype(inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids,
                inputs_embeds,
                image_features=image_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)

        if pixel_values_videos is not None and self.visual is not None:
            video_features = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_features = video_features.astype(inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds,
                video_features=video_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_features)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        if self.config.sequence_parallel:
            inputs_embeds = inputs_embeds.transpose([1, 0, 2]).contiguous()
            inputs_embeds = scatter_to_sequence_parallel_region(inputs_embeds, group=self.tp_group)

        dict_args["position_ids"] = position_ids
        dict_args["input_ids"] = None
        dict_args["decoder_input"] = inputs_embeds

        lm_dict_args = self.language_embedding(dict_args, decoder_input=inputs_embeds)

        for layer in self.language_backbone:
            lm_dict_args = layer(lm_dict_args)

        if self.language_lm_head is not None:
            logits = self.language_lm_head(lm_dict_args)
            return logits

        return lm_dict_args


class FleetQwen3_5ForConditionalGeneration(FleetLayer):
    def __init__(self, config, model, criterion):
        super().__init__(config)
        self.model = model
        self.criterion = criterion

    def forward(self, dict_args=None, **kwargs):
        if dict_args is None:
            dict_args = kwargs
        labels = dict_args.get("labels", None)
        logits = self.model(dict_args)
        loss = self.criterion(logits, labels)
        return loss

    def sharded_state_dict(self, structured_name_prefix: str = ""):
        """Build sharded state dict with proper name mapping for checkpoint loading.

        The Qwen3.5 model wraps language_model and visual in NoPipelineParallel,
        which adds `_layers.` prefix to parameter keys. This method bypasses
        NoPipelineParallel and directly calls sharded_state_dict on the underlying
        models (GPTModel for language, Qwen3_5VisionModel for vision).

        Both models handle pipeline layer name mapping internally via
        _pp_to_single_mapping, which converts numeric layer indices to semantic
        names with proper prefixes:
        - Language model: `0.embedding` -> `model.language_model.embedding`
        - Vision model: `0.patch_embed` -> `model.vision_model.patch_embed`

        The resulting keys will match the AOA config target format:
        - Language: `model.language_model.embedding.embed_tokens.weight`
        - Vision: `model.vision_model.patch_embed.proj.weight`
        """
        sharded_state_dict = {}

        # Get sharded state dict from language model (GPTModel wrapped in NoPipelineParallel)
        if self.model.language_model is not None:
            # Access the underlying PipelineLayer (GPTModel) directly
            # GPTModel.sharded_state_dict handles the model.language_model. prefix internally
            language_model = self.model.language_model._layers
            if hasattr(language_model, "sharded_state_dict"):
                lm_sharded = language_model.sharded_state_dict(structured_name_prefix="")
                sharded_state_dict.update(lm_sharded)

        # Get sharded state dict from vision model (Qwen3_5VisionModel wrapped in NoPipelineParallel)
        if self.model.visual is not None:
            # Access the underlying Qwen3_5VisionModel (TransformerEncoder) directly
            # TransformerEncoder.sharded_state_dict handles the model.vision_model. prefix
            # via _pp_to_single_mapping (since modal="vision_model")
            vision_model = self.model.visual._layers
            if hasattr(vision_model, "sharded_state_dict"):
                vm_sharded = vision_model.sharded_state_dict(structured_name_prefix="")
                sharded_state_dict.update(vm_sharded)

        # Get criterion parameters if any
        if self.criterion is not None:
            criterion_sharded = self.criterion.sharded_state_dict(
                structured_name_prefix=f"{structured_name_prefix}criterion."
            )
            sharded_state_dict.update(criterion_sharded)

        return sharded_state_dict
