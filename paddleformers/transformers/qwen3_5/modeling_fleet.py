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

import contextlib
import copy
import itertools
import os
import types
from dataclasses import dataclass
from functools import partial

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    LayerSpec,
    NoPipelineParallel,
    build_spec_layer,
)
from paddlefleet.models.common.empty_layer import EmptyLayer
from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding
from paddlefleet.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_mtp_layers_spec,
    get_gpt_spec,
)
from paddlefleet.models.gpt.lm_head import GPTLMHead
from paddlefleet.models.qwen3_5.layer_specs import get_qwen3_5_vision_spec
from paddlefleet.models.qwen3_5.qwen3_5_model import Qwen3_5RMSNorm, Qwen3_5RMSNormPipe
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.transformer.paddle_norm import WrappedPaddleNorm, WrappedPaddleNormPipe
from paddlefleet.transformer.transformer_config import TransformerConfig

from ...nn.criterion.interface import CriterionLayer
from ...utils.log import logger
from ..gpt_provider import GPTModelProvider
from ..model_utils import (
    HFFormatFullParamSaver,
    PretrainedModel,
    clean_unrelated_safetensors,
)


@dataclass
class Qwen3_5VisionProvider(TransformerConfig):
    transform_rules = {
        "num_heads": "num_attention_heads",
        "depth": "num_hidden_layers",
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
        return build_spec_layer(
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

    @classmethod
    def from_config(cls, config):
        # PretrainedConfig keeps model_type as a class attribute, while the
        # provider converter only copies instance attributes.
        provider_config = copy.copy(config)
        provider_config.model_type = config.model_type
        return super().from_config(provider_config)

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
        # Unify MTP layer configuration
        # "config" source: mtp_num_hidden_layers (from model's config.json)
        # "yaml" source: num_nextn_predict_layers (from training yaml)
        # Priority: yaml > config > default (0 = no MTP)
        config_mtp = getattr(self, "mtp_num_hidden_layers", 0) or 0
        yaml_mtp = self.num_nextn_predict_layers or 0

        if yaml_mtp > 0:
            self.mtp_num_layers = yaml_mtp
            self.num_nextn_predict_layers = yaml_mtp
        elif config_mtp > 0:
            self.mtp_num_layers = config_mtp
            self.num_nextn_predict_layers = config_mtp
        else:
            self.mtp_num_layers = 0
            self.num_nextn_predict_layers = 0

    moe_expert_fusion: bool = True
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

    def provide(self, pre_process=None, post_process=None, vp_stage=None, loss_fn=None):
        """Override GPTModelProvider.provide() to use Qwen3.5-specific layer spec.

        The default provide() uses gpt_builder() which calls get_gpt_decoder_layers_spec()
        and does not handle mixed attention types (linear_attention / full_attention).
        This override uses get_qwen3_5_language_spec() instead.

        Because this bypasses ``gpt_builder``, every behaviour the base
        implementation derives from the config has to be reproduced here or
        rejected outright. Silently ignoring one is how the pipeline path ends up
        building a subtly different model from the non-pipeline path.
        """
        # gpt_builder() decides these from the config; this override cannot, so
        # refuse instead of quietly building something else.
        if self.separate_mtp_headloss:
            raise ValueError(
                "separate_mtp_headloss is not supported by "
                "Qwen3_5TextModelProvider.provide(): gpt_builder() would both "
                "add MultiTokenPredictionLayer to seg_method and replace "
                "loss_fn with MainLanguageLoss, and whether that replacement "
                "should override the pipeline criterion is undecided. Set "
                "separate_mtp_headloss=False."
            )
        for name, value in (
            ("pre_process", pre_process),
            ("post_process", post_process),
            ("vp_stage", vp_stage),
        ):
            if value is not None:
                raise ValueError(
                    f"Qwen3_5TextModelProvider.provide() ignores {name!r}; "
                    f"got {value!r}. build_spec_layer() derives the stage "
                    "layout from num_stages and the config instead."
                )

        # Same flattening gpt_builder()'s caller performs: the spec reads
        # rope_theta / rope_type / mscale_all_dim off the config directly.
        if getattr(self, "rope_parameters", None):
            rope_type = self.rope_parameters.get("rope_type", None)
            if rope_type is not None and rope_type != "default":
                self.rope_type = rope_type
            if "rope_theta" in self.rope_parameters:
                self.rope_theta = self.rope_parameters["rope_theta"]
        if getattr(self, "rope_scaling", None) and "mscale_all_dim" in self.rope_scaling:
            self.mscale_all_dim = self.rope_scaling["mscale_all_dim"]

        pp_size = self.pipeline_model_parallel_size or 1

        # Build spec using Qwen3.5-specific function that handles mixed attention
        language_spec = get_qwen3_5_language_spec(self)

        # Use build_spec_layer with PP stage splitting
        seg_method = "layer:TransformerLayer|EmptyLayer"
        kwargs = {}
        if loss_fn is not None:
            kwargs["loss_fn"] = loss_fn

        model_init_device_context = contextlib.nullcontext
        if self.init_model_with_meta_device:
            model_init_device_context = partial(paddle.device, device="meta")
        with model_init_device_context():
            model = build_spec_layer(
                language_spec,
                seg_method=seg_method,
                num_stages=pp_size,
                **kwargs,
            )
        return model


def _build_mtp_layers_spec(config, transformer_layers_spec):
    """Build MTP layer specs with moe_expert_fusion disabled for qwen3.5.

    The AOA engine cannot handle concat+reshape with EP sharding for per-expert
    2D HF keys, so MTP layers use per-expert storage instead of expert fusion.
    """
    mtp_cfg = copy.copy(config)
    mtp_cfg.moe_expert_fusion = False
    # Create a new spec identical to the last decoder layer but with mtp_cfg
    # embedded, so MoELayer inside TransformerLayer uses per-expert weights.
    base_spec = transformer_layers_spec[-1]
    mtp_transformer_spec = LayerSpec(
        layer=base_spec.layer,
        sublayers_spec=base_spec.sublayers_spec,
        extra_kwargs={**base_spec.extra_kwargs, "config": mtp_cfg},
    )
    return get_gpt_mtp_layers_spec(mtp_cfg, [mtp_transformer_spec])


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
            moe_expert_fusion=config.moe_expert_fusion,
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
        mtp_layers_spec=_build_mtp_layers_spec(config, transformer_layers_spec) if config.mtp_num_layers > 0 else None,
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


_PIPELINE_FIRST_STAGE_KEYS = [
    "input_ids",
    "attention_mask",
    # The collator emits attn_mask_startend_row_indices instead of
    # attention_mask whenever use_attn_mask_startend_row_indices is on (the
    # default). Dropping it would silently switch the attention kernel from
    # flashmask_attention to scaled_dot_product_attention(is_causal=True).
    "attn_mask_startend_row_indices",
    "position_ids",
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "mm_token_type_ids",
    # Only produced when mtp_attention_flexible is enabled; GPTEmbedding
    # asserts these two are both present or both absent.
    "mtp_startend_row_indices_all",
    "mtp_hidden_inputs_mask_all",
]

# Keys the collator can emit that the PaddleFleet path deliberately does not
# forward, either because nothing reads them (grep confirms zero readers under
# paddlefleet/ for all of these) or because they belong to the last stage.
# Everything the collator produces must be in exactly one of the two lists;
# _warn_unforwarded_keys reports the rest, because a key that goes missing here
# fails silently. That already cost one precision bug: without
# attn_mask_startend_row_indices the pipeline path quietly fell back from
# flashmask_attention to scaled_dot_product_attention(is_causal=True).
_PIPELINE_IGNORED_KEYS = frozenset(
    {
        "labels",  # routed to the last stage separately
        "nbatch_pack_offset",  # only the legacy paddleformers/nn/pp_model.py path reads it
        "mtp_attn_mask",  # the non-startend_row_indices MTP mask variant
        "token_type_ids",  # get_token_type_func path (Ernie-style collation)
        "images",
        "grid_thw",
        "input_features",  # audio inputs, which Qwen3.5-VL does not have
        "feature_attention_mask",
    }
)

_warned_unforwarded_keys = set()


def _warn_unforwarded_keys(keys):
    """Warn once per key about batch entries the pipeline drops on the floor."""
    unknown = {k for k in keys if k not in _PIPELINE_FIRST_STAGE_KEYS and k not in _PIPELINE_IGNORED_KEYS}
    unknown -= _warned_unforwarded_keys
    if unknown:
        _warned_unforwarded_keys.update(unknown)
        logger.warning(
            f"The data collator produced {sorted(unknown)}, which "
            "_PIPELINE_FIRST_STAGE_KEYS does not forward to pipeline stage 0. "
            "If the model needs them, add them there; if not, add them to "
            "_PIPELINE_IGNORED_KEYS to silence this."
        )


class Qwen3_5CriterionPipe(CriterionLayer):
    """``CriterionLayer`` that accepts the pipeline last stage's output.

    With MTP enabled, ``GPTLMHead.forward`` returns a list
    ``[main_logits, mtp_logits...]``, but the pipeline scheduler calls the loss
    function as ``loss_fn(output_tensor, labels)`` and asserts the result is a
    single Tensor — it never forwards ``mtp_logits`` itself. Split the list here
    so that PP and non-PP end up in the same ``mtp_sft_loss_forward``.
    """

    def forward(self, logits, labels, loss_mask=None, **kwargs):
        if isinstance(logits, list):
            return super().forward(logits[0], labels, loss_mask, mtp_logits=logits[1:], **kwargs)
        return super().forward(logits, labels, loss_mask, **kwargs)


def _prepare_qwen3_5_pipeline_inputs(inputs, gather_pp_need_data=True):
    """Prepare pipeline inputs for Qwen3.5 VL model.

    Splits batch dict into first_stage inputs and last_stage labels.
    Compatible with PaddleFleet's pipeline scheduler.

    For the list-of-dicts (multiple microbatches) path, each microbatch is
    kept as a list element so that _load_micro_batch_impl can index into it
    without needing all tensors to have the same seq_len dimension (variable
    sequence lengths in VL).

    ``gather_pp_need_data`` is accepted for signature compatibility only. The
    trainer passes ``False`` once it has buffered the data itself, which for
    other models selects an ``(inputs, labels)`` return instead of a data
    provider; this function always returns that tuple, so both call sites in
    trainer.py work with the same value.
    """
    if isinstance(inputs, dict):
        _warn_unforwarded_keys(inputs.keys())
        first_stage_batch = {k: inputs[k] for k in _PIPELINE_FIRST_STAGE_KEYS if k in inputs}
        last_stage_inputs = inputs.get("labels", None)
        return (first_stage_batch, last_stage_inputs)

    # List of dicts (multiple microbatches from gradient_accumulation)
    # Keep each microbatch as a separate list element so PP scheduler
    # can index into them individually (avoids concat across different seq_lens).
    if inputs:
        _warn_unforwarded_keys(inputs[0].keys())
    first_stage_batch = {}
    for key in _PIPELINE_FIRST_STAGE_KEYS:
        values = [data.get(key, None) for data in inputs]
        if any(v is not None for v in values):
            first_stage_batch[key] = values

    last_stage_inputs = [data.get("labels", None) for data in inputs]
    return (first_stage_batch, last_stage_inputs)


def _pp_save_pretrained(self, save_dir, is_main_process: bool = True, **kwargs):
    """``save_pretrained`` for the PP model, which is a bare ``GPTModel``.

    In PP mode ``build_qwen3_5_model`` returns the ``GPTModel``/``PipelineLayer``
    itself (the trainer requires a PipelineLayer), so it does not inherit
    ``PretrainedModel`` and has no ``save_pretrained``. This reimplements the
    ``flex_checkpoint`` HF-export branch of ``PretrainedModel.save_pretrained``,
    reading the composite ``PretrainedConfig`` from ``config_to_save`` instead of
    ``self.config`` (which is the PaddleFleet ``TransformerConfig``).
    """
    max_shard_size = kwargs.get("max_shard_size", "10GB")
    memory_growth_threshold = kwargs.get("memory_growth_threshold", 8 * (2**30))

    if os.path.isfile(save_dir):
        raise ValueError(f"Saving directory ({save_dir}) should be a directory, not a file")
    os.makedirs(save_dir, exist_ok=True)

    config_to_save = copy.deepcopy(self.config_to_save)
    aoa_config = self._gen_inv_aoa_config(config_to_save)

    clean_unrelated_safetensors(save_dir)
    HFFormatFullParamSaver(self, aoa_config, memory_growth_threshold=memory_growth_threshold).save_checkpoint(
        save_dir, max_shard_size
    )

    if is_main_process:
        if config_to_save.tensor_model_parallel_size > 1:
            config_to_save.tensor_model_parallel_size = 1
        config_to_save.save_pretrained(save_dir)


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
    FleetQwen3_5ForConditionalGeneration or GPTModel
        With ``pipeline_model_parallel_size == 1``, the composite
        ``FleetQwen3_5ForConditionalGeneration``. With PP enabled, the bare
        ``GPTModel``/``PipelineLayer`` produced by ``provide()`` — the pipeline
        scheduler requires a ``PipelineLayer`` at the top, so the vision encoder
        is attached to it as a sublayer instead of wrapping it.
    """
    vision_config = config.vision_config
    text_config = config.text_config

    pp_size = getattr(config, "pipeline_model_parallel_size", 1) or 1
    vpp_size = getattr(config, "virtual_pipeline_model_parallel_size", 1) or 1
    spatial_merge_size = getattr(config, "spatial_merge_size", config.vision_config.spatial_merge_size)

    # --- Build vision model via Qwen3_5VisionProvider ---
    vision_provider = Qwen3_5VisionProvider.from_config(vision_config)
    # ``use_accuracy_compatible`` is declared on the top-level / text config, but
    # the ViT is built from its own provider that never saw the flag, so every
    # accuracy-compatible branch inside the vision tower (parallel linears, SDPA,
    # MLP) silently stayed off.
    vision_provider.use_accuracy_compatible = getattr(vision_config, "use_accuracy_compatible", False) or getattr(
        config, "use_accuracy_compatible", False
    )
    # The reference vision RoPE always rotates in FP32 (``q, k = q.float(), k.float()``
    # then casts back), which is what ``high_precision_rope`` selects — see
    # ``_apply_rotary_pos_emb_bshd``. It has to be forced here rather than declared
    # as a provider field default: ``from_config`` copies every attribute off the
    # HF config, and the generic ``PretrainedConfig`` carries
    # ``high_precision_rope=False``, which would clobber the field default.
    vision_provider.high_precision_rope = True
    vision_provider.gated_linear_unit = False
    vision_model = vision_provider.provide()

    # --- Build language model via Qwen3_5TextModelProvider ---
    language_config = Qwen3_5TextModelProvider.from_config(text_config)
    # The reference router unconditionally renormalizes the top-k weights
    # (``router_top_value /= router_top_value.sum(-1, keepdim=True)``) and never
    # reads ``norm_topk_prob``, so the ``norm_topk_prob=false`` carried by the
    # official config.json does not describe this model. Honoring it would leave
    # the routing weights unnormalized, which changes the magnitude of the routed
    # branch output.
    language_config.norm_topk_prob = True
    # Propagate parallelism settings
    language_config.pipeline_model_parallel_size = pp_size
    language_config.virtual_pipeline_model_parallel_size = vpp_size
    language_config.tensor_model_parallel_size = getattr(config, "tensor_model_parallel_size", 1) or 1
    language_config.context_parallel_size = getattr(config, "context_parallel_size", 1) or 1
    language_config.expert_model_parallel_size = getattr(config, "expert_model_parallel_size", 1) or 1
    language_config.sequence_parallel = getattr(config, "sequence_parallel", False)
    # Propagate multimodal settings
    language_config.multimodal_embedding = True
    language_config.image_token_id = config.image_token_id
    language_config.video_token_id = config.video_token_id

    if pp_size > 1:
        # ===== Pipeline Parallel path =====
        # MTP trains normally here: GPTEmbedding performs the multimodal merge
        # before the MTP split, so the shifted MTP embeddings carry the visual
        # features. enable_mtp_magic_send must stay off — it re-embeds input_ids
        # on the last stage, where the vision features do not exist.
        # Use GPTModelProvider.provide() which internally calls gpt_builder
        # with num_stages=pp_size and handles VPP automatically.
        # The returned model IS a GPTModel (PipelineLayer) — trainer requires this.
        # Pass criterion as loss_fn so PipelineLayer scheduler can compute loss on last stage.
        # Raise instead of assert: with ``python -O`` the assertion is stripped
        # and the run would silently re-embed input_ids on the last stage, where
        # the vision features do not exist.
        if getattr(language_config, "enable_mtp_magic_send", False):
            raise ValueError(
                "enable_mtp_magic_send re-embeds input_ids on the last pipeline "
                "stage and is incompatible with multimodal inputs."
            )
        language_model = language_config.provide(loss_fn=criterion)

        from paddlefleet import parallel_state

        is_first_stage = parallel_state.is_pipeline_first_stage()

        # Attach vision merge layer on first stage and insert into run_function
        # so it runs BEFORE GPTEmbedding in the PipelineLayer forward pass.
        if is_first_stage:
            vision_merge = Qwen3_5VisionMergeLayer(vision_model=vision_model)
            # Layer.__setattr__ already registers a Layer value in _sub_layers
            # under this name, so no add_sublayer call is needed.
            language_model.vision_merge = vision_merge
            # Insert vision_merge at position 0 in run_function (before GPTEmbedding)
            # For VPP, only insert into the first chunk (chunk 0)
            if hasattr(language_model, "_model_chunks") and language_model._model_chunks:
                # VPP mode: insert into first chunk's run_function
                first_chunk = language_model._model_chunks[0]
                first_chunk.run_function.insert(0, vision_merge)
            else:
                # 1F1B mode: insert into language_model.run_function directly
                language_model.run_function.insert(0, vision_merge)
        else:
            language_model.vision_merge = None

        # Attach _prepare_pipeline_inputs_func for the trainer
        language_model._prepare_pipeline_inputs_func = _prepare_qwen3_5_pipeline_inputs

        # mm_collate_fn resolves the mRoPE function as model.get_rope_index /
        # model.model.get_rope_index. Without this the collator finds nothing on
        # the bare GPTModel, silently drops position_ids from the batch, and the
        # model degrades to plain sequential positions instead of mRoPE.
        language_model.get_rope_index = _Qwen3_5RopeIndexHelper(
            spatial_merge_size,
            config.image_token_id,
            config.video_token_id,
        ).get_rope_index

        # Attach config for downstream use
        language_model.config_to_save = config

        # GPTModel is not a PretrainedModel, so provide the HF-export entry point
        # the trainer calls at the end of training.
        language_model.save_pretrained = types.MethodType(_pp_save_pretrained, language_model)

        return language_model
    else:
        # ===== Original non-PP path =====
        language_spec = get_qwen3_5_language_spec(language_config)
        language_model = build_spec_layer(
            language_spec,
            seg_method="layer:TransformerLayer|EmptyLayer",
            num_stages=1,
        )

        strategy = fleet.DistributedStrategy()

        model = Qwen3_5Model(
            config=language_config,
            vision_model=NoPipelineParallel(vision_model, strategy),
            language_model=NoPipelineParallel(language_model, strategy),
            spatial_merge_size=spatial_merge_size,
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

        # Only the batch and sequence extents are needed here, and the embedding
        # lookup now lives inside GPTEmbedding, so inputs_embeds is normally
        # unavailable; fall back to input_ids, which carries the same two dims.
        shape_source = inputs_embeds if inputs_embeds is not None else input_ids
        if self.rope_deltas is not None and shape_source is not None:
            batch_size, seq_length = shape_source.shape[0], shape_source.shape[1]
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
        """Run the vision encoder, then delegate everything else to GPTEmbedding.

        ``input_ids`` is passed through untouched so that GPTEmbedding takes its
        ``decoder_input is None`` branch, which is the same branch PP stage 0
        takes. That branch owns the embedding lookup, the padding zero-fill and
        the MoE routing mask, the multimodal merge and the MTP split — all in the
        right order. Re-implementing any of it here is what used to make the two
        parallel layouts produce different numbers.
        """
        input_ids = dict_args.get("input_ids", None)
        pixel_values = dict_args.get("pixel_values", None)
        pixel_values_videos = dict_args.get("pixel_values_videos", None)
        image_grid_thw = dict_args.get("image_grid_thw", None)
        video_grid_thw = dict_args.get("video_grid_thw", None)
        attention_mask = dict_args.get("attention_mask", None)
        position_ids = dict_args.get("position_ids", None)
        mm_token_type_ids = dict_args.get("mm_token_type_ids", None)
        past_key_values = dict_args.get("past_key_values", None)

        if pixel_values is not None and self.visual is not None:
            dict_args["image_embeds"] = self.get_image_features(pixel_values, image_grid_thw)

        if pixel_values_videos is not None and self.visual is not None:
            dict_args["video_embeds"] = self.get_video_features(pixel_values_videos, video_grid_thw)

        if position_ids is None:
            # Normally the collator supplies mRoPE position_ids via
            # get_rope_index; this is the generation / no-collator fallback.
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=dict_args.get("inputs_embeds", None),
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )
            if position_ids is None:
                raise ValueError(
                    "Could not derive position_ids: the collator did not supply "
                    "them and compute_3d_position_ids has neither the mRoPE "
                    "inputs (input_ids + mm_token_type_ids + a grid_thw) nor a "
                    "cached rope_deltas to extend."
                )

        dict_args["position_ids"] = position_ids

        lm_dict_args = self.language_embedding(dict_args)

        for layer in self.language_backbone:
            lm_dict_args = layer(lm_dict_args)

        if self.language_lm_head is not None:
            logits = self.language_lm_head(lm_dict_args)
            return logits

        return lm_dict_args


class FleetQwen3_5ForConditionalGeneration(FleetLayer, PretrainedModel):
    config_class = None

    def _post_init(self, original_init, *args, **kwargs):
        pass

    def __init__(self, config, model, criterion):
        super().__init__(config)
        self.model = model
        self.criterion = criterion

    def forward(self, dict_args=None, **kwargs):
        if dict_args is None:
            dict_args = kwargs
        labels = dict_args.get("labels", None)
        logits = self.model(dict_args)
        if isinstance(logits, list):
            mtp_logits = logits[1:]
            logits = logits[0]
            loss = self.criterion(logits, labels, mtp_logits=mtp_logits)
        else:
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
            language_model = self.model.language_model._layers
            if hasattr(language_model, "sharded_state_dict"):
                lm_sharded = language_model.sharded_state_dict(structured_name_prefix="")
                sharded_state_dict.update(lm_sharded)

        # Get sharded state dict from vision model (Qwen3_5VisionModel wrapped in NoPipelineParallel)
        if self.model.visual is not None:
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


# ======================================================================
# Pipeline Parallel support classes
# ======================================================================


class Qwen3_5VisionMergeLayer(paddle.nn.Layer):
    """PP stage-0 layer that runs the vision encoder.

    Only instantiated on PP stage 0, where it is inserted at position 0 of the
    chunk's ``run_function`` so it runs before ``GPTEmbedding``. It encodes
    pixel values and hands the features over as ``image_embeds`` /
    ``video_embeds``; the merge into the text embeddings and the mRoPE
    ``position_ids`` are deliberately *not* done here (see ``forward``).
    """

    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def encode(self, pixel_values, grid_thw):
        """Run the vision encoder. Images and videos share the same tower."""
        dict_input = {"pixel_values": pixel_values, "grid_thw": grid_thw}
        output = self.vision_model.forward(dict_input)
        if isinstance(output, tuple):
            return output[0]
        return output

    def forward(self, dict_args):
        """Run the vision encoder and inject its output for GPTEmbedding.

        This layer runs BEFORE GPTEmbedding in the PP run_function chain. It
        extracts pixel_values, runs the vision encoder, and puts the resulting
        image_embeds/video_embeds into dict_args so that GPTEmbedding's
        multimodal_embedding path can merge them.

        ``position_ids`` are deliberately not computed here: the collator
        already produces the mRoPE ``[3, B, S]`` tensor via the model's
        ``get_rope_index``, and duplicating that computation is what made the PP
        and non-PP paths disagree.
        """
        pixel_values = dict_args.get("pixel_values", None)
        image_grid_thw = dict_args.get("image_grid_thw", None)
        pixel_values_videos = dict_args.get("pixel_values_videos", None)
        video_grid_thw = dict_args.get("video_grid_thw", None)

        if pixel_values is not None:
            dict_args["image_embeds"] = self.encode(pixel_values, image_grid_thw)

        if pixel_values_videos is not None:
            dict_args["video_embeds"] = self.encode(pixel_values_videos, video_grid_thw)

        # Raise instead of assert: with ``python -O`` the assertion is stripped
        # and the run degrades to plain sequential positions instead of mRoPE,
        # which is a silent accuracy loss rather than a crash.
        if dict_args.get("position_ids", None) is None:
            raise ValueError(
                "position_ids must be supplied by the data collator. It resolves "
                "get_rope_index off the model; build_qwen3_5_model attaches that "
                "method to the PP model for exactly this reason."
            )

        # Remove large vision tensors to save P2P communication bandwidth
        # (they're no longer needed after encoding)
        for key in ["pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "mm_token_type_ids"]:
            dict_args.pop(key, None)

        return dict_args


class _Qwen3_5RopeIndexHelper:
    """Standalone holder for ``get_rope_index``.

    ``mm_collate_fn`` resolves the mRoPE function as ``model.get_rope_index`` or
    ``model.model.get_rope_index``. The non-PP model exposes it through
    ``Qwen3_5Model``, but the PP model is a bare ``GPTModel``: without this the
    collator finds nothing, silently drops ``position_ids`` from the batch, and
    the model falls back to plain sequential positions instead of mRoPE.

    ``image_token_id`` / ``video_token_id`` are required as well, not just
    ``spatial_merge_size``: the collator reads them off this object
    (``get_rope_func.__self__``) to build ``mm_token_type_ids``. Without them the
    token-type map stays all-zero, every token looks like text, and
    ``get_rope_index`` degrades to plain sequential positions.
    """

    get_vision_position_ids = Qwen3_5Model.get_vision_position_ids
    get_rope_index = Qwen3_5Model.get_rope_index

    def __init__(self, spatial_merge_size, image_token_id, video_token_id):
        self.spatial_merge_size = spatial_merge_size
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
