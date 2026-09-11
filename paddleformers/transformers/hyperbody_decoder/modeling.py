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

"""HyperBody decoder: "HF-style config -> ``GPTConfig``" conversion + model assembly.

## Split of responsibilities

| Side | Contents |
|---|---|
| ``paddlefleet.models.hyperbody_decoder`` | **Components**: ``get_hyperbody_decoder_layer_specs`` |
| ``configuration.py`` (this package) | Geometry constants + ``HyperBodyDecoderConfig`` (HF-style config source of truth) |
| **This file** | (1) HF-style config -> ``GPTConfig`` (:class:`HyperBodyDecoderModelProvider`) (2) model assembly (:func:`build_hyperbody_decoder_model`) |

## Why assembly is written out here instead of using ``gpt_builder`` directly

``paddlefleet.gpt_builders.gpt_builder`` calls ``get_gpt_decoder_layers_spec`` itself
when ``n_routed_experts`` is set, with **no hook to inject a layer spec**. To keep the
"fleet provides components, formers does assembly" split, this file calls
``get_gpt_spec`` + ``build_spec_layer`` directly and feeds in the fleet spec.

:func:`build_hyperbody_decoder_model` is a narrowed version of ``gpt_builder``: it
keeps only the path this model actually uses (embedding + N-layer backbone + norm +
lm_head + LanguageLoss); every other branch (MTP, head/tail EmptyLayer,
``separate_mtp_headloss``, ringmoe subgroups, meta-device init) is **explicitly
rejected** rather than silently skipped, so turning on one of those switches later
fails loudly instead of silently.

## ``__new__`` instead of ``__init__``

``AutoModelForCausalLM.from_config`` runs ``with dtype_guard(dtype): model =
cls(config)`` (``model_utils.py:1349-1368``), but what we need to return is a fleet
``GPTModel`` (a ``PipelineLayer``), not an instance of this class. So ``__new__``
returns that fleet ``GPTModel`` directly instead of leaving ``__init__`` to populate
an instance of this class.

## Why ``_gen_aoa_config`` cannot be skipped

aoa on the surface only serves **HF-format interconversion**, and normal
flex_checkpoint archives never touch it. But the **CLI does one HF export after
training**: ``cli/train/sft/workflow.py:793`` hard-codes ``last_fc_to_hf=True`` with
no yaml switch, so missing aoa raises ``RuntimeError: ... must implement either the
_gen_inv_aoa_config ...`` only after the whole training run finishes.

We write only the forward direction (HF -> fleet): ``model_utils.py:3286-3289`` adds
``aoa_config_reverse=True`` on its own to derive the save direction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from paddle.distributed.fleet.meta_parallel import build_spec_layer
from paddlefleet.models.common.language_loss.language_loss import LanguageLoss
from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_spec
from paddlefleet.models.hyperbody_decoder import get_hyperbody_decoder_layer_specs

from ...nn.pp_model import CriterionLayerPipe, GeneralModelForCausalLMPipe
from ..gpt_provider import GPTModel, GPTModelProvider
from ..model_utils import PretrainedModel
from .configuration import HyperBodyDecoderConfig, check_hyperbody_decoder_divisibility

logger = logging.getLogger(__name__)

__all__ = [
    "HyperBodyDecoderForCausalLM",
    "HyperBodyDecoderForCausalLMPipe",
    "HyperBodyDecoderModelProvider",
    "build_hyperbody_decoder_model",
]


def build_hyperbody_decoder_model(config, *, num_stages: int, loss_fn=None):
    """Assembly: fleet layer spec -> ``get_gpt_spec`` -> ``build_spec_layer``.

    This is a narrowed version of ``paddlefleet.gpt_builders.gpt_builder``. The
    ``config`` argument is an already-converted ``GPTConfig`` (i.e.
    :class:`HyperBodyDecoderModelProvider` itself).

    Args:
        num_stages: number of pipeline stages, equal to ``pipeline_model_parallel_size``.
        loss_fn: defaults to ``LanguageLoss(config)`` (matching ``gpt_builder``).
    """
    # These branches are unused by this model; silently skipping would fail silently later.
    if getattr(config, "mtp_num_layers", None):
        raise NotImplementedError("HyperBody decoder has no MTP layers.")
    if getattr(config, "separate_mtp_headloss", False):
        raise NotImplementedError("HyperBody decoder does not use separate_mtp_headloss.")
    if config.num_empty_layers_add_in_head or config.num_empty_layers_add_in_tail:
        raise NotImplementedError("HyperBody decoder inserts no EmptyLayer (pp split relies on seg_method).")
    if getattr(config, "moe_token_dispatcher_type", None) == "ringmoe":
        raise NotImplementedError("ringmoe needs world-level subgroup init, not supported by this model.")
    if getattr(config, "init_model_with_meta_device", False):
        raise NotImplementedError("HyperBody decoder does not use meta-device init.")

    gpt_spec = get_gpt_spec(
        config=config,
        head_empty_layers_spec=[],
        # The only line that differs from gpt_builder: spec comes from the fleet hyperbody_decoder component.
        transformer_layers_spec=get_hyperbody_decoder_layer_specs(config),
        tail_empty_layers_spec=[],
        mtp_layers_spec=None,
        vocab_size=config.vocab_size,
        tie_word_embeddings=config.tie_word_embeddings,
        max_sequence_length=config.max_sequence_length,
        position_embedding_type=config.position_embedding_type,
        rotary_percent=config.rotary_percent,
        rotary_base=config.rope_theta,
        swa_rotary_base=config.swa_rope_theta,
        rope_scaling=config.rope_scaling,
        parallel_output=config.parallel_output,
    )
    return build_spec_layer(
        gpt_spec,
        loss_fn=LanguageLoss(config) if loss_fn is None else loss_fn,
        num_stages=num_stages,
        # Same split criterion as GPTModelProvider.provide().
        seg_method="layer:TransformerLayer|EmptyLayer",
    )


@dataclass
class HyperBodyDecoderModelProvider(GPTModelProvider):
    """Landing point for HF-style config -> ``GPTConfig`` (``GPTModelProvider`` is itself a ``GPTConfig``).

    Only the switches that **must be pinned** are listed here. Geometry (num layers /
    hidden / experts / topk ...) and other numerics are all injected by
    :class:`HyperBodyDecoderConfig` via ``TransformerConfig.register_attributes`` and
    are not redeclared here -- redeclaring would create a second source of truth,
    whose real home is ``configuration.py``.

    WARNING: ``from_config`` goes through ``object.__new__`` + ``register_attributes``
    (``transformer_config.py:1942-1948``) and **does not run the dataclass
    ``__init__``**. The defaults below still take effect because a dataclass field
    with no ``default_factory`` is a class attribute, and attribute lookup falls back
    to the class when the instance attribute is missing. So do not change them to
    ``field(default_factory=...)``.
    """

    # ---- attention: MLA fully off, pure MHA (Paddle must turn it off explicitly) ----
    multi_latent_attention: bool = False
    use_qk_norm: bool = False

    # ---- position embedding: plain RoPE, no scaling ----
    # WARNING: must explicitly write None back: ``GPTConfig`` overrides the inherited
    # ``rope_scaling: dict = None`` to ``float = 1.0`` (``gpt_config.py:32`` vs
    # ``transformer_config.py:485``), and ``gpt_provider.py:210`` only checks
    # ``is not None``, so ``1.0`` falls into the ``"mscale_all_dim" in
    # self.rope_scaling`` check at ``:211`` and blows up with
    # ``TypeError: argument of type 'float' is not iterable``.
    rope_scaling: dict = None

    # ---- FFN / norm ----
    gated_linear_unit: bool = True
    normalization: str = "RMSNorm"

    # ---- MoE ----
    moe_token_dispatcher_type: str = "alltoall"
    moe_router_load_balancing_type: str = "seq_aux_loss"
    moe_shared_expert_overlap: bool = True

    # ---- fusion switches ----
    bias_activation_fusion: bool = True
    masked_softmax_fusion: bool = True
    bias_dropout_fusion: bool = True
    apply_rope_fusion: bool = False
    cross_entropy_loss_fusion: bool = False

    # ---- misc ----
    # GPTModelProvider defaults tie_word_embeddings to True; not overriding would
    # silently tie embedding and lm_head.
    tie_word_embeddings: bool = False
    # Recompute disabled.
    recompute_granularity: str = None

    # ``dtype`` -> ``params_dtype`` is already handled by ``_process_attribute``
    # (``transformer_config.py:1981-1982``); writing it here explicitly only keeps
    # the same shape as other fleet providers in this repo, for easy comparison.
    transform_rules = {
        **GPTModelProvider.transform_rules,
        "dtype": "params_dtype",
    }

    def provide(self, pre_process=None, post_process=None, vp_stage=None, loss_fn=None) -> GPTModel:
        """Override the parent's ``provide()``, swapping assembly for :func:`build_hyperbody_decoder_model`.

        The parent's version calls ``gpt_builder`` (which picks its own layer spec);
        this model needs the spec provided by fleet ``hyperbody_decoder``, so it
        assembles by itself.

        Two other parent segments are dead code for this model and are not carried
        over: ``mtp_block_spec`` is computed but never passed to ``gpt_builder``, and
        ``is_pipeline_asymmetric`` is computed but never read.
        """
        # Parent's rope flattening: ``GPTConfig`` may carry a ``rope_parameters`` wrapper.
        if getattr(self, "rope_parameters", None):
            if self.rope_parameters.get("rope_type", "default") != "default":
                self.rope_type = self.rope_parameters["rope_type"]
            if "rope_theta" in self.rope_parameters:
                self.rope_theta = self.rope_parameters["rope_theta"]
        if isinstance(self.rope_scaling, dict) and "mscale_all_dim" in self.rope_scaling:
            self.mscale_all_dim = self.rope_scaling["mscale_all_dim"]

        fleet_model = build_hyperbody_decoder_model(
            self,
            num_stages=self.pipeline_model_parallel_size,
            loss_fn=loss_fn,
        )
        # Convert FleetGPTModel into formers' GPTModel so it inherits PretrainedModel's
        # methods (attribute-by-attribute copy, same technique as parent
        # gpt_provider.py:232-240).
        model = GPTModel.__new__(GPTModel)
        for attr_name in dir(fleet_model):
            if not attr_name.startswith("__"):
                try:
                    setattr(model, attr_name, getattr(fleet_model, attr_name))
                except Exception:  # read-only attribute / property, just skip
                    pass
        return model


class HyperBodyDecoderPretrainedModel(PretrainedModel):
    config_class = HyperBodyDecoderConfig
    base_model_prefix = "hyperbody_decoder"
    # The gate is fp32 on the fleet side (PaddleFleet's router forces fp32 as long as
    # use_accuracy_compatible=False), and the archive metadata is indeed float32.
    _keep_in_fp32_modules = ["mlp.gate.weight"]

    @classmethod
    def _gen_aoa_config(cls, config: HyperBodyDecoderConfig):
        """Forward mapping table from HF weight names to fleet structured names.

        The right-hand side matches the real archive:

        =========================================  ==================
        fleet structured name                       shape (smoke geometry)
        =========================================  ==================
        model.embedding.embed_tokens.weight        (V, H)
        model.layers.i.self_attn.qkv_proj.weight   (H, (nq+2nkv)*hd)
        model.layers.i.self_attn.o_proj.weight     (H, H)
        model.layers.i.mlp.up_gate_proj.weight     (H, 2I)   <- dense layer
        model.layers.i.mlp.down_proj.weight        (I, H)    <- dense layer
        model.layers.i.mlp.gate.weight             (E, H) fp32
        model.layers.i.mlp.experts.e.*             (H, 2mI) / (mI, H)
        model.layers.i.mlp.shared_experts.*        (H, 2*ns*mI) / (ns*mI, H)
        model.lm_head.weight                       (V, H)
        model.norm.weight                          (H,)
        =========================================  ==================

        ``^T`` is because HF's nn.Linear weight is ``(out, in)`` while Paddle is
        ``(in, out)``; ``embed_tokens`` / ``lm_head`` / ``gate`` / norm have the same
        order on both sides and carry no ``^T``.

        Three deliberate specifics of this model:

        * No ``gate.e_score_correction_bias`` -- that is a product of noaux_tc
          routing; this model uses ``scoring_func="softmax"`` + ``topk_method="greedy"``.
        * No ``self_attn.{q,k}_norm`` -- ``use_qk_norm=False``.
        * The dense-layer set is derived from the per-layer ``moe_layer_freq`` table,
          **not** from ``first_k_dense_replace`` (which must be ``None`` here).
        """
        # There are only the two ForCausalLM classes; fleet structured names uniformly
        # carry the "model." prefix.
        pd_root = "model"
        num_experts = config.n_routed_experts
        moe_layer_freq = config.moe_layer_freq

        statements = [
            f"model.embed_tokens.weight -> {pd_root}.embedding.embed_tokens.weight",
            f"model.norm.weight -> {pd_root}.norm.weight",
        ]
        # untie => the normal independent lm_head path; the tie branch is kept only for completeness.
        if config.tie_word_embeddings:
            statements.append(f"model.embed_tokens.weight -> {pd_root}.lm_head.weight")
        else:
            statements.append(f"lm_head.weight -> {pd_root}.lm_head.weight")

        for layer_idx in range(config.num_hidden_layers):
            hf = f"model.layers.{layer_idx}"
            pd = f"{pd_root}.layers.{layer_idx}"
            statements += [
                f"{hf}.input_layernorm.weight -> {pd}.input_layernorm.weight",
                f"{hf}.post_attention_layernorm.weight -> {pd}.post_attention_layernorm.weight",
                f"{hf}.self_attn.o_proj.weight^T -> {pd}.self_attn.o_proj.weight",
                f"{hf}.self_attn.q_proj.weight^T, {hf}.self_attn.k_proj.weight^T, "
                f"{hf}.self_attn.v_proj.weight^T -> {pd}.self_attn.qkv_proj.weight, fused_qkv, "
                f"num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}",
            ]

            if not moe_layer_freq[layer_idx]:
                # Dense layer (only layer 0 in this model): the intermediate_size FFN.
                statements += [
                    f"{hf}.mlp.gate_proj.weight^T, {hf}.mlp.up_proj.weight^T "
                    f"-> {pd}.mlp.up_gate_proj.weight, fused_ffn",
                    f"{hf}.mlp.down_proj.weight^T -> {pd}.mlp.down_proj.weight",
                ]
                continue

            statements += [
                # The gate has different dtypes on the two sides (bf16 vs fp32). It
                # must be written as the src_dtype+dst_dtype pair: a single dtype=...
                # is rejected outright by an assertion in aoa_engine.py:736 on the
                # reverse direction.
                f"{hf}.mlp.gate.weight -> {pd}.mlp.gate.weight, src_dtype='bfloat16',dst_dtype='float32'",
                f"{hf}.mlp.shared_experts.gate_proj.weight^T, {hf}.mlp.shared_experts.up_proj.weight^T "
                f"-> {pd}.mlp.shared_experts.up_gate_proj.weight, fused_ffn",
                f"{hf}.mlp.shared_experts.down_proj.weight^T -> {pd}.mlp.shared_experts.down_proj.weight",
                # Per expert: $EXPERT_ID is expanded by the aoa parser into
                # 0..num_experts-1. This uses axis=1 rather than fused_ffn -- a routed
                # expert's up/gate do not participate in TP sharding, so they are
                # concatenated directly along the last axis.
                f"{hf}.mlp.experts.$EXPERT_ID.gate_proj.weight^T, {hf}.mlp.experts.$EXPERT_ID.up_proj.weight^T "
                f"-> {pd}.mlp.experts.$EXPERT_ID.up_gate_proj.weight, axis=1",
                f"{hf}.mlp.experts.$EXPERT_ID.down_proj.weight^T -> {pd}.mlp.experts.$EXPERT_ID.down_proj.weight",
            ]

            if config.moe_expert_fusion:
                # grouped GEMM: stack the per-expert 2-D weights along axis=0 into a
                # 3-D [E, ...] tensor. The left-hand side is the already-mapped fleet
                # name, i.e. a two-stage mapping.
                w1 = ",".join(f"{pd}.mlp.experts.{e}.up_gate_proj.weight" for e in range(num_experts))
                w2 = ",".join(f"{pd}.mlp.experts.{e}.down_proj.weight" for e in range(num_experts))
                statements += [
                    f"{w1} -> {pd}.mlp.grouped_gemm_experts.weight1, axis=0",
                    f"{w2} -> {pd}.mlp.grouped_gemm_experts.weight2, axis=0",
                ]

        return {"aoa_statements": statements}


def _build_hyperbody_decoder(cls, config):
    """Shared body of the two ``__new__``: normalize HF config -> build provider -> assemble."""
    # Parallelism fallback: when yaml omits them, LlmMetaConfig may inject 0 or -1,
    # and a non-positive value would compute an empty pp split during assembly.
    for name in (
        "tensor_model_parallel_size",
        "context_parallel_size",
        "pipeline_model_parallel_size",
        "virtual_pipeline_model_parallel_size",
        "expert_model_parallel_size",
    ):
        setattr(config, name, max(getattr(config, name, 1) or 1, 1))

    # Key switches for the HyperBody decoder: force them back even if config.json /
    # yaml sets them wrong. Getting these three wrong raises no error -- it only
    # silently swaps the attention implementation or adds a QK norm.
    config.multi_latent_attention = False
    config.use_qk_norm = False
    config.gated_linear_unit = True

    # If config carries a non-dict rope_scaling (e.g. a stray 1.0 from config.json),
    # register_attributes would override the provider's None with it and re-trigger
    # gpt_provider.py:211. HyperBody has no RoPE scaling, so normalize it to None.
    if not isinstance(getattr(config, "rope_scaling", None), dict):
        config.rope_scaling = None

    check_hyperbody_decoder_divisibility(config)

    # The HF-style config -> GPTConfig step.
    provider = HyperBodyDecoderModelProvider.from_config(config)

    loss_fn = None
    if getattr(config, "dpo_config", None):
        loss_fn = CriterionLayerPipe(config, use_infohub=True)

    gpt_model = provider.provide(loss_fn=loss_fn)
    if not hasattr(config, "architectures"):
        config.architectures = [cls.__name__.replace("Pipe", "")]
    # provide() returns a fleet GPTModel, not an instance of this class, so aoa and
    # the fp32 whitelist must be attached manually (save_pretrained reads them off
    # this object). Only the forward table is attached: model_utils.py:3286-3289
    # derives the save direction on its own.
    gpt_model._gen_aoa_config = cls._gen_aoa_config
    gpt_model._keep_in_fp32_modules = cls._keep_in_fp32_modules
    # The Trainer reads config_to_save when archiving, not the provider's flattened config.
    gpt_model.config_to_save = config
    gpt_model.is_fleet = cls.is_fleet
    return gpt_model


class HyperBodyDecoderForCausalLM(HyperBodyDecoderPretrainedModel):
    is_fleet = True

    def __new__(cls, config):
        return _build_hyperbody_decoder(cls, config)


class HyperBodyDecoderForCausalLMPipe(HyperBodyDecoderPretrainedModel, GeneralModelForCausalLMPipe):
    is_fleet = True

    def __new__(cls, config):
        return _build_hyperbody_decoder(cls, config)
