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

"""HF-style config for the HyperBody decoder -- the **single source of truth for geometry and numeric switches**.

## Split of responsibilities

| Side | Responsibility |
|---|---|
| **This file (PaddleFormers)** | Geometry constants + ``HyperBodyDecoderConfig``: the on-disk shape of ``config.json``, fed to ``AutoConfig`` / ``AutoModelForCausalLM`` |
| ``modeling.py`` (PaddleFormers) | HF-style config -> ``GPTConfig`` conversion (``HyperBodyDecoderModelProvider``) + model assembly |
| ``paddlefleet.models.hyperbody_decoder`` | Provides components only: the ``LayerSpec`` list for the backbone |

The "HF-style config -> GPTConfig" step is done by ``AutoConfig`` -> this class ->
``HyperBodyDecoderModelProvider.from_config``, all landing in PaddleFormers.

## Fields with no Paddle-side counterpart

* ``persist_layer_norm`` -- Paddle's norm has no such switch.
* ``moe_router_dtype="fp32"`` -- Paddle's router already forces fp32 (as long as
  ``use_accuracy_compatible`` is off).
* ``moe_router_pre_softmax`` -- Paddle routing is natively pre-softmax; pairing
  ``scoring_func="softmax"`` + ``topk_method="greedy"`` + ``norm_topk_prob=False``
  is equivalent.
* attention backend -- corresponds to ``_attn_implementation`` on the Paddle side.

## WARNING: which fields get overwritten by the CLI (must know when writing yaml)

The ``paddleformers-cli`` chain is:

1. ``AutoConfig.from_pretrained`` -> goes through this class's ``__init__``;
2. ``set_expected_keys`` inside ``PretrainedConfig.__init__``
   (``configuration_utils.py:889``) writes **every** key of
   ``LlmMetaConfig._get_init()`` onto config with its own default -- so any field
   set as ``self.x = ...`` before ``super().__init__()`` in this class's ``__init__``
   gets reset to default if it belongs to llm_meta. **Fix: put it into the
   ``super().__init__(**)`` kwargs**, since ``set_expected_keys`` prefers kwargs
   (``:206-207``).
3. ``LlmMetaConfig.set_llm_config(model_config, training_args)``
   (``cli/train/sft/workflow.py:283``) runs again, this time taking values from
   ``SFTConfig`` -- and ``SFTConfig`` is decorated by ``@llmmetaclass``
   (``cli/train/sft/sft_config.py:28``), so llm_meta fields on it are all dataclass
   fields. **config.json cannot block this step.**

Therefore the following fields, whose intended values **differ** from llm_meta
defaults, must also be written into the yaml, otherwise they get silently changed:

| Field | Intended | llm_meta default |
|---|---|---|
| ``moe_expert_fusion`` | ``True`` | ``False`` |
| ``router_aux_loss_coef`` | ``0.001`` | ``0.0`` |
| ``fp32_residual_connection`` | ``False`` | ``True`` |

All three are explicitly written in ``yaml/hyperbody_decoder_*.yaml``; do not delete them.

Key fields that are NOT llm_meta and are therefore **owned by this class**:
``masked_softmax_fusion``, ``bias_activation_fusion``, ``bias_dropout_fusion``,
``cross_entropy_loss_fusion``, ``attention_softmax_in_fp32``,
``calculate_per_token_loss``, ``rms_norm_eps``, ``rope_theta``, ``rotary_percent``,
``head_dim``, ``moe_layer_freq``, ``n_shared_experts``, ``moe_intermediate_size``,
``scoring_func``, ``topk_method``, ``norm_topk_prob``, ``routed_scaling_factor``,
``n_group``, ``topk_group``, ``use_cpu_initialization``, ``use_accuracy_compatible``,
``bf16``.
"""

from __future__ import annotations

from ..configuration_utils import PretrainedConfig

__all__ = [
    "HyperBodyDecoderConfig",
    "CONTEXT_TOKEN",
    "HYPERBODY_DECODER_VOCAB_SIZE",
    "HYPERBODY_DECODER_HIDDEN_SIZE",
    "HYPERBODY_DECODER_NUM_LAYERS",
    "HYPERBODY_DECODER_NUM_HEADS",
    "HYPERBODY_DECODER_FFN_HIDDEN",
    "HYPERBODY_DECODER_MOE_FFN_HIDDEN",
    "HYPERBODY_DECODER_NUM_MOE_EXPERTS",
    "HYPERBODY_DECODER_MOE_TOPK",
    "HYPERBODY_DECODER_NUM_SHARED_EXPERTS",
    "build_moe_layer_freq",
    "check_hyperbody_decoder_divisibility",
]

# Special token: wherever <context> appears in input_ids, the LLM splices the
# encoder output into the embedding stream. Only used in the full HyperBody
# multimodal setup; unused when the decoder is trained standalone.
CONTEXT_TOKEN = 128830

HYPERBODY_DECODER_VOCAB_SIZE = 129280
HYPERBODY_DECODER_HIDDEN_SIZE = 1280
HYPERBODY_DECODER_NUM_LAYERS = 12
HYPERBODY_DECODER_NUM_HEADS = 10
HYPERBODY_DECODER_FFN_HIDDEN = 6848  # dense FFN, layer 0 only
HYPERBODY_DECODER_MOE_FFN_HIDDEN = 896
HYPERBODY_DECODER_NUM_MOE_EXPERTS = 64
HYPERBODY_DECODER_MOE_TOPK = 6
# Shared-expert intermediate size 1792 is expressed as
# n_shared_experts x moe_intermediate_size: 2 x 896 = 1792.
HYPERBODY_DECODER_NUM_SHARED_EXPERTS = 2


def build_moe_layer_freq(num_hidden_layers: int = HYPERBODY_DECODER_NUM_LAYERS) -> list[int]:
    """Per-layer dense/MoE 0-1 table: ``[0] + [1]*(L-1)`` (layer 0 dense, rest MoE).

    WARNING: must be a list. Passing an int makes Paddle take ``i % N``
    (``gpt_layer_specs.py:803-807``), which is not the intended per-layer semantics.
    """
    if num_hidden_layers < 1:
        raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")
    return [0] + [1] * (num_hidden_layers - 1)


def check_hyperbody_decoder_divisibility(config: "HyperBodyDecoderConfig") -> None:
    """Parallelism divisibility check.

    Checked here early to fail **before** building the model, rather than waiting
    until some GEMM shape does not line up. Reads the real geometry off config, not
    the module constants -- a smoke config.json may use a shrunken geometry.
    """
    tp_size = max(getattr(config, "tensor_model_parallel_size", 1) or 1, 1)
    ep_size = max(getattr(config, "expert_model_parallel_size", 1) or 1, 1)
    for name in ("hidden_size", "num_attention_heads", "intermediate_size", "moe_intermediate_size"):
        value = getattr(config, name)
        if value % tp_size != 0:
            raise ValueError(f"decoder {name}={value} must be divisible by TP={tp_size}")
    if config.n_routed_experts % ep_size != 0:
        raise ValueError(f"decoder n_routed_experts={config.n_routed_experts} must be divisible by EP={ep_size}")


class HyperBodyDecoderConfig(PretrainedConfig):
    r"""HyperBody's LLM backbone (DeepSeekV2-Lite MoE, 12 layers / 64 experts / top-6).

    ``config.json`` only needs ``{"model_type": "hyperbody_decoder", "architectures":
    [...]}`` plus whatever fields you want to change from the defaults.

    Args:
        moe_layer_freq: per-layer dense/MoE 0-1 table. When ``None``, generated by
            ``build_moe_layer_freq(num_hidden_layers)`` as ``[0] + [1]*(L-1)``.
            WARNING: passing an int makes Paddle take ``i % N``
            (``gpt_layer_specs.py:803-807``), which is not the intended per-layer
            semantics.
        first_k_dense_replace: exists only to explicitly declare it **must be
            ``None``** -- ``TransformerConfig.__post_init__`` (``:2275-2297``) raises
            outright when it and a non-int ``moe_layer_freq`` are both given.
        use_cpu_initialization: **deliberately ``False``**. Two reasons:
            1. Paddle's cpu-init path is broken under bf16:
               ``tensor_parallel/layers.py:1849`` calls
               ``_initialize_affine_weight_cpu`` **without forwarding
               ``params_dtype``**, so the master weight stays at that function's
               ``paddle.float32`` default, and at ``:269`` the
               ``weight.copy_(cpu_weight)`` blows up with ``dtype 16 != 10``
               (``VocabParallelEmbedding``'s path forwards it, so only
               Column/RowParallelLinear trigger it).
            2. The "TP invariance" that CPU init buys is not needed here.
            Cost: the model must be built after the fleet RNG tracker is ready --
            distributed launch (``paddleformers-cli``) does this, but a bare script
            must first call
            ``paddlefleet.tensor_parallel.random.model_parallel_cuda_manual_seed(seed)``,
            otherwise it reports ``cuda rng state model-parallel-rng is not added``.
    """

    model_type = "hyperbody_decoder"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # === structure ===
        vocab_size: int = HYPERBODY_DECODER_VOCAB_SIZE,
        hidden_size: int = HYPERBODY_DECODER_HIDDEN_SIZE,
        num_hidden_layers: int = HYPERBODY_DECODER_NUM_LAYERS,
        num_attention_heads: int = HYPERBODY_DECODER_NUM_HEADS,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        intermediate_size: int = HYPERBODY_DECODER_FFN_HIDDEN,
        hidden_act: str = "silu",
        gated_linear_unit: bool = True,
        multi_latent_attention: bool = False,
        use_qk_norm: bool = False,
        normalization: str = "RMSNorm",
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        # === position embedding ===
        position_embedding_type: str = "rope",
        rope_theta: float = 10000,
        rotary_percent: float = 1.0,
        max_position_embeddings: int = 8192,
        # === dropout / bias ===
        attention_dropout: float = 0.0,
        hidden_dropout_prob: float = 0.0,
        use_bias: bool = False,
        attention_bias: bool = False,
        # === MoE ===
        n_routed_experts: int = HYPERBODY_DECODER_NUM_MOE_EXPERTS,
        moe_intermediate_size: int = HYPERBODY_DECODER_MOE_FFN_HIDDEN,
        num_experts_per_tok: int = HYPERBODY_DECODER_MOE_TOPK,
        n_shared_experts: int = HYPERBODY_DECODER_NUM_SHARED_EXPERTS,
        moe_layer_freq: list[int] | None = None,
        first_k_dense_replace: None = None,
        moe_token_dispatcher_type: str = "alltoall",
        moe_expert_fusion: bool = True,
        n_group: int = 1,
        topk_group: int = 1,
        router_aux_loss_coef: float = 0.001,
        scoring_func: str = "softmax",
        topk_method: str = "greedy",
        norm_topk_prob: bool = False,
        moe_router_load_balancing_type: str = "seq_aux_loss",
        moe_shared_expert_overlap: bool = True,
        routed_scaling_factor: float = 1.0,
        routed_scaling_factor_learnable: bool = False,
        # === dtype / numerics ===
        bf16: bool = True,
        attention_softmax_in_fp32: bool = False,
        fp32_residual_connection: bool = False,
        variable_seq_lengths: bool = True,
        calculate_per_token_loss: bool = False,
        # === fusion switches ===
        bias_activation_fusion: bool = True,
        masked_softmax_fusion: bool = True,
        bias_dropout_fusion: bool = True,
        apply_rope_fusion: bool = False,
        cross_entropy_loss_fusion: bool = False,
        # === initialization ===
        use_cpu_initialization: bool = False,
        use_accuracy_compatible: bool = False,
        # === parallelism (overridden at runtime by yaml) ===
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        virtual_pipeline_model_parallel_size: int = 1,
        expert_model_parallel_size: int = 1,
        context_parallel_size: int = 1,
        sequence_parallel: bool = False,
        pp_seg_method: str = "layer:TransformerLayer|EmptyLayer",
        # === misc ===
        tie_word_embeddings: bool = False,
        hyperbody_context_token_id: int = CONTEXT_TOKEN,
        **kwargs,
    ):
        # ---- structure ----
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        # num_key_value_heads == num_attention_heads => pure MHA, not GQA
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        # ---- position embedding ----
        self.rope_theta = rope_theta
        self.rotary_percent = rotary_percent
        # seq_length and max_position_embeddings share the same value. On the CLI
        # path all three get overwritten by data_args.max_seq_len
        # (cli/train/sft/workflow.py:338-339).
        self.max_position_embeddings = max_position_embeddings
        self.max_sequence_length = max_position_embeddings
        self.seq_length = max_position_embeddings

        # ---- dropout / bias ----
        self.attention_dropout = attention_dropout
        self.hidden_dropout_prob = hidden_dropout_prob
        self.use_bias = use_bias
        self.attention_bias = attention_bias

        # ---- MoE ----
        self.n_routed_experts = n_routed_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.moe_layer_freq = (
            build_moe_layer_freq(num_hidden_layers) if moe_layer_freq is None else list(moe_layer_freq)
        )
        self.first_k_dense_replace = first_k_dense_replace
        self.n_group = n_group
        self.topk_group = topk_group
        # pre-softmax routing expressed via the three fields below
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.routed_scaling_factor_learnable = routed_scaling_factor_learnable

        # ---- dtype / numerics ----
        self.bf16 = bf16
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.variable_seq_lengths = variable_seq_lengths
        self.calculate_per_token_loss = calculate_per_token_loss

        # ---- fusion switches (none belong to llm_meta, so this class owns them) ----
        self.bias_activation_fusion = bias_activation_fusion
        self.masked_softmax_fusion = masked_softmax_fusion
        self.bias_dropout_fusion = bias_dropout_fusion
        self.cross_entropy_loss_fusion = cross_entropy_loss_fusion

        # ---- initialization ----
        self.use_cpu_initialization = use_cpu_initialization
        self.use_accuracy_compatible = use_accuracy_compatible

        self.pp_seg_method = pp_seg_method
        # HyperBody-specific: the LLM splices the encoder output into the embedding
        # stream at the <context> position. Not used when the decoder is trained
        # standalone.
        self.hyperbody_context_token_id = hyperbody_context_token_id

        # WARNING: the following all belong to LlmMetaConfig and must go through kwargs
        # so they are not reset to defaults by set_expected_keys (see module docstring step 2).
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            multi_latent_attention=multi_latent_attention,
            use_qk_norm=use_qk_norm,
            normalization=normalization,
            gated_linear_unit=gated_linear_unit,
            position_embedding_type=position_embedding_type,
            fp32_residual_connection=fp32_residual_connection,
            apply_rope_fusion=apply_rope_fusion,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_expert_fusion=moe_expert_fusion,
            moe_router_load_balancing_type=moe_router_load_balancing_type,
            moe_shared_expert_overlap=moe_shared_expert_overlap,
            router_aux_loss_coef=router_aux_loss_coef,
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            context_parallel_size=context_parallel_size,
            sequence_parallel=sequence_parallel,
            **kwargs,
        )
