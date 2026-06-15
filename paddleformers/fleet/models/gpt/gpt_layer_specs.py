# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddleformers.fleet.fusions.fused_bias_dropout import get_bias_dropout_add
from paddleformers.fleet.models.backends import BackendSpecProvider, LocalSpecProvider
from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
    MultimodalRotaryEmbedding,
    RotaryEmbedding,
)
from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding,
)
from paddleformers.fleet.models.common.language_loss.language_loss import (
    MTPLanguageLoss,
)
from paddleformers.fleet.models.gpt import GPTModel
from paddleformers.fleet.models.gpt.gpt_embedding import GPTEmbedding, GPTEmbeddingSpec
from paddleformers.fleet.models.gpt.gpt_model import GPTSublayersSpec
from paddleformers.fleet.models.gpt.lm_head import (
    GPTLMHead,
    GPTMainLMHead,
    GPTMTPLMHead,
)
from paddleformers.fleet.models.gpt.moe_layer_specs import (
    get_moe_layer_spec_for_backend,
)
from paddleformers.fleet.models.gpt.mtp_embedding_layer import MTPEmbeddingLayer
from paddleformers.fleet.transformer.attention import (
    SelfAttention,
    SelfAttentionSublayersSpec,
    SelfAttentionVHA,
    SelfAttentionVHASublayersSpec,
)
from paddleformers.fleet.transformer.block_attn_res import (
    BlockAttnRes,
    BlockAttnResSublayersSpec,
)
from paddleformers.fleet.transformer.csa_attention import (
    CompressedSparseAttention,
    CompressedSparseAttentionSublayersSpec,
    Compressor,
    CompressorSublayersSpec,
    CSAIndexer,
    CSAIndexerSublayersSpec,
)
from paddleformers.fleet.transformer.dsa_attention import (
    DSAIndexer,
    DSAIndexerSublayersSpec,
    DSAttention,
    DSAttentionSublayersSpec,
)
from paddleformers.fleet.transformer.dsv4_hybrid_attention import (
    DSv4HybridSelfAttention,
    DSv4HybridSelfAttentionSublayersSpec,
)
from paddleformers.fleet.transformer.enums import AttnMaskType
from paddleformers.fleet.transformer.gated_delta_net import (
    GatedDeltaNet,
    GatedDeltaNetSublayersSpec,
)
from paddleformers.fleet.transformer.identity_op import IdentityOp
from paddleformers.fleet.transformer.mlp import MLP, MLPSublayersSpec
from paddleformers.fleet.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSublayersSpec,
)
from paddleformers.fleet.transformer.multi_token_prediction import (
    get_mtp_layer_spec_for_backend,
)
from paddleformers.fleet.transformer.paddle_norm import L2Norm
from paddleformers.fleet.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSublayersSpec,
    TransformerLayerWithOverlap,
)

if TYPE_CHECKING:
    from paddleformers.fleet.transformer.transformer_config import TransformerConfig

from paddleformers.fleet.transformer.paddle_norm import (
    WrappedPaddleNorm,
    WrappedPaddleNormPipe,
)

LNImpl = WrappedPaddleNorm


def get_attention_spec(
    config: TransformerConfig,
    attention_layer_type: str,
    attn_mask_type: AttnMaskType = AttnMaskType.causal,
    is_mtp_layer: bool = False,
) -> LayerSpec:
    """Build the self_attn LayerSpec based on attention_layer_type.

    Args:
        config: Transformer configuration.
        attention_layer_type: ``"self_attention"`` for standard multi-head
            attention or ``"gated_delta_net"`` for the GDN linear-attention
            variant.
        attn_mask_type: Attention mask type (only used for SelfAttention).

    Returns:
        LayerSpec for the attention sublayer inside a TransformerLayer.
    """
    assert config is not None, "config must be specified."
    backend = LocalSpecProvider()

    # Standard RMSNorm for general use (MLA, etc.)
    if config.normalization == "RMSNorm":
        qk_norm_standard = backend.layer_norm(rms_norm=True, for_qk=True)
    else:
        qk_norm_standard = backend.layer_norm(rms_norm=False, for_qk=True)

    # Triton-optimized RMSNorm only for self_attention QK norm (head_dim=128)
    # MLA uses larger latent_dim (1536) which exceeds Triton kernel limit (≤1024)
    use_triton_qk_norm = config.normalization == "RMSNorm" and getattr(config, "qk_norm_fusion", False)
    if use_triton_qk_norm:
        from paddleformers.fleet.transformer.paddle_norm import WrappedRMSNormTriton

        qk_norm = WrappedRMSNormTriton
    else:
        qk_norm = qk_norm_standard

    use_qk_norm = getattr(config, "use_qk_norm", False)
    qk_l2_norm = getattr(config, "qk_l2_norm", False)
    gated_attention = getattr(config, "gated_attention", False)
    align_mode = getattr(config, "gpt_model_use_experimental_version", None)

    if attention_layer_type == "self_attention":
        if getattr(config, "use_vha_attention", False):
            return LayerSpec(
                layer=SelfAttentionVHA,
                extra_kwargs={
                    "attn_mask_type": attn_mask_type,
                    "is_mtp_layer": is_mtp_layer,
                },
                sublayers_spec=SelfAttentionVHASublayersSpec(
                    q_proj=backend.column_parallel_linear(),
                    k_proj=backend.column_parallel_linear(),
                    v_proj=backend.column_parallel_linear(),
                    gate_proj=backend.column_parallel_linear() if getattr(config, "gated_attention", False) else None,
                    qkv_proj=backend.column_parallel_linear(),
                    core_attention=backend.core_attention(),
                    o_proj=backend.row_parallel_linear(),
                    q_norm=(L2Norm if qk_l2_norm else (qk_norm if use_qk_norm else IdentityOp)),
                    k_norm=(L2Norm if qk_l2_norm else (qk_norm if use_qk_norm else IdentityOp)),
                ),
            )
        return LayerSpec(
            layer=SelfAttention,
            extra_kwargs={
                "attn_mask_type": attn_mask_type,
                "is_mtp_layer": is_mtp_layer,
            },
            sublayers_spec=SelfAttentionSublayersSpec(
                qkv_proj=backend.column_parallel_linear(),
                core_attention=backend.core_attention(),
                o_proj=backend.row_parallel_linear(),
                gate_proj=backend.column_parallel_linear() if gated_attention and align_mode else IdentityOp,
                q_norm=(L2Norm if qk_l2_norm else (qk_norm if use_qk_norm else IdentityOp)),
                k_norm=(L2Norm if qk_l2_norm else (qk_norm if use_qk_norm else IdentityOp)),
            ),
        )
    elif attention_layer_type == "gated_delta_net":
        gdn_extra_kwargs = {
            "conv_kernel_dim": getattr(config, "linear_conv_kernel_dim", 4),
            "key_head_dim": getattr(config, "linear_key_head_dim", 128),
            "value_head_dim": getattr(config, "linear_value_head_dim", 128),
            "num_key_heads": getattr(config, "linear_num_key_heads", 16),
            "num_value_heads": getattr(config, "linear_num_value_heads", 32),
        }
        return LayerSpec(
            layer=GatedDeltaNet,
            sublayers_spec=GatedDeltaNetSublayersSpec(
                in_proj=backend.column_parallel_linear(),
                out_norm=backend.layer_norm(
                    rms_norm=(config.normalization == "RMSNorm"),
                    for_qk=False,
                ),
                out_proj=backend.row_parallel_linear(),
            ),
            extra_kwargs=gdn_extra_kwargs,
        )
    elif attention_layer_type == "multi_latent_attention":
        assert qk_l2_norm is False, "qk_l2_norm is not supported with MLA."
        # Decide attention class: always MLASelfAttention (DSA is a pluggable core_attention)
        attn_cls = MLASelfAttention
        # Gated attention
        gated_attention = getattr(config, "gated_attention", False)

        # Decide core_attention: DSAttention if dsa_index_n_heads is configured, else standard
        use_dsa = config is not None and getattr(config, "dsa_index_n_heads", None) is not None

        if use_dsa:
            # DSA Indexer sublayers spec (duplicated linear, NOT tensor-parallel)
            dsa_indexer_sublayers = DSAIndexerSublayersSpec(
                linear_wq_b=backend.linear(),
                linear_wk=backend.linear(),
                k_norm=paddle.nn.LayerNorm,  # LayerNorm (not RMSNorm) for Indexer K
                linear_weights_proj=backend.linear(),
            )
            # DSAttention as core_attention (pluggable component)
            core_attention = LayerSpec(
                layer=DSAttention,
                sublayers_spec=DSAttentionSublayersSpec(
                    indexer=LayerSpec(
                        layer=DSAIndexer,
                        sublayers_spec=dsa_indexer_sublayers,
                    ),
                ),
            )
        else:
            # Standard core_attention
            core_attention = backend.core_attention()

        return LayerSpec(
            layer=attn_cls,
            extra_kwargs={
                "attn_mask_type": attn_mask_type,
                "is_mtp_layer": is_mtp_layer,
            },
            sublayers_spec=MLASelfAttentionSublayersSpec(
                q_proj=backend.column_parallel_linear(),
                q_a_proj=backend.column_parallel_linear(),
                q_b_proj=backend.column_parallel_linear(),
                kv_a_proj_with_mqa=backend.column_parallel_linear(),
                kv_b_proj=backend.column_parallel_linear(),
                core_attention=core_attention,
                o_proj=backend.row_parallel_linear(),
                q_a_layernorm=qk_norm_standard if use_qk_norm else IdentityOp,
                kv_a_layernorm=qk_norm_standard if use_qk_norm else IdentityOp,
                gate_proj=backend.column_parallel_linear() if gated_attention else None,
            ),
        )
    elif attention_layer_type == "dsv4_hybrid_attention":
        # Build nested CSA spec tree matching Megatron's structure
        compressor_spec = LayerSpec(
            layer=Compressor,
            sublayers_spec=CompressorSublayersSpec(
                linear_wkv=backend.linear(),
                linear_wgate=backend.linear(),
                norm=backend.layer_norm(rms_norm=True, for_qk=False),
            ),
        )

        indexer_spec = LayerSpec(
            layer=CSAIndexer,
            sublayers_spec=CSAIndexerSublayersSpec(
                linear_wq_b=backend.linear(),
                linear_weights_proj=backend.linear(),
                compressor=compressor_spec,
            ),
        )

        core_attention_spec = LayerSpec(
            layer=CompressedSparseAttention,
            sublayers_spec=CompressedSparseAttentionSublayersSpec(
                compressor=compressor_spec,
                indexer=indexer_spec,
            ),
        )

        # DSA indexer requires normalized q as input, so here we cannot fuse
        # qk layernorm with linear projection and have to use unfused qk layernorm.
        qk_norm = (
            backend.layer_norm(rms_norm=config.normalization == "RMSNorm", for_qk=True)
            if getattr(config, "qk_layernorm", True)
            else IdentityOp
        )

        return LayerSpec(
            layer=DSv4HybridSelfAttention,
            extra_kwargs={
                "attn_mask_type": attn_mask_type,
                "is_mtp_layer": is_mtp_layer,
            },
            sublayers_spec=DSv4HybridSelfAttentionSublayersSpec(
                linear_q_down_proj=backend.linear(),
                linear_q_up_proj=backend.column_parallel_linear(),
                linear_kv_proj=backend.column_parallel_linear(),
                core_attention=core_attention_spec,
                o_proj=backend.row_parallel_linear(),
                q_layernorm=qk_norm,
                kv_layernorm=qk_norm,
            ),
        )
    else:
        raise ValueError(
            f"Unknown attention_layer_type: {attention_layer_type!r}. "
            f"Expected 'self_attention' or 'gated_delta_net'."
        )


def get_gpt_layer_local_spec(
    config: TransformerConfig | None = None,
    num_experts: int | None = None,
    moe_expert_fusion: bool | None = False,
    use_qk_norm: bool | None = False,
    multi_latent_attention: bool | None = False,
    normalization: str | None = None,
    qk_l2_norm: bool | None = False,
    layer_number: int | None = 1,
    attention_layer_type: str = "self_attention",
    attn_mask_type: AttnMaskType = AttnMaskType.causal,
    is_mtp_layer: bool = False,
) -> LayerSpec:
    """Use this spec for an implementation using only layers in Fleet-Core.


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_expert_fusion (bool, optional): To use Grouped GEMM. Defaults to False.
        use_qk_norm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fp8 (str, optional): Deprecated. For temporary Nemo compatibility.
        qk_l2_norm (bool, optional): To use l2 norm for queries/keys. Defaults to False.
        attention_layer_type (str, optional): Type of attention layer.
            ``"self_attention"`` for standard multi-head attention,
            ``"gated_delta_net"`` for the GDN linear-attention variant.
            Defaults to ``"self_attention"``.

    Returns:
        LayerSpec: Layer specification with Fleet-Core layers
    """

    backend = LocalSpecProvider()
    # Adjust for RMS norm.
    if normalization == "RMSNorm":
        layer_norm = backend.layer_norm(rms_norm=True, for_qk=False)
    else:
        layer_norm = backend.layer_norm(rms_norm=False, for_qk=False)

    mlp = get_mlp_layer_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_expert_fusion=moe_expert_fusion,
    )

    block_attn_res = IdentityOp
    if config is not None and config.block_attention_residuals:
        block_attn_res = LayerSpec(
            layer=BlockAttnRes,
            sublayers_spec=BlockAttnResSublayersSpec(
                norm=layer_norm,
            ),
        )
    transformer_cls = getattr(config, "specific_layer", TransformerLayer)

    # mHC: use HyperConnectionTransformerLayer when enabled
    if config is not None and config.enable_hyper_connections:
        from paddleformers.fleet.transformer.transformer_layer import (
            HyperConnectionTransformerLayer,
        )

        transformer_cls = HyperConnectionTransformerLayer

    if paddle.distributed.is_initialized():
        try:
            pp_configs = fleet.fleet._user_defined_strategy.hybrid_configs["pp_configs"]
            use_overlap = pp_configs.forward_backward_overlap_scheduler
        except KeyError:
            # pp_configs key does not exist, no overlap configured
            use_overlap = False
        except AttributeError:
            # pp_configs attribute does not exist, no overlap configured
            use_overlap = False
        if use_overlap:
            assert not config.enable_hyper_connections, "HyperConnectionTransformerLayer not supported for overlap."
            assert (
                transformer_cls.__name__ == TransformerLayer.__name__
            ), "Only base TransformerLayer can be overlapped."
            transformer_cls = TransformerLayerWithOverlap
    exp_variant = getattr(config, "experimental_attention_variant", None)
    if exp_variant == "dsv4_hybrid":
        # Route to DSv4 Hybrid if configured
        self_attn_spec = get_attention_spec(
            config=config,
            attention_layer_type="dsv4_hybrid_attention",
            attn_mask_type=AttnMaskType.causal,
            is_mtp_layer=is_mtp_layer,
        )
    elif multi_latent_attention:
        self_attn_spec = get_attention_spec(
            config=config,
            attention_layer_type="multi_latent_attention",
            attn_mask_type=AttnMaskType.causal,
            is_mtp_layer=is_mtp_layer,
        )
    else:
        self_attn_spec = get_attention_spec(
            config=config,
            attention_layer_type=attention_layer_type,
            attn_mask_type=attn_mask_type,
            is_mtp_layer=is_mtp_layer,
        )

    # mHC: build HC LayerSpec for sublayers_spec
    self_attention_hc_spec = IdentityOp
    mlp_hc_spec = IdentityOp
    if config is not None and config.enable_hyper_connections:
        from paddleformers.fleet.transformer.hyper_connection import (
            HyperConnectionModule,
        )

        self_attention_hc_spec = LayerSpec(layer=HyperConnectionModule)
        mlp_hc_spec = LayerSpec(layer=HyperConnectionModule)

    return LayerSpec(
        layer=transformer_cls,
        sublayers_spec=TransformerLayerSublayersSpec(
            input_layernorm=layer_norm,
            self_attention_hyper_connection=self_attention_hc_spec,
            self_attn=self_attn_spec,
            self_attn_bda=get_bias_dropout_add,
            post_attention_layernorm=layer_norm,
            mlp_hyper_connection=mlp_hc_spec,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            block_attn_res=block_attn_res,
            sharded_state_dict_keys_map={
                "input_layernorm.": "self_attn.qkv_proj.layer_norm_",
                "post_attention_layernorm.": "mlp.up_gate_proj.layer_norm_",
            },
        ),
        extra_kwargs={
            "config": config,
            "layer_number": layer_number,
            "is_mtp_layer": is_mtp_layer,
            "hidden_dropout_prob": config.hidden_dropout_prob if config is not None else None,
        },
    )


def get_mlp_layer_spec_for_backend(
    backend: BackendSpecProvider,
    num_experts: int | None = None,
    moe_expert_fusion: bool | None = False,
) -> LayerSpec:
    """Helper function to get layer spec for MLP/MoE"""

    down_proj = backend.row_parallel_linear()
    hidden_act = None

    if num_experts is None:
        # Dense MLP w/ or w/o TE layers.
        layer = MLP
        if backend.fuse_layernorm_and_linear():
            up_gate_proj = backend.column_parallel_layer_norm_linear()
            assert up_gate_proj is not None
        else:
            up_gate_proj = backend.column_parallel_linear()
        return LayerSpec(
            layer=layer,
            sublayers_spec=MLPSublayersSpec(
                up_gate_proj=up_gate_proj,
                down_proj=down_proj,
                hidden_act=hidden_act,
            ),
        )
    else:
        return get_moe_layer_spec_for_backend(
            backend=backend,
            num_experts=num_experts,
            moe_expert_fusion=moe_expert_fusion,
        )


def get_gpt_decoder_layers_spec(
    config: TransformerConfig,
    normalization: str | None = None,
    qk_l2_norm: bool | None = False,
) -> list[LayerSpec]:
    """GPT block spec."""
    dense_layer_spec_func = partial(
        get_gpt_layer_local_spec,
        config=config,
        num_experts=None,
        moe_expert_fusion=False,
        use_qk_norm=config.use_qk_norm,
        multi_latent_attention=config.multi_latent_attention,
        normalization=normalization,
        qk_l2_norm=qk_l2_norm,
    )

    moe_layer_spec_func = partial(
        get_gpt_layer_local_spec,
        config=config,
        num_experts=config.n_routed_experts,
        moe_expert_fusion=config.moe_expert_fusion,
        use_qk_norm=config.use_qk_norm,
        multi_latent_attention=config.multi_latent_attention,
        normalization=normalization,
        qk_l2_norm=qk_l2_norm,
    )

    # Parse config.moe_layer_freq to determine the pattern of expert/dense layers.
    # 0 stands for dense layers, 1 stands for expert layers.
    # For integer N: Creates a pattern with one expert layer every N layers.
    # For string pattern: Evaluates the str directly (e.g. "[1,0,1]" for alternating expert/dense).
    if isinstance(config.moe_layer_freq, int):
        moe_layer_pattern = [1 if (i % config.moe_layer_freq == 0) else 0 for i in range(config.num_hidden_layers)]
    elif isinstance(config.moe_layer_freq, list):
        moe_layer_pattern = config.moe_layer_freq
        assert len(moe_layer_pattern) == config.num_hidden_layers, (
            f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
            f"expected {config.num_hidden_layers}, "
            f"current moe layer pattern: {config.moe_layer_freq}"
        )
    else:
        raise ValueError(f"Invalid moe_layer_freq: {type(config.moe_layer_freq)}, {config.moe_layer_freq}")

    # Create the layer specs for the model.
    layer_specs = []
    for layer_number in range(config.num_hidden_layers):
        real_layer_number = layer_number + config.num_empty_layers_add_in_head
        if moe_layer_pattern[layer_number] == 1:
            layer_specs.append(moe_layer_spec_func(layer_number=real_layer_number))
        elif moe_layer_pattern[layer_number] == 0:
            layer_specs.append(dense_layer_spec_func(layer_number=real_layer_number))
        else:
            raise ValueError(f"Invalid layer pattern: {moe_layer_pattern}")

    return layer_specs


def get_gpt_mtp_layers_spec(
    config: TransformerConfig,
    spec: list[LayerSpec],
) -> list[LayerSpec]:
    """GPT Multi-Token Prediction (MTP) block spec."""
    backend = LocalSpecProvider()
    return get_gpt_mtp_layers_spec_for_backend(
        config=config,
        spec=spec,
        backend=backend,
    )


def get_gpt_mtp_layers_spec_for_backend(
    config: TransformerConfig,
    spec: list[LayerSpec],
    backend: BackendSpecProvider,
) -> list[LayerSpec]:
    assert isinstance(spec, list) and isinstance(spec[-1], LayerSpec)

    if config.mtp_num_layers > 0:
        mtp_num_layers = config.mtp_num_layers
    else:
        mtp_num_layers = config.num_nextn_predict_layers or 0

    mtp_layer_specs = []
    for i in range(mtp_num_layers):
        if config.use_dense_mtp and config.n_routed_experts is not None:
            num_experts = None
            moe_expert_fusion = False
        else:
            num_experts = config.n_routed_experts
            moe_expert_fusion = config.moe_expert_fusion

        transformer_layer_spec = get_gpt_layer_local_spec(
            config=config,
            num_experts=num_experts,
            moe_expert_fusion=moe_expert_fusion,
            use_qk_norm=config.use_qk_norm,
            multi_latent_attention=config.multi_latent_attention,
            normalization=config.normalization,
            layer_number=i,
            is_mtp_layer=True,
        )

        mtp_layer_specs.append(
            get_mtp_layer_spec_for_backend(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                backend=backend,
                layer_number=i,
            )
        )
    return mtp_layer_specs


def get_gpt_spec(
    config: TransformerConfig,
    transformer_layers_spec: list[LayerSpec],
    mtp_layers_spec: list[LayerSpec],
    vocab_size: int,
    max_sequence_length: int,
    head_empty_layers_spec: list[LayerSpec] | None = None,
    tail_empty_layers_spec: list[LayerSpec] | None = None,
    position_embedding_type: Literal["learned_absolute", "rope", "mrope", "none"] = "learned_absolute",
    rotary_percent: float = 1.0,
    rotary_base: int = 10000,
    swa_rotary_base: int = 10000,
    rope_scaling: bool = False,
    parallel_output: bool = False,
    tie_word_embeddings: bool = False,
):
    embedding_extra_kwargs = {
        "config": config,
        "vocab_size": vocab_size,
        "max_sequence_length": max_sequence_length,
        "position_embedding_type": position_embedding_type,
    }

    skip_weight_param_allocation = config.tie_word_embeddings and config.pipeline_model_parallel_size == 1

    language_embedding_spec = LayerSpec(layer=LanguageModelEmbedding)
    rope_embedding_spec = None
    if position_embedding_type == "rope" and not config.multi_latent_attention:
        rope_embedding_spec = LayerSpec(layer=RotaryEmbedding)
        rope_embedding_extra_kwargs = {
            "rotary_percent": rotary_percent,
            "rotary_base": rotary_base,
            "swa_rotary_base": swa_rotary_base,
            "rope_scaling": rope_scaling,
        }
        embedding_extra_kwargs = {
            **embedding_extra_kwargs,
            **rope_embedding_extra_kwargs,
        }
    elif position_embedding_type == "yarn":
        rope_embedding_spec = LayerSpec(layer=YarnRotaryEmbedding)
        rope_embedding_extra_kwargs = {
            "rotary_percent": rotary_percent,
            "rotary_base": rotary_base,
            "swa_rotary_base": swa_rotary_base,
            "rope_scaling": rope_scaling,
        }
        embedding_extra_kwargs = {
            **embedding_extra_kwargs,
            **rope_embedding_extra_kwargs,
        }
    elif position_embedding_type == "mrope" and not config.multi_latent_attention:
        rope_embedding_spec = LayerSpec(layer=MultimodalRotaryEmbedding)
        rope_embedding_extra_kwargs = {
            "rotary_percent": rotary_percent,
            "rotary_base": rotary_base,
            "swa_rotary_base": swa_rotary_base,
            "rope_scaling": rope_scaling,
            "mrope_section": config.mrope_section,
        }
        embedding_extra_kwargs = {
            **embedding_extra_kwargs,
            **rope_embedding_extra_kwargs,
        }
        assert (
            config.mrope_section is not None
        ), "mrope require mrope_section setting, but we got None from TransformerConfig"

    embedding_spec = GPTEmbeddingSpec(
        language_embedding=language_embedding_spec,
        rope_embedding=rope_embedding_spec,
    )

    # Build block_attn_res spec for GPTLMHead
    lm_head_block_attn_res = IdentityOp
    if config.block_attention_residuals:
        backend = LocalSpecProvider()
        lm_head_norm = backend.layer_norm(
            rms_norm=(config.normalization == "RMSNorm"),
            for_qk=False,
        )
        lm_head_block_attn_res = LayerSpec(
            layer=BlockAttnRes,
            sublayers_spec=BlockAttnResSublayersSpec(
                norm=lm_head_norm,
            ),
        )

    # separate mtp head & loss
    mtp_lm_head_spec = None
    mtp_loss_spec = None
    if config.separate_mtp_headloss:
        assert (
            config.num_nextn_predict_layers is not None and config.num_nextn_predict_layers > 0
        ), "If you set separate_mtp_headloss to True, mtp layer num must be greater than 0."

        mtp_lm_head_spec = LayerSpec(
            layer=GPTMTPLMHead,
            extra_kwargs={
                "input_size": config.hidden_size,
                "output_size": vocab_size,
                "config": config,
                "init_method": config.init_method,
                "bias": False,
                "skip_bias_add": False,
                "gather_output": not parallel_output,
                "skip_weight_param_allocation": skip_weight_param_allocation,
                "block_attn_res": lm_head_block_attn_res,
            },
        )
        mtp_loss_spec = LayerSpec(
            layer=MTPLanguageLoss,
            extra_kwargs={
                "config": config,
            },
        )
        lm_head_spec = LayerSpec(
            layer=GPTMainLMHead,
            extra_kwargs={
                "input_size": config.hidden_size,
                "output_size": vocab_size,
                "config": config,
                "init_method": config.init_method,
                "bias": False,
                "skip_bias_add": False,
                "gather_output": not parallel_output,
                "skip_weight_param_allocation": skip_weight_param_allocation,
                "block_attn_res": lm_head_block_attn_res,
            },
        )
    else:
        lm_head_spec = LayerSpec(
            layer=GPTLMHead,
            extra_kwargs={
                "input_size": config.hidden_size,
                "output_size": vocab_size,
                "config": config,
                "init_method": config.init_method,
                "bias": False,
                "skip_bias_add": False,
                "gather_output": not parallel_output,
                "skip_weight_param_allocation": skip_weight_param_allocation,
                "block_attn_res": lm_head_block_attn_res,
            },
        )

    norm_input_parallel = config.sequence_parallel and config.tensor_model_parallel_size > 1

    if config.enable_hyper_connections:
        from paddleformers.fleet.transformer.hyper_connection import (
            HyperConnectionContractLayer,
            HyperConnectionExpandLayer,
        )

    # MTP magic send: re-embed input_ids at the last stage
    mtp_embedding_spec = None
    if config.enable_mtp_magic_send and config.num_nextn_predict_layers > 0:
        mtp_embedding_spec = LayerSpec(
            layer=MTPEmbeddingLayer,
            extra_kwargs={"config": config},
        )

    return LayerSpec(
        layer=GPTModel,
        extra_kwargs={
            "config": config,
            "tie_word_embeddings": tie_word_embeddings,
        },
        sublayers_spec=GPTSublayersSpec(
            embedding=LayerSpec(
                layer=GPTEmbedding,
                sublayers_spec=embedding_spec,
                extra_kwargs=embedding_extra_kwargs,
            ),
            head_empty_layers=head_empty_layers_spec,
            mhc_expand=LayerSpec(
                layer=HyperConnectionExpandLayer,
                extra_kwargs={"config": config},
            )
            if config.enable_hyper_connections
            else None,
            transformer_layers=transformer_layers_spec,
            mhc_contract=LayerSpec(
                layer=HyperConnectionContractLayer,
                extra_kwargs={"config": config},
            )
            if config.enable_hyper_connections
            else None,
            tail_empty_layers=tail_empty_layers_spec,
            mtp=mtp_layers_spec,
            mtp_embedding=mtp_embedding_spec,
            mtp_lm_head=mtp_lm_head_spec,
            mtp_loss=mtp_loss_spec,
            layer_norm=LayerSpec(
                layer=WrappedPaddleNormPipe,
                extra_kwargs={
                    "config": config,
                    "hidden_size": config.hidden_size,
                    "eps": config.rms_norm_eps,
                    "input_is_parallel": norm_input_parallel,
                },
            ),
            lm_head=lm_head_spec,
        ),
    )
