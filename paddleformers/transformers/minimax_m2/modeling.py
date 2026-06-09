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

import logging
import math

import paddle
from paddlefleet.transformer.utils import is_layer_window_attention

from ...nn.pp_model import CriterionLayerPipe, GeneralModelForCausalLMPipe
from ..glm4_moe.modeling import GLMMoEModelProvider
from ..model_utils import PretrainedModel
from .configuration import MiniMaxM2Config

logger = logging.getLogger(__name__)


class MiniMaxM2PreTrainedModel(PretrainedModel):
    config: MiniMaxM2Config

    @staticmethod
    def get_layer_attn_split_info(config, layer_idx):
        layer_is_swa = is_layer_window_attention(
            config.sliding_window,
            config.window_attn_skip_freq,
            layer_idx - config.num_empty_layers_add_in_head,
        )
        if layer_is_swa:
            head_dim = config.swa_head_dim
            v_head_dim = config.swa_v_head_dim
            num_attention_heads = config.swa_num_attention_heads
            num_key_value_heads = config.swa_num_key_value_heads

        else:
            head_dim = config.head_dim
            v_head_dim = config.v_head_dim
            num_attention_heads = config.num_attention_heads
            num_key_value_heads = config.num_key_value_heads

        # NOTE(GouxiaWang): only support tp=1
        gated_attn = config.use_gated_attn or getattr(config, "gated_attention", False)
        num_query_groups = num_key_value_heads
        q_heads_per_group = num_attention_heads // num_key_value_heads
        heads_per_group = q_heads_per_group
        if config.use_vha_attention:
            q_heads_per_group = q_heads_per_group // num_key_value_heads
        q_dim = q_heads_per_group * head_dim

        if gated_attn:
            # Per group: Q + Gate + K + V
            gate_dim = heads_per_group * v_head_dim
            split_dims = [q_dim, gate_dim, head_dim, v_head_dim]
        else:
            # Per group: Q + K + V
            split_dims = [q_dim, head_dim, v_head_dim]

        return num_attention_heads, num_key_value_heads, num_query_groups, split_dims, layer_is_swa

    @classmethod
    def _build_muon_slice_config(cls, model, config) -> dict:
        """Build declarative slice configuration for Muon optimizer.

        Constructs a mapping from parameter original names to (slice_fn, slice_kwargs)
        tuples. This allows slice strategies to be defined declaratively in the model
        configuration rather than being hard-coded inside the muon.py optimizer.

        Args:
            model: The GPTModel (PipelineLayer) instance.
            config: The model configuration object.

        Returns:
            A dict mapping parameter name strings to (slice_fn, slice_kwargs) tuples.
        """

        def _qkv_per_head(matrix_2d_global, ortho_fn, num_query_groups=None, split_dims=None):
            """Slice QKV by heads, orthogonalise each head independently."""
            groups = paddle.split(matrix_2d_global, num_query_groups, axis=1)

            processed_groups = []
            if len(split_dims) == 4:
                for group in groups:
                    q_part, gate_part, k_head, v_head = paddle.split(
                        group,
                        split_dims,
                        axis=1,
                    )
                    q_heads = paddle.split(q_part, num_query_groups, axis=1)
                    q_ortho = paddle.concat([ortho_fn(h) for h in q_heads], axis=1)
                    gate_heads = paddle.split(gate_part, num_query_groups, axis=1)
                    gate_ortho = paddle.concat([ortho_fn(g) for g in gate_heads], axis=1)
                    processed_groups.append(
                        paddle.concat([q_ortho, gate_ortho, ortho_fn(k_head), ortho_fn(v_head)], axis=1)
                    )
            else:
                for group in groups:
                    q_part, k_head, v_head = paddle.split(
                        group,
                        split_dims,
                        axis=1,
                    )
                    q_heads = paddle.split(q_part, num_query_groups, axis=1)
                    q_ortho = paddle.concat([ortho_fn(h) for h in q_heads], axis=1)
                    processed_groups.append(paddle.concat([q_ortho, ortho_fn(k_head), ortho_fn(v_head)], axis=1))

            return paddle.concat(processed_groups, axis=1)

        def _qkv_sep(matrix_2d, ortho_fn, num_query_groups=None, split_dims=None):
            """Slice QKV into Q, K, V blocks, orthogonalise each as whole."""

            groups = paddle.split(matrix_2d, num_query_groups, axis=1)
            q_parts, g_parts, k_parts, v_parts = [], [], [], []
            for group in groups:
                if len(split_dims) == 4:
                    q_p, g_p, k_p, v_p = paddle.split(group, split_dims, axis=1)
                    g_parts.append(g_p)
                else:
                    q_p, k_p, v_p = paddle.split(group, split_dims, axis=1)
                q_parts.append(q_p)
                k_parts.append(k_p)
                v_parts.append(v_p)

            q_ortho = ortho_fn(paddle.concat(q_parts, axis=1))
            if len(split_dims) == 4:
                g_ortho = ortho_fn(paddle.concat(g_parts, axis=1))
            k_ortho = ortho_fn(paddle.concat(k_parts, axis=1))
            v_ortho = ortho_fn(paddle.concat(v_parts, axis=1))

            q_groups = paddle.split(q_ortho, num_query_groups, axis=1)
            if len(split_dims) == 4:
                g_groups = paddle.split(g_ortho, num_query_groups, axis=1)
            k_groups = paddle.split(k_ortho, num_query_groups, axis=1)
            v_groups = paddle.split(v_ortho, num_query_groups, axis=1)

            if len(split_dims) == 4:
                return paddle.concat(
                    [
                        paddle.concat([q_groups[i], g_groups[i], k_groups[i], v_groups[i]], axis=1)
                        for i in range(num_query_groups)
                    ],
                    axis=1,
                )
            else:
                return paddle.concat(
                    [paddle.concat([q_groups[i], k_groups[i], v_groups[i]], axis=1) for i in range(num_query_groups)],
                    axis=1,
                )

        def _ffn_gate_up(matrix, ortho_fn, intermediate_size=None):
            """Slice FFN gate_up, orthogonalise gate and up independently."""

            if matrix.ndim == 2:
                gate, up = paddle.split(matrix, [intermediate_size, intermediate_size], axis=1)
                return paddle.concat([ortho_fn(gate), ortho_fn(up)], axis=1)
            elif matrix.ndim == 3:
                expert_updates = []
                for ei in range(matrix.shape[0]):
                    gate, up = paddle.split(matrix[ei], [intermediate_size, intermediate_size], axis=1)
                    expert_updates.append(paddle.concat([ortho_fn(gate), ortho_fn(up)], axis=1))
                return paddle.stack(expert_updates, axis=0)
            else:
                raise ValueError(f"FFN gate_up split expects 2D or 3D tensor, got shape {matrix.shape}")

        def _mla_per_head(matrix_2d_global, ortho_fn, head_num=None, axis=None, head_split_sizes=None):
            """Slice MLA weights by heads."""

            split_args = head_num if head_split_sizes is None else head_split_sizes * head_num
            groups = paddle.split(matrix_2d_global, split_args, axis=axis)
            processed_groups = [ortho_fn(group) for group in groups]
            return paddle.concat(processed_groups, axis=axis)

        def _moe_experts(matrix_3d_global, ortho_fn):
            """Slice MoE weights by experts."""

            if matrix_3d_global.ndim != 3:
                raise ValueError(f"MoE expert split expects 3D tensor, got shape {matrix_3d_global.shape}")
            n_experts = matrix_3d_global.shape[0]
            return paddle.stack(
                [ortho_fn(matrix_3d_global[ei]) for ei in range(n_experts)],
                axis=0,
            )

        slice_config = {}

        muon_configs = config.muon_configs

        num_hidden_layers = config.num_hidden_layers
        use_mla = bool(getattr(config, "multi_latent_attention", False))
        moe_expert_fusion = getattr(config, "moe_expert_fusion", False)
        use_gated_attn = getattr(config, "use_gated_attn", False)
        csa_compress_ratios = getattr(config, "csa_compress_ratios", None)

        # Get Muon configuration from muon_configs
        muon_qkv_update_mode = muon_configs.get("muon_qkv_update_mode", "split_head")
        muon_ffn_split = muon_configs.get("muon_ffn_split", False)

        # Determine QKV slice strategy based on mode
        qkv_slice_fn = None
        if muon_qkv_update_mode == "split_head":
            qkv_slice_fn = _qkv_per_head
        elif muon_qkv_update_mode == "split_qkv":
            qkv_slice_fn = _qkv_sep

        # Determine FFN slice strategy
        ffn_slice_fn = _ffn_gate_up if muon_ffn_split else None

        # Determine Fused MoE slice strategy
        fused_moe_fn = _moe_experts if moe_expert_fusion else None

        # vha_premix_weight shape: [num_key_value_heads, head_dim, head_dim]
        # we reuse 3D shape moe_expert_fusion func
        vha_premix_fn = _moe_experts if config.use_vha_attention else None

        # Determine MLA slice strategy
        mla_slice_fn = None
        if use_mla and muon_qkv_update_mode == "split_head":
            mla_slice_fn = _mla_per_head

        def _add_layer_slice_config(prefix, layer_idx):

            # DeepSeekV4 Attention weights:
            if csa_compress_ratios is not None and mla_slice_fn is not None:
                real_layer_idx = layer_idx - config.num_empty_layers_add_in_head
                ratio = csa_compress_ratios[real_layer_idx]
                num_attention_heads = config.num_attention_heads
                # common weights (Sliding Window Attenion)
                slice_config[f"{prefix}.self_attn.linear_q_up_proj.weight"] = (
                    mla_slice_fn,
                    {
                        "head_num": num_attention_heads,
                        "axis": 1,
                    },
                )

                # Compressor weights
                if ratio == 4:
                    slice_config[f"{prefix}.self_attn.core_attention.compressor.linear_wkv.weight"] = (
                        mla_slice_fn,
                        {
                            "head_num": 1,
                            "axis": 1,
                            "head_split_sizes": [config.v_head_dim, config.v_head_dim],
                        },
                    )
                    slice_config[f"{prefix}.self_attn.core_attention.compressor.linear_wgate.weight"] = (
                        mla_slice_fn,
                        {
                            "head_num": 1,
                            "axis": 1,
                            "head_split_sizes": [config.v_head_dim, config.v_head_dim],
                        },
                    )
                # Indexer weights
                print(f"layer: {real_layer_idx}, ratio: {ratio}, dense_mode: {config.csa_dense_mode}")
                if ratio == 4 and config.csa_dense_mode is False:
                    slice_config[f"{prefix}.self_attn.core_attention.indexer.linear_wq_b.weight"] = (
                        mla_slice_fn,
                        {
                            "head_num": config.dsa_index_n_heads,
                            "axis": 1,
                        },
                    )
                    # Compressed weights
                    slice_config[f"{prefix}.self_attn.core_attention.indexer.compressor.linear_wkv.weight"] = (
                        mla_slice_fn,
                        {
                            "head_num": 1,
                            "axis": 1,
                            "head_split_sizes": [config.dsa_index_head_dim, config.dsa_index_head_dim],
                        },
                    )
                    slice_config[f"{prefix}.self_attn.core_attention.indexer.compressor.linear_wgate.weight"] = (
                        mla_slice_fn,
                        {
                            "head_num": 1,
                            "axis": 1,
                            "head_split_sizes": [config.dsa_index_head_dim, config.dsa_index_head_dim],
                        },
                    )

            num_heads, num_kv_heads, num_query_groups, split_dims, layer_is_swa = cls.get_layer_attn_split_info(
                config, layer_idx
            )
            # Fused QKV weights (non-MLA path)
            if not use_mla and qkv_slice_fn is not None:
                if config.use_vha_attention:
                    # VHA: independent projections, split by heads then ortho each head
                    if muon_qkv_update_mode == "split_head":
                        g = num_heads // num_kv_heads  # heads per group in q_proj
                        q_lora_rank = config.swa_vha_q_lora_rank if layer_is_swa else config.vha_q_lora_rank
                        head_dim = config.swa_head_dim if layer_is_swa else config.head_dim
                        v_head_dim = config.swa_v_head_dim if layer_is_swa else config.v_head_dim
                        slice_config[f"{prefix}.self_attn.q_proj.weight"] = (
                            _mla_per_head,
                            {"head_num": g, "axis": 1, "head_split_sizes": [q_lora_rank]},
                        )
                        slice_config[f"{prefix}.self_attn.k_proj.weight"] = (
                            _mla_per_head,
                            {"head_num": num_kv_heads, "axis": 1, "head_split_sizes": [head_dim]},
                        )
                        slice_config[f"{prefix}.self_attn.v_proj.weight"] = (
                            _mla_per_head,
                            {"head_num": num_kv_heads, "axis": 1, "head_split_sizes": [v_head_dim]},
                        )
                        if use_gated_attn:
                            slice_config[f"{prefix}.self_attn.gate_proj.weight"] = (
                                _mla_per_head,
                                {"head_num": num_heads, "axis": 1, "head_split_sizes": [v_head_dim]},
                            )
                        if vha_premix_fn is not None:
                            slice_config[f"{prefix}.self_attn.vha_premix_weight"] = (vha_premix_fn, {})
                    # split_qkv: no fuse, skip
                else:
                    qkv_kwargs = {"num_query_groups": num_query_groups, "split_dims": split_dims}
                    slice_config[f"{prefix}.self_attn.qkv_proj.weight"] = (qkv_slice_fn, qkv_kwargs)

            # FFN gate_up weights
            if ffn_slice_fn is not None:
                moe_intermediate_size = config.moe_intermediate_size
                intermediate_size = config.intermediate_size

                # Fused experts
                param_name = f"{prefix}.mlp.experts.up_gate_proj.weight"
                slice_config[param_name] = (ffn_slice_fn, {"intermediate_size": moe_intermediate_size})

                # Shared experts
                slice_config[f"{prefix}.mlp.shared_experts.up_gate_proj.weight"] = (
                    ffn_slice_fn,
                    {"intermediate_size": moe_intermediate_size},
                )
                slice_config[f"{prefix}.mlp.grouped_gemm_experts.weight1"] = (
                    ffn_slice_fn,
                    {"intermediate_size": moe_intermediate_size},
                )
                # Common experts
                param_name = f"{prefix}.mlp.up_gate_proj.weight"
                slice_config[param_name] = (ffn_slice_fn, {"intermediate_size": intermediate_size})

                # Routed experts (per-expert)
                if hasattr(config, "n_routed_experts") and config.n_routed_experts > 0:
                    for expert_idx in range(config.n_routed_experts):
                        slice_config[f"{prefix}.mlp.experts.{expert_idx}.up_gate_proj.weight"] = (
                            ffn_slice_fn,
                            {"intermediate_size": moe_intermediate_size},
                        )

            # Fused MoE weights (grouped_gemm)
            if moe_expert_fusion and fused_moe_fn is not None:
                slice_config[f"{prefix}.mlp.experts.down_proj.weight"] = (fused_moe_fn, {})
                slice_config[f"{prefix}.mlp.grouped_gemm_experts.weight2"] = (fused_moe_fn, {})

            # MLA weights
            if use_mla and mla_slice_fn is not None:
                # NOTE(XiangruiYU): reset num_attention_heads and v_head_dim for SWA.
                num_attention_heads = num_heads
                v_head_dim = split_dims[-1]
                assert (
                    hasattr(config, "qk_nope_head_dim")
                    and hasattr(config, "qk_rope_head_dim")
                    and hasattr(config, "kv_lora_rank")
                    and hasattr(config, "v_head_dim")
                )

                slice_config[f"{prefix}.self_attn.q_b_proj.weight"] = (
                    mla_slice_fn,
                    {
                        "head_num": num_attention_heads,
                        "axis": 1,
                        "head_split_sizes": [config.qk_nope_head_dim, config.qk_rope_head_dim],
                    },
                )

                slice_config[f"{prefix}.self_attn.kv_a_proj_with_mqa.weight"] = (
                    mla_slice_fn,
                    {"head_num": 1, "axis": 1, "head_split_sizes": [config.kv_lora_rank, config.qk_rope_head_dim]},
                )

                slice_config[f"{prefix}.self_attn.kv_b_proj.weight"] = (
                    mla_slice_fn,
                    {
                        "head_num": num_attention_heads,
                        "axis": 1,
                        "head_split_sizes": [config.qk_nope_head_dim, v_head_dim],
                    },
                )

                if use_gated_attn:
                    slice_config[f"{prefix}.self_attn.gate_proj.weight"] = (
                        mla_slice_fn,
                        {"head_num": num_attention_heads, "axis": 1},
                    )

        # Main layers
        for layer_idx in range(num_hidden_layers):
            real_layer_number = layer_idx + config.num_empty_layers_add_in_head
            _add_layer_slice_config(f"model.layers.{real_layer_number}", real_layer_number)

        # MTP layers
        if config.mtp_num_layers > 0:
            num_nextn_predict_layers = config.mtp_num_layers
        else:
            num_nextn_predict_layers = config.num_nextn_predict_layers if config.num_nextn_predict_layers else 0

        for layer_idx in range(num_nextn_predict_layers):
            real_layer_number = layer_idx + config.num_empty_layers_add_in_head + num_hidden_layers
            _add_layer_slice_config(f"model.layers.{real_layer_number}", real_layer_number)
        for layer_idx in range(num_nextn_predict_layers):
            real_layer_number = layer_idx + config.num_empty_layers_add_in_head + num_hidden_layers
            _add_layer_slice_config(f"model.layers.{real_layer_number}.transformer_layer", real_layer_number)

        return slice_config

    @classmethod
    def build_muon_param_info_map(cls, model, config):
        """Build parameter info map for Muon optimizer.

        Args:
            model: The GPTModel (PipelineLayer) instance.
            config: The model configuration object.

        Returns:
            Dict[str, MuonParamInfo]: Mapping from parameter name to Muon metadata.
        """
        from functools import partial

        from paddle.optimizer.muon import MuonParamInfo, _default_should_use_muon

        info_map = {}
        exclude_patterns = config.muon_configs["muon_exclude_patterns"]

        # Get slice config from model (keys are original names like "model.layers.0.xxx")
        slice_config = cls._build_muon_slice_config(model, config)

        # Build pipeline name -> original name mapping by inverting the forward mapping
        # returned by _set_pipeline_name_mapping(). We use the return value instead of
        # model._pp_to_single_mapping because Paddle Layer.__setattr__ prevents the
        # instance attribute from persisting after super().__init__().
        pp_to_single = getattr(model, "_pp_to_single_mapping", None)
        if pp_to_single is None:
            try:
                single_to_pp = model._set_pipeline_name_mapping()
                if single_to_pp:
                    pp_to_single = {v: k for k, v in single_to_pp.items()}
            except Exception as e:
                logger.warning(f"_set_pipeline_name_mapping failed: {e}")
        if pp_to_single is None:
            pp_to_single = {}

        for pp_name, param in model.named_parameters():
            name = pp_to_single.get(pp_name, pp_name)
            use_muon = (
                _default_should_use_muon(name, param.shape, exclude_patterns)
                and _default_should_use_muon(param.name, param.shape, exclude_patterns)
                and "hc_head_fn" not in name
                and "mapping_proj" not in name
            )

            if name in slice_config:
                slice_fn, slice_kwargs = slice_config[name]
                param_info = MuonParamInfo(
                    use_muon=use_muon,
                    split_concat_func=partial(slice_fn, **slice_kwargs),
                )
            else:
                param_info = MuonParamInfo(
                    use_muon=use_muon,
                    split_concat_func=None,
                )

            info_map[param.name] = param_info

            sc_func = param_info.split_concat_func
            func_name = sc_func.func.__name__ if sc_func else None
            func_kwargs = sc_func.keywords if sc_func else {}

            logger.info(
                f"name: {name}, param.name: {param.name}, shape: {param.shape}, "
                f"use_muon: {use_muon}, "
                f"split_concat_func: {func_name}, "
                f"split_concat_func_kwargs: {func_kwargs}"
            )

        return info_map

    @classmethod
    def _gen_aoa_config(cls, config: MiniMaxM2Config):
        model_prefix = "model."  # "" if cls == cls.base_model_class else
        using_sonic_moe = config.using_sonic_moe
        if hasattr(config, "n_routed_experts"):
            num_experts = config.n_routed_experts
        else:
            num_experts = config.num_experts
        aoa_config = {
            "aoa_statements": [
                f"model.norm.weight -> {model_prefix}norm.weight",
            ]
        }

        aoa_config["aoa_statements"] += [
            f"model.embed_tokens.weight -> {model_prefix}embedding.embed_tokens.weight",
        ]

        assert not (
            config.tie_word_embeddings and getattr(config, "separate_mtp_headloss", False)
        ), "tie_word_embeddings and separate_mtp_headloss cannot be enabled simultaneously in aoa"
        if config.tie_word_embeddings:
            aoa_config["aoa_statements"] += [f"model.embed_tokens.weight -> {model_prefix}lm_head.weight"]
        elif getattr(config, "separate_mtp_headloss", False):
            aoa_config["aoa_statements"] += [f"lm_head.weight -> {model_prefix}shared_mtp_lm_head.weight"]
            aoa_config["aoa_statements"] += [f"lm_head.weight -> {model_prefix}shared_head.weight"]
        else:
            aoa_config["aoa_statements"] += [f"lm_head.weight -> {model_prefix}lm_head.weight"]

        num_hidden_layers = config.num_hidden_layers
        num_head_empty_layers = (
            config.num_empty_layers_add_in_head
            if hasattr(config, "num_empty_layers_add_in_head") and config.num_empty_layers_add_in_head
            else 0
        )

        # NOTE: MiniMax-M2 has no dense layers (first_k_dense_replace=0)

        if config.mtp_num_layers > 0:
            num_nextn_predict_layers = config.mtp_num_layers
        else:
            num_nextn_predict_layers = config.num_nextn_predict_layers if config.num_nextn_predict_layers else 0

        n_shared_experts = config.n_shared_experts if hasattr(config, "n_shared_experts") else 0
        if n_shared_experts > 0:
            assert n_shared_experts == 1, f"n_shared_experts must be 0 or 1 in MiniMax-M2, but got {n_shared_experts}"

        # mtp layers
        for layer_idx in reversed(range(num_hidden_layers, num_hidden_layers + num_nextn_predict_layers)):
            layer_idx_offset = layer_idx + num_head_empty_layers
            prefix = f"model.layers.{layer_idx}"
            prefix_offset = f"{model_prefix}layers.{layer_idx_offset}"
            aoa_config["aoa_statements"] += [
                f"{prefix}.eh_proj.weight^T -> {prefix_offset}.eh_proj.weight",
                f"{prefix}.enorm.weight -> {prefix_offset}.enorm.weight",
                f"{prefix}.hnorm.weight -> {prefix_offset}.hnorm.weight",
                f"{prefix}.shared_head.norm.weight -> {prefix_offset}.norm.weight",
            ]

            # transformer_layer.mlp.up_gate_proj.weight
            if config.use_dense_mtp:
                prefix_offset += ".transformer_layer"
                aoa_config["aoa_statements"] += [
                    f"{prefix}.mlp.gate_proj.weight^T, {prefix}.mlp.up_proj.weight^T -> {prefix_offset}.mlp.up_gate_proj.weight, fused_ffn",
                    f"{prefix}.mlp.down_proj.weight^T -> {prefix_offset}.mlp.down_proj.weight",
                ]

        # layer0 - layer_num_hidden_layers
        for layer_idx in reversed(range(0, num_hidden_layers + num_nextn_predict_layers)):
            layer_idx_offset = layer_idx + num_head_empty_layers
            prefix = f"model.layers.{layer_idx}"
            prefix_offset = f"{model_prefix}layers.{layer_idx_offset}"
            if layer_idx >= num_hidden_layers:
                # for mtp
                prefix_offset += ".transformer_layer"
            aoa_config["aoa_statements"] += [
                f"{prefix}.input_layernorm.weight -> {prefix_offset}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight -> {prefix_offset}.post_attention_layernorm.weight",
                f"{prefix}.self_attn.o_proj.weight^T -> {prefix_offset}.self_attn.o_proj.weight",
            ]

            use_mla = bool(getattr(config, "multi_latent_attention", False))

            if config.use_gated_attn and use_mla:
                # MLA mode: gate_proj is a separate parameter
                aoa_config["aoa_statements"] += [
                    f"{prefix}.self_attn.gate_proj.weight^T -> {prefix_offset}.self_attn.gate_proj.weight",
                ]

            is_swa = is_layer_window_attention(config.sliding_window, config.window_attn_skip_freq, layer_idx)
            if (
                config.softmax_type == "learnable"
                or (config.add_full_attention_sink_bias and not is_swa)
                or (config.add_swa_attention_sink_bias and is_swa)
            ):
                aoa_config["aoa_statements"] += [
                    f"{prefix}.self_attn.core_attention.softmax_offset -> {prefix_offset}.self_attn.core_attention.softmax_offset",
                ]

            if config.use_vha_attention:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.self_attn.vha_premix_weight -> {prefix_offset}.self_attn.vha_premix_weight",
                ]
                aoa_config["aoa_statements"] += [
                    f"{prefix}.self_attn.vha_postmix_U -> {prefix_offset}.self_attn.vha_postmix_U",
                ]
                aoa_config["aoa_statements"] += [
                    f"{prefix}.self_attn.vha_postmix_V -> {prefix_offset}.self_attn.vha_postmix_V",
                ]

            if use_mla:
                # MLA attention
                aoa_config["aoa_statements"] += [
                    f"{prefix}.self_attn.q_a_proj.weight^T -> {prefix_offset}.self_attn.q_a_proj.weight",
                    f"{prefix}.self_attn.q_b_proj.weight^T -> {prefix_offset}.self_attn.q_b_proj.weight",
                    f"{prefix}.self_attn.kv_a_proj_with_mqa.weight^T -> {prefix_offset}.self_attn.kv_a_proj_with_mqa.weight",
                    f"{prefix}.self_attn.kv_b_proj.weight^T -> {prefix_offset}.self_attn.kv_b_proj.weight",
                ]
                if config.use_qk_norm:
                    aoa_config["aoa_statements"] += [
                        f"{prefix}.self_attn.q_a_layernorm.weight -> {prefix_offset}.self_attn.q_a_layernorm.weight",
                        f"{prefix}.self_attn.kv_a_layernorm.weight -> {prefix_offset}.self_attn.kv_a_layernorm.weight",
                    ]

            elif config.experimental_attention_variant == "dsv4_hybrid":
                # csa_compress_ratios has length num_hidden_layers + num_nextn_predict_layers,
                # i.e. it covers both main layers and MTP layers.
                assert len(config.csa_compress_ratios) == num_hidden_layers + num_nextn_predict_layers, (
                    f"csa_compress_ratios length ({len(config.csa_compress_ratios)}) must equal "
                    f"num_hidden_layers + num_nextn_predict_layers "
                    f"({num_hidden_layers} + {num_nextn_predict_layers})"
                )
                csa_ratio = config.csa_compress_ratios[layer_idx]
                aoa_config["aoa_statements"] += [
                    # Linear projections (transpose: HF [out, in] -> paddle [in, out])
                    f"{prefix}.self_attn.linear_q_down_proj.weight^T -> {prefix_offset}.self_attn.linear_q_down_proj.weight",
                    f"{prefix}.self_attn.linear_q_up_proj.weight^T -> {prefix_offset}.self_attn.linear_q_up_proj.weight",
                    f"{prefix}.self_attn.linear_kv_proj.weight^T -> {prefix_offset}.self_attn.linear_kv_proj.weight",
                    f"{prefix}.self_attn.o_proj.weight^T -> {prefix_offset}.self_attn.o_proj.weight",
                    # Layer norms (no transpose, 1D)
                    f"{prefix}.self_attn.q_layernorm.weight -> {prefix_offset}.self_attn.q_layernorm.weight",
                    f"{prefix}.self_attn.kv_layernorm.weight -> {prefix_offset}.self_attn.kv_layernorm.weight",
                    # Grouped output projection (raw parameter, shape [out, in] on both sides)
                    f"{prefix}.self_attn.linear_o_group_proj -> {prefix_offset}.self_attn.linear_o_group_proj",
                    # Core attention: learnable attention sink (1D, no transpose)
                    f"{prefix}.self_attn.core_attention.attn_sink -> {prefix_offset}.self_attn.core_attention.attn_sink",
                ]
                # Compressor exists only when compress_ratio > 1 (i.e. ratio in {4, 128})
                if csa_ratio > 1:
                    aoa_config["aoa_statements"] += [
                        f"{prefix}.self_attn.core_attention.compressor.linear_wkv.weight^T -> {prefix_offset}.self_attn.core_attention.compressor.linear_wkv.weight",
                        f"{prefix}.self_attn.core_attention.compressor.linear_wgate.weight^T -> {prefix_offset}.self_attn.core_attention.compressor.linear_wgate.weight",
                        f"{prefix}.self_attn.core_attention.compressor.norm.weight -> {prefix_offset}.self_attn.core_attention.compressor.norm.weight",
                        f"{prefix}.self_attn.core_attention.compressor.ape -> {prefix_offset}.self_attn.core_attention.compressor.ape",
                    ]
                # Indexer exists only when compress_ratio == 4 and not csa_dense_mode
                if csa_ratio == 4 and not getattr(config, "csa_dense_mode", False):
                    aoa_config["aoa_statements"] += [
                        f"{prefix}.self_attn.core_attention.indexer.linear_wq_b.weight^T -> {prefix_offset}.self_attn.core_attention.indexer.linear_wq_b.weight",
                        f"{prefix}.self_attn.core_attention.indexer.linear_weights_proj.weight^T -> {prefix_offset}.self_attn.core_attention.indexer.linear_weights_proj.weight",
                        f"{prefix}.self_attn.core_attention.indexer.compressor.linear_wkv.weight^T -> {prefix_offset}.self_attn.core_attention.indexer.compressor.linear_wkv.weight",
                        f"{prefix}.self_attn.core_attention.indexer.compressor.linear_wgate.weight^T -> {prefix_offset}.self_attn.core_attention.indexer.compressor.linear_wgate.weight",
                        f"{prefix}.self_attn.core_attention.indexer.compressor.norm.weight -> {prefix_offset}.self_attn.core_attention.indexer.compressor.norm.weight",
                        f"{prefix}.self_attn.core_attention.indexer.compressor.ape -> {prefix_offset}.self_attn.core_attention.indexer.compressor.ape",
                    ]

            else:
                if config.use_qk_norm:
                    aoa_config["aoa_statements"] += [
                        f"{prefix}.self_attn.q_norm.weight -> {prefix_offset}.self_attn.q_norm.weight",
                        f"{prefix}.self_attn.k_norm.weight -> {prefix_offset}.self_attn.k_norm.weight",
                    ]

                # attention qkv
                # get_layer_attn_split_info will minus num_empty_layers_add_in_head
                num_heads, num_kv_heads, num_query_groups, split_dims, layer_is_swa = cls.get_layer_attn_split_info(
                    config, layer_idx + config.num_empty_layers_add_in_head
                )

                # Non-MLA gated attention: gate is fused in qkv_proj
                # Fleet layout per group: [Q_heads(hpg*hd), Gate_heads(hpg*hdv), K(hd), V(hdv)]
                # HF q_proj layout: [Q_h0(hd), G_h0(hd), Q_h1(hd), G_h1(hd), ...]
                if len(split_dims) == 4:
                    q_dim, gate_dim, head_dim, v_head_dim = split_dims
                    use_gated_attn = True
                else:
                    q_dim, head_dim, v_head_dim = split_dims
                    use_gated_attn = False

                gcd_head_dim = math.gcd(head_dim, v_head_dim)

                head_dim_chunks = head_dim // gcd_head_dim
                v_head_dim_chunks = v_head_dim // gcd_head_dim

                if config.use_vha_attention:
                    aoa_config["aoa_statements"] += [
                        f"{prefix}.self_attn.q_proj.weight^T -> {prefix_offset}.self_attn.q_proj.weight",
                        f"{prefix}.self_attn.k_proj.weight^T -> {prefix_offset}.self_attn.k_proj.weight",
                        f"{prefix}.self_attn.v_proj.weight^T -> {prefix_offset}.self_attn.v_proj.weight",
                    ]
                    if use_gated_attn:
                        aoa_config["aoa_statements"].append(
                            f"{prefix}.self_attn.gate_proj.weight^T -> {prefix_offset}.self_attn.gate_proj.weight"
                        )
                    if config.attention_bias:
                        aoa_config["aoa_statements"] += [
                            f"{prefix}.self_attn.q_proj.bias -> {prefix_offset}.self_attn.q_proj.bias",
                            f"{prefix}.self_attn.k_proj.bias -> {prefix_offset}.self_attn.k_proj.bias",
                            f"{prefix}.self_attn.v_proj.bias -> {prefix_offset}.self_attn.v_proj.bias",
                        ]
                        if use_gated_attn:
                            aoa_config["aoa_statements"].append(
                                f"{prefix}.self_attn.gate_proj.bias -> {prefix_offset}.self_attn.gate_proj.bias"
                            )

                else:
                    q_names = []
                    k_names = []
                    v_names = []
                    q_bias_names = []
                    k_bias_names = []
                    v_bias_names = []
                    heads_per_group = num_heads // num_kv_heads
                    for g in range(num_kv_heads):
                        for h in range(heads_per_group):
                            for c in range(head_dim_chunks):
                                q_names.append(f"{prefix}.self_attn._q_g{g}_h{h}_c{c}")
                                q_bias_names.append(f"{prefix}.self_attn._q_g{g}_h{h}_c{c}_bias")
                            if use_gated_attn:
                                for c in range(v_head_dim_chunks):
                                    q_names.append(f"{prefix}.self_attn._gate_g{g}_h{h}_c{c}")
                                    q_bias_names.append(f"{prefix}.self_attn._gate_g{g}_h{h}_c{c}_bias")

                        for c in range(head_dim_chunks):
                            k_names.append(f"{prefix}.self_attn._k_g{g}_c{c}")
                            k_bias_names.append(f"{prefix}.self_attn._k_g{g}_c{c}_bias")
                        for c in range(v_head_dim_chunks):
                            v_names.append(f"{prefix}.self_attn._v_g{g}_c{c}")
                            v_bias_names.append(f"{prefix}.self_attn._v_g{g}_c{c}_bias")
                    aoa_config["aoa_statements"].append(
                        f"{prefix}.self_attn.q_proj.weight -> {','.join(q_names)}, axis=0"
                    )
                    aoa_config["aoa_statements"].append(
                        f"{prefix}.self_attn.k_proj.weight -> {','.join(k_names)}, axis=0"
                    )
                    aoa_config["aoa_statements"].append(
                        f"{prefix}.self_attn.v_proj.weight -> {','.join(v_names)}, axis=0"
                    )

                    # Reassemble per-group in fleet order: Q_heads, Gate_heads, K, V
                    ordered = []
                    bias_ordered = []
                    for g in range(num_kv_heads):
                        for h in range(heads_per_group):
                            for c in range(head_dim_chunks):
                                ordered.append(f"{prefix}.self_attn._q_g{g}_h{h}_c{c}")
                                bias_ordered.append(f"{prefix}.self_attn._q_g{g}_h{h}_c{c}_bias")

                        if use_gated_attn:
                            for h in range(heads_per_group):
                                for c in range(v_head_dim_chunks):
                                    ordered.append(f"{prefix}.self_attn._gate_g{g}_h{h}_c{c}")
                                    bias_ordered.append(f"{prefix}.self_attn._gate_g{g}_h{h}_c{c}_bias")

                        for c in range(head_dim_chunks):
                            ordered.append(f"{prefix}.self_attn._k_g{g}_c{c}")
                            bias_ordered.append(f"{prefix}.self_attn._k_g{g}_c{c}_bias")
                        for c in range(v_head_dim_chunks):
                            ordered.append(f"{prefix}.self_attn._v_g{g}_c{c}")
                            bias_ordered.append(f"{prefix}.self_attn._v_g{g}_c{c}_bias")

                    fused_tmp = f"{prefix}.self_attn.qkv_fused_tmp"
                    aoa_config["aoa_statements"].append(f"{','.join(ordered)} -> {fused_tmp}, axis=0")
                    aoa_config["aoa_statements"].append(f"{fused_tmp}^T -> {prefix_offset}.self_attn.qkv_proj.weight")

                    if config.attention_bias:
                        aoa_config["aoa_statements"].append(
                            f"{prefix}.self_attn.q_proj.bias -> {','.join(q_bias_names)}, axis=0"
                        )
                        aoa_config["aoa_statements"].append(
                            f"{prefix}.self_attn.k_proj.bias -> {','.join(k_bias_names)}, axis=0"
                        )
                        aoa_config["aoa_statements"].append(
                            f"{prefix}.self_attn.v_proj.bias -> {','.join(v_bias_names)}, axis=0"
                        )
                        aoa_config["aoa_statements"].append(
                            f"{','.join(bias_ordered)} -> {prefix_offset}.self_attn.qkv_proj.bias, axis=0"
                        )

        moe_layer_start = config.first_k_dense_replace
        moe_layer_end = num_hidden_layers if config.use_dense_mtp else num_hidden_layers + num_nextn_predict_layers
        # All layers are MoE (first_k_dense_replace=0)
        for layer_idx in reversed(range(moe_layer_start, moe_layer_end)):
            layer_idx_offset = layer_idx + num_head_empty_layers
            prefix = f"model.layers.{layer_idx}"
            prefix_offset = f"{model_prefix}layers.{layer_idx_offset}"
            if layer_idx >= num_hidden_layers:
                # for mtp
                prefix_offset += ".transformer_layer"
            aoa_config["aoa_statements"] += [
                f"{prefix}.block_sparse_moe.e_score_correction_bias -> {prefix_offset}.mlp.gate.e_score_correction_bias",
                f"{prefix}.block_sparse_moe.gate.weight -> {prefix_offset}.mlp.gate.weight",
            ]

            if config.routed_scaling_factor_learnable:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.block_sparse_moe.gate.routed_scaling_factor_param -> {prefix_offset}.mlp.gate.routed_scaling_factor_param",
                ]

            if config.moe_latent_size is not None and config.moe_latent_size > 0:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.block_sparse_moe.fc1_latent_proj.weight^T -> {prefix_offset}.mlp.fc1_latent_proj.weight",
                    f"{prefix}.block_sparse_moe.fc2_latent_proj.weight^T -> {prefix_offset}.mlp.fc2_latent_proj.weight",
                ]

            if using_sonic_moe:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.block_sparse_moe.experts.$EXPERT_ID.w2.weight -> {prefix_offset}.mlp.experts.$EXPERT_ID.down_proj.weight",
                ]
            else:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.block_sparse_moe.experts.$EXPERT_ID.w2.weight^T -> {prefix_offset}.mlp.experts.$EXPERT_ID.down_proj.weight",
                ]

            if n_shared_experts > 0:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.block_sparse_moe.shared_experts.w1.weight^T, {prefix}.block_sparse_moe.shared_experts.w3.weight^T -> {prefix_offset}.mlp.shared_experts.up_gate_proj.weight, fused_ffn",
                    f"{prefix}.block_sparse_moe.shared_experts.w2.weight^T -> {prefix_offset}.mlp.shared_experts.down_proj.weight",
                ]

            for expert_id in range(config.n_routed_experts):
                if using_sonic_moe:
                    aoa_config["aoa_statements"] += [
                        f"{prefix}.block_sparse_moe.experts.{expert_id}.w1.weight, {prefix}.block_sparse_moe.experts.{expert_id}.w3.weight -> {prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight, axis=0",
                    ]
                else:
                    aoa_config["aoa_statements"] += [
                        f"{prefix}.block_sparse_moe.experts.{expert_id}.w1.weight^T, {prefix}.block_sparse_moe.experts.{expert_id}.w3.weight^T -> {prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight, axis=1",
                    ]

            use_fused_weight = config.moe_expert_fusion
            if config.fp8 and (config.moe_expert_fusion is False) and config.moe_deep_gemm:
                raise ValueError(
                    "For fp8 deep_gemm (i.e. use k-grouped gemm in backward), moe_expert_fusion must be True."
                )
            if config.fp8 and config.moe_expert_fusion and config.moe_deep_gemm is False:
                use_fused_weight = False

            if use_fused_weight:
                ep_weight1 = []
                ep_weight2 = []
                for expert_id in range(num_experts):
                    ep_weight1.append(f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight")
                    ep_weight2.append(f"{prefix_offset}.mlp.experts.{expert_id}.down_proj.weight")
                group_gemm1 = ",".join(ep_weight1)
                group_gemm2 = ",".join(ep_weight2)
                aoa_config["aoa_statements"] += [
                    f"{group_gemm1} -> {prefix_offset}.mlp.grouped_gemm_experts.weight1, axis=0",
                    f"{group_gemm2} -> {prefix_offset}.mlp.grouped_gemm_experts.weight2, axis=0",
                ]
            else:
                if config.get("fd_fallback", False):
                    ep_weight1 = []
                    ep_weight2 = []
                    for expert_id in range(num_experts):
                        ep_weight1.append(f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight")
                        ep_weight2.append(f"{prefix_offset}.mlp.experts.{expert_id}.down_proj.weight")
                    group1 = ",".join(ep_weight1)
                    group2 = ",".join(ep_weight2)
                    aoa_config["aoa_statements"] += [
                        f"{group1} -> {prefix_offset}.mlp.experts.up_gate_proj, axis=0",
                        f"{group2} -> {prefix_offset}.mlp.experts.down_proj, axis=0",
                    ]

        return aoa_config

    # NOTE: These aoa_config items will be removed later. The subsequent AOA parsing module will automatically generate the reverse AOA based on the forward (from_pretrained) AOA.
    @classmethod
    def _gen_inv_aoa_config(cls, config: MiniMaxM2Config):
        model_prefix = "" if cls == getattr(cls, "base_model_class", None) else "model."
        using_sonic_moe = config.using_sonic_moe
        if hasattr(config, "n_routed_experts"):
            num_experts = config.n_routed_experts
        else:
            num_experts = config.num_experts
        aoa_statements = [
            f"{model_prefix}norm.weight -> model.norm.weight",
        ]

        aoa_statements += [
            "model.embedding.embed_tokens.weight -> model.embed_tokens.weight",
        ]

        assert not (
            config.tie_word_embeddings and getattr(config, "separate_mtp_headloss", False)
        ), "tie_word_embeddings and separate_mtp_headloss cannot be enabled simultaneously in aoa"
        if config.tie_word_embeddings:
            aoa_statements += [f"{model_prefix}lm_head.weight -> _"]
        elif getattr(config, "separate_mtp_headloss", False):
            aoa_statements += [f"{model_prefix}shared_mtp_lm_head.weight -> lm_head.weight"]
            aoa_statements += [f"{model_prefix}shared_head.weight -> _"]
        else:
            aoa_statements += [f"{model_prefix}lm_head.weight -> lm_head.weight"]

        num_hidden_layers = config.num_hidden_layers
        num_head_empty_layers = (
            config.num_empty_layers_add_in_head
            if hasattr(config, "num_empty_layers_add_in_head") and config.num_empty_layers_add_in_head
            else 0
        )

        # NOTE: MiniMax-M2 has no dense layers (first_k_dense_replace=0)

        if config.mtp_num_layers > 0:
            num_nextn_predict_layers = config.mtp_num_layers
        else:
            num_nextn_predict_layers = config.num_nextn_predict_layers if config.num_nextn_predict_layers else 0

        n_shared_experts = config.n_shared_experts if hasattr(config, "n_shared_experts") else 0
        if n_shared_experts > 0:
            assert n_shared_experts == 1, f"n_shared_experts must be 0 or 1 in MiniMax-M2, but got {n_shared_experts}"

        # mtp layers
        for layer_idx in reversed(range(num_hidden_layers, num_hidden_layers + num_nextn_predict_layers)):
            layer_idx_offset = layer_idx + num_head_empty_layers
            prefix = f"model.layers.{layer_idx}"
            prefix_offset = f"{model_prefix}layers.{layer_idx_offset}"
            aoa_statements += [
                f"{prefix_offset}.eh_proj.weight^T -> {prefix}.eh_proj.weight",
                f"{prefix_offset}.enorm.weight -> {prefix}.enorm.weight",
                f"{prefix_offset}.hnorm.weight -> {prefix}.hnorm.weight",
                f"{prefix_offset}.norm.weight -> {prefix}.shared_head.norm.weight",
            ]

            # dense MTP: inverse mapping for dense MLP weights
            if config.use_dense_mtp:
                prefix_offset_tf = f"{prefix_offset}.transformer_layer"
                aoa_statements += [
                    f"{prefix_offset_tf}.mlp.up_gate_proj.weight -> {prefix}.mlp.gate_proj.weight, {prefix}.mlp.up_proj.weight, fused_ffn",
                    f"{prefix}.mlp.gate_proj.weight^T -> {prefix}.mlp.gate_proj.weight",
                    f"{prefix}.mlp.up_proj.weight^T -> {prefix}.mlp.up_proj.weight",
                    f"{prefix_offset_tf}.mlp.down_proj.weight^T -> {prefix}.mlp.down_proj.weight",
                ]

        # layer 0 -> layer num_hidden_layers-1
        for layer_idx in range(0, num_hidden_layers + num_nextn_predict_layers):
            layer_idx_offset = layer_idx + num_head_empty_layers
            prefix_offset = f"{model_prefix}layers.{layer_idx_offset}"
            prefix = f"model.layers.{layer_idx}"
            if layer_idx >= num_hidden_layers:
                # for mtp
                prefix_offset += ".transformer_layer"

            aoa_statements += [
                f"{prefix_offset}.input_layernorm.weight -> {prefix}.input_layernorm.weight",
                f"{prefix_offset}.post_attention_layernorm.weight -> {prefix}.post_attention_layernorm.weight",
                f"{prefix_offset}.self_attn.o_proj.weight^T -> {prefix}.self_attn.o_proj.weight",
            ]

            use_mla = bool(getattr(config, "multi_latent_attention", False))
            is_swa = is_layer_window_attention(config.sliding_window, config.window_attn_skip_freq, layer_idx)

            if config.use_gated_attn and use_mla:
                # MLA mode: gate_proj is a separate parameter
                aoa_statements += [
                    f"{prefix_offset}.self_attn.gate_proj.weight^T -> {prefix}.self_attn.gate_proj.weight",
                ]

            if config.use_vha_attention:
                aoa_statements += [
                    f"{prefix_offset}.self_attn.vha_premix_weight -> {prefix}.self_attn.vha_premix_weight",
                ]
                aoa_statements += [
                    f"{prefix_offset}.self_attn.vha_postmix_U -> {prefix}.self_attn.vha_postmix_U",
                ]
                aoa_statements += [
                    f"{prefix_offset}.self_attn.vha_postmix_V -> {prefix}.self_attn.vha_postmix_V",
                ]

            if (
                config.softmax_type == "learnable"
                or (config.add_full_attention_sink_bias and not is_swa)
                or (config.add_swa_attention_sink_bias and is_swa)
            ):
                aoa_statements += [
                    f"{prefix_offset}.self_attn.core_attention.softmax_offset -> {prefix}.self_attn.core_attention.softmax_offset",
                ]

            if use_mla:
                # MLA attention
                aoa_statements += [
                    f"{prefix_offset}.self_attn.q_a_proj.weight^T -> {prefix}.self_attn.q_a_proj.weight",
                    f"{prefix_offset}.self_attn.q_b_proj.weight^T -> {prefix}.self_attn.q_b_proj.weight",
                    f"{prefix_offset}.self_attn.kv_a_proj_with_mqa.weight^T -> {prefix}.self_attn.kv_a_proj_with_mqa.weight",
                    f"{prefix_offset}.self_attn.kv_b_proj.weight^T -> {prefix}.self_attn.kv_b_proj.weight",
                ]
                if config.use_qk_norm:
                    aoa_statements += [
                        f"{prefix_offset}.self_attn.q_a_layernorm.weight -> {prefix}.self_attn.q_a_layernorm.weight",
                        f"{prefix_offset}.self_attn.kv_a_layernorm.weight -> {prefix}.self_attn.kv_a_layernorm.weight",
                    ]
            elif config.experimental_attention_variant == "dsv4_hybrid":
                # csa_compress_ratios has length num_hidden_layers + num_nextn_predict_layers,
                # i.e. it covers both main layers and MTP layers.
                assert len(config.csa_compress_ratios) == num_hidden_layers + num_nextn_predict_layers, (
                    f"csa_compress_ratios length ({len(config.csa_compress_ratios)}) must equal "
                    f"num_hidden_layers + num_nextn_predict_layers "
                    f"({num_hidden_layers} + {num_nextn_predict_layers})"
                )
                csa_ratio = config.csa_compress_ratios[layer_idx]
                aoa_statements += [
                    # Linear projections (transpose: paddle [in, out] -> HF [out, in])
                    f"{prefix_offset}.self_attn.linear_q_down_proj.weight^T -> {prefix}.self_attn.linear_q_down_proj.weight",
                    f"{prefix_offset}.self_attn.linear_q_up_proj.weight^T -> {prefix}.self_attn.linear_q_up_proj.weight",
                    f"{prefix_offset}.self_attn.linear_kv_proj.weight^T -> {prefix}.self_attn.linear_kv_proj.weight",
                    f"{prefix_offset}.self_attn.o_proj.weight^T -> {prefix}.self_attn.o_proj.weight",
                    # Layer norms (no transpose, 1D)
                    f"{prefix_offset}.self_attn.q_layernorm.weight -> {prefix}.self_attn.q_layernorm.weight",
                    f"{prefix_offset}.self_attn.kv_layernorm.weight -> {prefix}.self_attn.kv_layernorm.weight",
                    # Grouped output projection (raw parameter, shape [out, in] on both sides)
                    f"{prefix_offset}.self_attn.linear_o_group_proj -> {prefix}.self_attn.linear_o_group_proj",
                    # Core attention: learnable attention sink (1D, no transpose)
                    f"{prefix_offset}.self_attn.core_attention.attn_sink -> {prefix}.self_attn.core_attention.attn_sink",
                ]
                # Compressor exists only when compress_ratio > 1 (i.e. ratio in {4, 128})
                if csa_ratio > 1:
                    aoa_statements += [
                        f"{prefix_offset}.self_attn.core_attention.compressor.linear_wkv.weight^T -> {prefix}.self_attn.core_attention.compressor.linear_wkv.weight",
                        f"{prefix_offset}.self_attn.core_attention.compressor.linear_wgate.weight^T -> {prefix}.self_attn.core_attention.compressor.linear_wgate.weight",
                        f"{prefix_offset}.self_attn.core_attention.compressor.norm.weight -> {prefix}.self_attn.core_attention.compressor.norm.weight",
                        f"{prefix_offset}.self_attn.core_attention.compressor.ape -> {prefix}.self_attn.core_attention.compressor.ape",
                    ]
                # Indexer exists only when compress_ratio == 4 and not csa_dense_mode
                if csa_ratio == 4 and not getattr(config, "csa_dense_mode", False):
                    aoa_statements += [
                        f"{prefix_offset}.self_attn.core_attention.indexer.linear_wq_b.weight^T -> {prefix}.self_attn.core_attention.indexer.linear_wq_b.weight",
                        f"{prefix_offset}.self_attn.core_attention.indexer.linear_weights_proj.weight^T -> {prefix}.self_attn.core_attention.indexer.linear_weights_proj.weight",
                        f"{prefix_offset}.self_attn.core_attention.indexer.compressor.linear_wkv.weight^T -> {prefix}.self_attn.core_attention.indexer.compressor.linear_wkv.weight",
                        f"{prefix_offset}.self_attn.core_attention.indexer.compressor.linear_wgate.weight^T -> {prefix}.self_attn.core_attention.indexer.compressor.linear_wgate.weight",
                        f"{prefix_offset}.self_attn.core_attention.indexer.compressor.norm.weight -> {prefix}.self_attn.core_attention.indexer.compressor.norm.weight",
                        f"{prefix_offset}.self_attn.core_attention.indexer.compressor.ape -> {prefix}.self_attn.core_attention.indexer.compressor.ape",
                    ]
            else:
                if config.use_qk_norm:
                    aoa_statements += [
                        f"{prefix_offset}.self_attn.q_norm.weight -> {prefix}.self_attn.q_norm.weight",
                        f"{prefix_offset}.self_attn.k_norm.weight -> {prefix}.self_attn.k_norm.weight",
                    ]

                num_heads, num_kv_heads, num_query_groups, split_dims, layer_is_swa = cls.get_layer_attn_split_info(
                    config, layer_idx + config.num_empty_layers_add_in_head
                )
                # Non-MLA gated attention: gate is fused in qkv_proj
                # Fleet layout per group: [Q_heads(hpg*hd), Gate_heads(hpg*hd), K(hd), V(hd)]
                # Need to split and reassemble to HF format

                if len(split_dims) == 4:
                    q_dim, gate_dim, head_dim, v_head_dim = split_dims
                    use_gated_attn = True
                else:
                    q_dim, head_dim, v_head_dim = split_dims
                    use_gated_attn = False

                gcd_head_dim = math.gcd(head_dim, v_head_dim)
                head_dim_chunks = head_dim // gcd_head_dim
                v_head_dim_chunks = v_head_dim // gcd_head_dim

                if config.use_vha_attention:
                    aoa_statements += [
                        f"{prefix_offset}.self_attn.q_proj.weight^T -> {prefix}.self_attn.q_proj.weight",
                        f"{prefix_offset}.self_attn.k_proj.weight^T -> {prefix}.self_attn.k_proj.weight",
                        f"{prefix_offset}.self_attn.v_proj.weight^T -> {prefix}.self_attn.v_proj.weight",
                    ]
                    if use_gated_attn:
                        aoa_statements.append(
                            f"{prefix_offset}.self_attn.gate_proj.weight^T -> {prefix}.self_attn.gate_proj.weight"
                        )
                    if config.attention_bias:
                        aoa_statements += [
                            f"{prefix_offset}.self_attn.q_proj.bias -> {prefix}.self_attn.q_proj.bias",
                            f"{prefix_offset}.self_attn.k_proj.bias -> {prefix}.self_attn.k_proj.bias",
                            f"{prefix_offset}.self_attn.v_proj.bias -> {prefix}.self_attn.v_proj.bias",
                        ]
                        if use_gated_attn:
                            aoa_statements.append(
                                f"{prefix_offset}.self_attn.gate_proj.bias -> {prefix}.self_attn.gate_proj.bias"
                            )

                else:
                    fleet_key = f"{prefix_offset}.self_attn.qkv_proj.weight"
                    fused_tmp = f"{prefix}.self_attn.qkv_fused_tmp"

                    # Step 1: Transpose fleet weight
                    aoa_statements.append(f"{fleet_key}^T -> {fused_tmp}")

                    heads_per_group = num_heads // num_kv_heads

                    # Step 2: Split into per-group chunks along axis=0
                    chunk_names = []
                    bias_chunk_names = []
                    for g in range(num_kv_heads):
                        for h in range(heads_per_group):
                            for c in range(head_dim_chunks):
                                chunk_names.append(f"{prefix}.self_attn._q_g{g}_h{h}_c{c}")
                                bias_chunk_names.append(f"{prefix}.self_attn._q_g{g}_h{h}_c{c}_bias")
                        if use_gated_attn:
                            for h in range(heads_per_group):
                                for c in range(v_head_dim_chunks):
                                    chunk_names.append(f"{prefix}.self_attn._gate_g{g}_h{h}_c{c}")
                                    bias_chunk_names.append(f"{prefix}.self_attn._gate_g{g}_h{h}_c{c}_bias")

                        for c in range(head_dim_chunks):
                            chunk_names.append(f"{prefix}.self_attn._k_g{g}_c{c}")
                            bias_chunk_names.append(f"{prefix}.self_attn._k_g{g}_c{c}_bias")
                        for c in range(v_head_dim_chunks):
                            chunk_names.append(f"{prefix}.self_attn._v_g{g}_c{c}")
                            bias_chunk_names.append(f"{prefix}.self_attn._v_g{g}_c{c}_bias")
                    aoa_statements.append(f"{fused_tmp} -> {','.join(chunk_names)}, axis=0")

                    # Step 3: Reassemble q_proj (interleaved Q+Gate)
                    q_ordered = []
                    bias_q_ordered = []
                    for g in range(num_kv_heads):
                        for h in range(heads_per_group):
                            for c in range(head_dim_chunks):
                                q_ordered.append(f"{prefix}.self_attn._q_g{g}_h{h}_c{c}")
                                bias_q_ordered.append(f"{prefix}.self_attn._q_g{g}_h{h}_c{c}_bias")
                            if use_gated_attn:
                                for c in range(v_head_dim_chunks):
                                    q_ordered.append(f"{prefix}.self_attn._gate_g{g}_h{h}_c{c}")
                                    bias_q_ordered.append(f"{prefix}.self_attn._gate_g{g}_h{h}_c{c}_bias")
                    aoa_statements.append(f"{','.join(q_ordered)} -> {prefix}.self_attn.q_proj.weight, axis=0")

                    # k_proj
                    k_ordered = []
                    bias_k_ordered = []
                    for g in range(num_kv_heads):
                        for c in range(head_dim_chunks):
                            k_ordered.append(f"{prefix}.self_attn._k_g{g}_c{c}")
                            bias_k_ordered.append(f"{prefix}.self_attn._k_g{g}_c{c}_bias")
                    aoa_statements.append(f"{','.join(k_ordered)} -> {prefix}.self_attn.k_proj.weight, axis=0")

                    # v_proj
                    v_ordered = []
                    bias_v_ordered = []
                    for g in range(num_kv_heads):
                        for c in range(v_head_dim_chunks):
                            v_ordered.append(f"{prefix}.self_attn._v_g{g}_c{c}")
                            bias_v_ordered.append(f"{prefix}.self_attn._v_g{g}_c{c}_bias")
                    aoa_statements.append(f"{','.join(v_ordered)} -> {prefix}.self_attn.v_proj.weight, axis=0")

                    if config.attention_bias:
                        aoa_statements.append(
                            f"{prefix_offset}.self_attn.qkv_proj.bias -> {','.join(bias_chunk_names)}, axis=0"
                        )
                        aoa_statements.append(f"{','.join(bias_q_ordered)} -> {prefix}.self_attn.q_proj.bias, axis=0")
                        aoa_statements.append(f"{','.join(bias_k_ordered)} -> {prefix}.self_attn.k_proj.bias, axis=0")
                        aoa_statements.append(f"{','.join(bias_v_ordered)} -> {prefix}.self_attn.v_proj.bias, axis=0")

        # All layers are MoE (first_k_dense_replace=0)
        moe_layer_end = num_hidden_layers if config.use_dense_mtp else num_hidden_layers + num_nextn_predict_layers
        for layer_idx in range(config.first_k_dense_replace, moe_layer_end):
            layer_idx_offset = layer_idx + num_head_empty_layers
            prefix_offset = f"{model_prefix}layers.{layer_idx_offset}"
            prefix = f"model.layers.{layer_idx}"
            if layer_idx >= num_hidden_layers:
                # for mtp
                prefix_offset += ".transformer_layer"

            use_fused_weight = config.moe_expert_fusion
            if config.fp8 and (config.moe_expert_fusion is False) and config.moe_deep_gemm:
                raise ValueError(
                    "For fp8 deep_gemm (i.e. use k-grouped gemm in backward), moe_expert_fusion must be True."
                )
            if config.fp8 and config.moe_expert_fusion and config.moe_deep_gemm is False:
                use_fused_weight = False

            if use_fused_weight:
                ep_weight1 = []
                ep_weight2 = []
                for expert_id in range(config.n_routed_experts):
                    ep_weight1.append(f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight")
                    ep_weight2.append(f"{prefix_offset}.mlp.experts.{expert_id}.down_proj.weight")
                group_gemm1 = ",".join(ep_weight1)
                group_gemm2 = ",".join(ep_weight2)
                aoa_statements += [
                    f"{prefix_offset}.mlp.grouped_gemm_experts.weight1 -> {group_gemm1}, axis=0",
                    f"{prefix_offset}.mlp.grouped_gemm_experts.weight2 -> {group_gemm2}, axis=0",
                ]
            else:
                if config.get("fd_fallback", False):
                    ep_weight1 = []
                    ep_weight2 = []
                    for expert_id in range(num_experts):
                        ep_weight1.append(f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight")
                        ep_weight2.append(f"{prefix_offset}.mlp.experts.{expert_id}.down_proj.weight")
                    group1 = ",".join(ep_weight1)
                    group2 = ",".join(ep_weight2)
                    aoa_statements += [
                        f"{prefix_offset}.mlp.experts.up_gate_proj -> {group1}, axis=0",
                        f"{prefix_offset}.mlp.experts.down_proj -> {group2}, axis=0",
                    ]

            if n_shared_experts > 0:
                aoa_statements += [
                    f"{prefix_offset}.mlp.shared_experts.down_proj.weight^T -> {prefix}.block_sparse_moe.shared_experts.w2.weight",
                    f"{prefix_offset}.mlp.shared_experts.up_gate_proj.weight -> {prefix_offset}.block_sparse_moe.shared_experts.gate_proj.weight, {prefix_offset}.block_sparse_moe.shared_experts.up_proj.weight, fused_ffn",
                    f"{prefix_offset}.block_sparse_moe.shared_experts.gate_proj.weight^T -> {prefix}.block_sparse_moe.shared_experts.w1.weight",
                    f"{prefix_offset}.block_sparse_moe.shared_experts.up_proj.weight^T -> {prefix}.block_sparse_moe.shared_experts.w3.weight",
                ]

            aoa_statements += [
                f"{prefix_offset}.mlp.gate.weight -> {prefix}.block_sparse_moe.gate.weight",
                f"{prefix_offset}.mlp.gate.e_score_correction_bias -> {prefix}.block_sparse_moe.e_score_correction_bias",
            ]

            if config.routed_scaling_factor_learnable:
                aoa_statements += [
                    f"{prefix_offset}.mlp.gate.routed_scaling_factor_param -> {prefix}.block_sparse_moe.gate.routed_scaling_factor_param",
                ]

            if config.moe_latent_size is not None and config.moe_latent_size > 0:
                aoa_statements += [
                    f"{prefix_offset}.mlp.fc1_latent_proj.weight^T -> {prefix}.block_sparse_moe.fc1_latent_proj.weight ",
                    f"{prefix_offset}.mlp.fc2_latent_proj.weight^T -> {prefix}.block_sparse_moe.fc2_latent_proj.weight",
                ]

            if using_sonic_moe:
                aoa_statements += [
                    f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight -> {prefix_offset}.block_sparse_moe.experts.{expert_id}.w1.weight, {prefix_offset}.block_sparse_moe.experts.{expert_id}.w3.weight, axis=0"
                    for expert_id in range(config.n_routed_experts)
                ]
            else:
                aoa_statements += [
                    f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight -> {prefix_offset}.block_sparse_moe.experts.{expert_id}.w1.weight, {prefix_offset}.block_sparse_moe.experts.{expert_id}.w3.weight, axis=1"
                    for expert_id in range(config.n_routed_experts)
                ]

            if not using_sonic_moe:
                aoa_statements += (
                    [
                        f"{prefix_offset}.block_sparse_moe.experts.{expert_id}.w1.weight^T -> {prefix}.block_sparse_moe.experts.{expert_id}.w1.weight"
                        for expert_id in range(config.n_routed_experts)
                    ]
                    + [
                        f"{prefix_offset}.block_sparse_moe.experts.{expert_id}.w3.weight^T -> {prefix}.block_sparse_moe.experts.{expert_id}.w3.weight"
                        for expert_id in range(config.n_routed_experts)
                    ]
                    + [
                        f"{prefix_offset}.mlp.experts.{expert_id}.down_proj.weight^T-> {prefix}.block_sparse_moe.experts.{expert_id}.w2.weight"
                        for expert_id in range(config.n_routed_experts)
                    ]
                )

        aoa_config = {"aoa_statements": aoa_statements}
        return aoa_config


class MiniMaxM2ForCausalLM(MiniMaxM2PreTrainedModel):
    is_fleet = True

    def __new__(cls, config):
        # Hybrid parallel config convert.
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)
        config.fuse_rms_norm = True
        # config.multi_latent_attention = True
        # config.rotary_interleaved = True

        model_provider_class = GLMMoEModelProvider
        model_provider = model_provider_class.from_config(config)
        loss_fn = None
        if getattr(config, "dpo_config", None):
            loss_fn = CriterionLayerPipe(config, use_infohub=True)
        gpt_model = model_provider.provide(loss_fn=loss_fn)
        gpt_model._gen_aoa_config = cls._gen_aoa_config
        gpt_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        gpt_model.build_muon_param_info_map = cls.build_muon_param_info_map
        gpt_model.config_to_save = config
        gpt_model.is_fleet = cls.is_fleet
        return gpt_model


class MiniMaxM2ForCausalLMPipe(MiniMaxM2PreTrainedModel, GeneralModelForCausalLMPipe):
    is_fleet = True

    def __new__(cls, config):
        # Hybrid parallel config convert.
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)
        config.fuse_rms_norm = True
        # config.multi_latent_attention = True
        # config.rotary_interleaved = True
        model_provider_class = GLMMoEModelProvider
        model_provider = model_provider_class.from_config(config)
        loss_fn = None
        if getattr(config, "dpo_config", None):
            loss_fn = CriterionLayerPipe(config, use_infohub=True)
        gpt_model = model_provider.provide(loss_fn=loss_fn)
        gpt_model._gen_aoa_config = cls._gen_aoa_config
        gpt_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        gpt_model.build_muon_param_info_map = cls.build_muon_param_info_map
        if not hasattr(config, "architectures"):
            config.architectures = [cls.__name__.replace("Pipe", "")]
        gpt_model.config_to_save = config
        gpt_model.is_fleet = cls.is_fleet
        return gpt_model


__all__ = [
    "MiniMaxM2ForCausalLMPipe",
    "MiniMaxM2ForCausalLM",
]
