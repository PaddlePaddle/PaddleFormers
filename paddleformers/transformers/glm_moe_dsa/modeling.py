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

from ...nn.pp_model import CriterionLayerPipe, GeneralModelForCausalLMPipe
from ..aoa_config_base import MoEAOAConfigGenerator
from ..glm4_moe.modeling import GLMMoEModelProvider
from ..model_utils import PretrainedModel
from .configuration import GlmMoeDsaConfig

logger = logging.getLogger(__name__)


class GlmMoeDsaPreTrainedModel(PretrainedModel):
    config: GlmMoeDsaConfig

    @classmethod
    def _gen_aoa_config(cls, config: GlmMoeDsaConfig):
        """Generate AOA config using the base class for minimal code duplication.

        GLM MoE DSA features:
        - Multi-Latent Attention (MLA) for efficient KV caching
        - Dense-to-MoE hybrid layers (first_k_dense_replace)
        - MTP (Multi-Token Prediction) support
        - Shared experts for routing efficiency

        Args:
            config: GlmMoeDsaConfig configuration object

        Returns:
            Dictionary with 'aoa_statements' key containing conversion statements
        """
        return MoEAOAConfigGenerator.gen_aoa_config(config)

    @classmethod
    def _gen_inv_aoa_config(cls, config: GlmMoeDsaConfig):
        """Generate inverse AOA config using the base class.

        Maps PaddleFleet weight names back to HuggingFace format,
        used during save_pretrained to convert weights back to HF convention.

        Args:
            config: GlmMoeDsaConfig configuration object

        Returns:
            Dictionary with 'aoa_statements' key containing inverse conversion statements
        """
        return MoEAOAConfigGenerator.gen_inv_aoa_config(config)

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

        def _qkv_per_head(matrix_2d_global, ortho_fn, kv_head_num=None, num_key_value_groups=None):
            import paddle

            head_dim = matrix_2d_global.shape[1] // (num_key_value_groups * kv_head_num + 2 * kv_head_num)
            groups = paddle.split(matrix_2d_global, kv_head_num, axis=1)

            processed_groups = []
            for group in groups:
                q_part, k_head, v_head = paddle.split(
                    group,
                    [num_key_value_groups * head_dim, head_dim, head_dim],
                    axis=1,
                )
                q_heads = paddle.split(q_part, num_key_value_groups, axis=1)
                q_ortho = paddle.concat([ortho_fn(h) for h in q_heads], axis=1)
                processed_groups.append(paddle.concat([q_ortho, ortho_fn(k_head), ortho_fn(v_head)], axis=1))

            return paddle.concat(processed_groups, axis=1)

        def _qkv_sep(matrix_2d, ortho_fn, kv_head_num=None, num_key_value_groups=None):
            import paddle

            head_dim = matrix_2d.shape[1] // (num_key_value_groups * kv_head_num + 2 * kv_head_num)
            q_group_size = num_key_value_groups * head_dim

            groups = paddle.split(matrix_2d, kv_head_num, axis=1)
            q_parts, k_parts, v_parts = [], [], []
            for group in groups:
                q_p, k_p, v_p = paddle.split(group, [q_group_size, head_dim, head_dim], axis=1)
                q_parts.append(q_p)
                k_parts.append(k_p)
                v_parts.append(v_p)

            q_ortho = ortho_fn(paddle.concat(q_parts, axis=1))
            k_ortho = ortho_fn(paddle.concat(k_parts, axis=1))
            v_ortho = ortho_fn(paddle.concat(v_parts, axis=1))

            q_groups = paddle.split(q_ortho, kv_head_num, axis=1)
            k_groups = paddle.split(k_ortho, kv_head_num, axis=1)
            v_groups = paddle.split(v_ortho, kv_head_num, axis=1)

            return paddle.concat(
                [paddle.concat([q_groups[i], k_groups[i], v_groups[i]], axis=1) for i in range(kv_head_num)],
                axis=1,
            )

        def _ffn_gate_up(matrix, ortho_fn, intermediate_size=None):
            import paddle

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
            import paddle

            split_args = head_num if head_split_sizes is None else head_split_sizes * head_num
            groups = paddle.split(matrix_2d_global, split_args, axis=axis)
            processed_groups = [ortho_fn(group) for group in groups]
            return paddle.concat(processed_groups, axis=axis)

        def _moe_experts(matrix_3d_global, ortho_fn):
            import paddle

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
        num_attention_head = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        num_key_value_groups = num_attention_head // num_key_value_heads
        use_mla = getattr(config, "q_lora_rank", None) and config.q_lora_rank > 0
        moe_expert_fusion = getattr(config, "moe_expert_fusion", False)
        use_gated_attn = getattr(config, "use_gated_attn", False)

        muon_qkv_update_mode = muon_configs.get("muon_qkv_update_mode", "split_head")
        muon_ffn_split = muon_configs.get("muon_ffn_split", False)

        qkv_slice_fn = None
        qkv_kwargs = {}
        if muon_qkv_update_mode == "split_head":
            qkv_slice_fn = _qkv_per_head
            qkv_kwargs = {"kv_head_num": num_key_value_heads, "num_key_value_groups": num_key_value_groups}
        elif muon_qkv_update_mode == "split_qkv":
            qkv_slice_fn = _qkv_sep
            qkv_kwargs = {"kv_head_num": num_key_value_heads, "num_key_value_groups": num_key_value_groups}

        ffn_slice_fn = _ffn_gate_up if muon_ffn_split else None

        fused_moe_fn = _moe_experts if moe_expert_fusion else None

        mla_slice_fn = None
        if use_mla and muon_qkv_update_mode == "split_head":
            mla_slice_fn = _mla_per_head

        def _add_layer_slice_config(prefix):
            if not use_mla and qkv_slice_fn is not None:
                slice_config[f"{prefix}.self_attn.qkv_proj.weight"] = (qkv_slice_fn, qkv_kwargs.copy())

            if ffn_slice_fn is not None:
                moe_intermediate_size = config.moe_intermediate_size
                intermediate_size = config.intermediate_size

                param_name = f"{prefix}.mlp.experts.up_gate_proj.weight"
                slice_config[param_name] = (ffn_slice_fn, {"intermediate_size": moe_intermediate_size})

                slice_config[f"{prefix}.mlp.shared_experts.up_gate_proj.weight"] = (
                    ffn_slice_fn,
                    {"intermediate_size": moe_intermediate_size},
                )
                slice_config[f"{prefix}.mlp.grouped_gemm_experts.weight1"] = (
                    ffn_slice_fn,
                    {"intermediate_size": moe_intermediate_size},
                )
                param_name = f"{prefix}.mlp.up_gate_proj.weight"
                slice_config[param_name] = (ffn_slice_fn, {"intermediate_size": intermediate_size})

                if hasattr(config, "n_routed_experts") and config.n_routed_experts > 0:
                    for expert_idx in range(config.n_routed_experts):
                        slice_config[f"{prefix}.mlp.experts.{expert_idx}.up_gate_proj.weight"] = (
                            ffn_slice_fn,
                            {"intermediate_size": moe_intermediate_size},
                        )

            if moe_expert_fusion and fused_moe_fn is not None:
                slice_config[f"{prefix}.mlp.experts.down_proj.weight"] = (fused_moe_fn, {})
                slice_config[f"{prefix}.mlp.grouped_gemm_experts.weight2"] = (fused_moe_fn, {})

            if use_mla and mla_slice_fn is not None:
                assert (
                    hasattr(config, "qk_nope_head_dim")
                    and hasattr(config, "qk_rope_head_dim")
                    and hasattr(config, "kv_lora_rank")
                    and hasattr(config, "v_head_dim")
                )

                slice_config[f"{prefix}.self_attn.q_b_proj.weight"] = (
                    mla_slice_fn,
                    {
                        "head_num": num_attention_head,
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
                        "head_num": num_attention_head,
                        "axis": 1,
                        "head_split_sizes": [config.qk_nope_head_dim, config.v_head_dim],
                    },
                )

            if use_gated_attn and mla_slice_fn is not None:
                slice_config[f"{prefix}.self_attn.gate_proj.weight"] = (
                    mla_slice_fn,
                    {"head_num": num_attention_head, "axis": 1},
                )

        for layer_idx in range(num_hidden_layers):
            _add_layer_slice_config(f"model.layers.{layer_idx}")

        mtp_num_layers = getattr(config, "mtp_num_layers", 0)
        if mtp_num_layers > 0:
            num_nextn_predict_layers = mtp_num_layers
        else:
            num_nextn_predict_layers = config.num_nextn_predict_layers if config.num_nextn_predict_layers else 0
        for layer_idx in range(num_nextn_predict_layers):
            _add_layer_slice_config(f"model.layers.{num_hidden_layers + layer_idx}")
        for layer_idx in range(num_nextn_predict_layers):
            _add_layer_slice_config(f"model.layers.{num_hidden_layers + layer_idx}.transformer_layer")

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

        slice_config = cls._build_muon_slice_config(model, config)

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
            use_muon = _default_should_use_muon(name, param.shape, exclude_patterns) and _default_should_use_muon(
                param.name, param.shape, exclude_patterns
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


class GlmMoeDsaForCausalLM(GlmMoeDsaPreTrainedModel):
    is_fleet = True

    def __new__(cls, config):
        # Hybrid parallel config convert.
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)
        config.fuse_rms_norm = True
        config.multi_latent_attention = True
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


class GlmMoeDsaForCausalLMPipe(GlmMoeDsaPreTrainedModel, GeneralModelForCausalLMPipe):
    is_fleet = True

    def __new__(cls, config):
        # Hybrid parallel config convert.
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)
        config.fuse_rms_norm = True
        config.multi_latent_attention = True
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
    "GlmMoeDsaForCausalLMPipe",
    "GlmMoeDsaForCausalLM",
]
