# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
import types

import paddle
import paddle.nn.functional as F
from paddle import nn

from ...nn.criterion.interface import CriterionLayer
from ..model_outputs import BaseModelOutputWithPooling
from ..model_utils import PretrainedModel
from ..qwen3_vl.modeling import Qwen3VLVisionModel
from .configuration import Qwen3_5VisionConfig
from .modeling_fleet import build_qwen3_5_model


class Qwen3_5VisionModel(Qwen3VLVisionModel):
    config_class = Qwen3_5VisionConfig
    _no_split_modules = ["Qwen3VLVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        if not hasattr(self, "pos_embed"):
            self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        del self.deepstack_visual_indexes
        del self.deepstack_merger_list

    def forward(self, hidden_states: paddle.Tensor, grid_thw: paddle.Tensor, **kwargs) -> paddle.Tensor:
        """
        Args:
            hidden_states (`paddle.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`paddle.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `paddle.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.reshape([seq_len, -1])
        rotary_pos_emb = rotary_pos_emb.reshape([seq_len, -1])
        emb = paddle.concat([rotary_pos_emb, rotary_pos_emb], axis=-1)
        position_embeddings = (paddle.cos(emb), paddle.sin(emb))

        cu_seqlens = paddle.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            axis=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - paddle.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            dtype=grid_thw.dtype if not paddle.in_dynamic_mode() else "int32",
        )
        cu_seqlens = F.pad(cu_seqlens, [1, 0], value=0)

        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        indices_per_segment = paddle.stack(
            [
                cu_seqlens[1:],
                paddle.full_like(cu_seqlens[1:], cu_seqlens[-1]),
                paddle.zeros_like(cu_seqlens[:-1]),
                cu_seqlens[:-1],
            ],
            axis=1,
        )
        attn_mask_startend_row_indices = paddle.repeat_interleave(indices_per_segment, lengths, axis=0)[
            None, None, ...
        ]

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                **kwargs,
            )

        merged_hidden_states = self.merger(hidden_states)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )


class Qwen3_5ForConditionalGeneration(PretrainedModel):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    is_fleet = True

    @classmethod
    def _gen_aoa_config(cls, config):
        mapping = cls._checkpoint_conversion_mapping
        llm_target = next((v for v in mapping.values() if "language_model" in v), "language_model")
        visual_target = "model.vision_model"
        llm_prefix = f"{llm_target}." if not llm_target.endswith(".") else llm_target
        visual_prefix = f"{visual_target}." if not visual_target.endswith(".") else visual_target

        text_config = config.text_config
        vision_config = config.vision_config

        layer_types = getattr(text_config, "layer_types", None)
        if layer_types is None:
            layer_types = ["full_attention"] * text_config.num_hidden_layers
        full_attn_layers = [i for i, lt in enumerate(layer_types) if lt == "full_attention"]
        linear_attn_layers = [i for i, lt in enumerate(layer_types) if lt == "linear_attention"]

        # language model — embedding & final norm
        aoa_config = {
            "aoa_statements": [
                f"model.language_model.embed_tokens.weight -> {llm_prefix}embedding.embed_tokens.weight",
                f"model.language_model.norm.weight -> {llm_prefix}norm.weight",
            ]
        }

        # language model — layer norms (common to all layer types)
        aoa_config["aoa_statements"] += [
            f"model.language_model.layers.$LAYER_ID.input_layernorm.weight -> {llm_prefix}layers.$LAYER_ID.input_layernorm.weight",
            f"model.language_model.layers.$LAYER_ID.post_attention_layernorm.weight -> {llm_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
        ]

        # ── full_attention layers: fused QKV, o_proj, qk norms ──
        gated_attention = getattr(text_config, "attn_output_gate", False)
        num_heads = text_config.num_attention_heads
        num_kv_heads = text_config.num_key_value_heads
        heads_per_group = num_heads // num_kv_heads  # e.g. 8

        if gated_attention:
            # HF q_proj includes gate: shape [num_heads * head_dim * 2, hidden_size]
            # Layout (dim-0): [Q_h0(hd), G_h0(hd), Q_h1(hd), G_h1(hd), ...]
            # Fleet expects per-group: [Q_heads(hpg*hd), Gate_heads(hpg*hd), K(hd), V(hd)]
            # Need to rearrange Q+Gate interleaved → Q separated, Gate separated
            for i in full_attn_layers:
                hf_pre = f"model.language_model.layers.{i}.self_attn"
                # Step 1: Split q_proj into 2*num_heads equal chunks (each = head_dim)
                # Even chunks = Q heads, Odd chunks = Gate heads
                n_chunks = 2 * num_heads  # 32
                qg_names = [f"{hf_pre}.q_proj._qg{c}" for c in range(n_chunks)]
                aoa_config["aoa_statements"].append(f"{hf_pre}.q_proj.weight -> {','.join(qg_names)}, axis=0")
                # Step 2: Split k_proj and v_proj into num_kv_heads chunks
                k_names = [f"{hf_pre}.k_proj._kh{c}" for c in range(num_kv_heads)]
                v_names = [f"{hf_pre}.v_proj._vh{c}" for c in range(num_kv_heads)]
                aoa_config["aoa_statements"].append(f"{hf_pre}.k_proj.weight -> {','.join(k_names)}, axis=0")
                aoa_config["aoa_statements"].append(f"{hf_pre}.v_proj.weight -> {','.join(v_names)}, axis=0")
                # Step 3: Assemble per-group in fleet order and concat
                # Per group g: Q_heads (even chunks), Gate_heads (odd chunks), K, V
                ordered = []
                for g in range(num_kv_heads):
                    base = g * heads_per_group * 2
                    # Q heads for this group (even indices within group)
                    for h in range(heads_per_group):
                        ordered.append(qg_names[base + h * 2])
                    # Gate heads for this group (odd indices within group)
                    for h in range(heads_per_group):
                        ordered.append(qg_names[base + h * 2 + 1])
                    ordered.append(k_names[g])
                    ordered.append(v_names[g])
                fused_tmp = f"{hf_pre}.qkv_fused_tmp"
                aoa_config["aoa_statements"].append(f"{','.join(ordered)} -> {fused_tmp}, axis=0")
                # Step 4: Transpose the fused weight
                aoa_config["aoa_statements"].append(
                    f"{fused_tmp}^T -> {llm_prefix}layers.{i}.self_attn.qkv_proj.weight"
                )
        else:
            aoa_config["aoa_statements"] += [
                f"model.language_model.layers.{i}.self_attn.q_proj.weight^T, model.language_model.layers.{i}.self_attn.k_proj.weight^T, model.language_model.layers.{i}.self_attn.v_proj.weight^T -> {llm_prefix}layers.{i}.self_attn.qkv_proj.weight, fused_qkv, num_heads={num_heads}, num_key_value_groups={num_kv_heads}"
                for i in full_attn_layers
            ]
        if getattr(text_config, "attention_bias", False):
            aoa_config["aoa_statements"] += [
                f"model.language_model.layers.{i}.self_attn.q_proj.bias, model.language_model.layers.{i}.self_attn.k_proj.bias, model.language_model.layers.{i}.self_attn.v_proj.bias -> {llm_prefix}layers.{i}.self_attn.qkv_proj.bias, fused_qkv, num_heads={text_config.num_attention_heads}, num_key_value_groups={text_config.num_key_value_heads}"
                for i in full_attn_layers
            ]
        aoa_config["aoa_statements"] += [
            f"model.language_model.layers.{i}.self_attn.o_proj.weight^T -> {llm_prefix}layers.{i}.self_attn.o_proj.weight"
            for i in full_attn_layers
        ]
        aoa_config["aoa_statements"] += [
            f"model.language_model.layers.{i}.self_attn.{x}_norm.weight -> {llm_prefix}layers.{i}.self_attn.{x}_norm.weight"
            for i in full_attn_layers
            for x in ("q", "k")
        ]

        # ── linear_attention layers: fused in_proj (qkv+z+b+a), conv1d, dt_bias, A_log, out_norm, out_proj ──
        # HF has 4 separate projections; fleet fuses them into a single in_proj
        # Fleet split order: [qkv, z(gate), beta, alpha]
        # Step 1: concat 4 HF weights (no transpose) along axis=0 into virtual intermediate
        # Step 2: transpose the intermediate -> fleet target
        for i in linear_attn_layers:
            hf_pre = f"model.language_model.layers.{i}.linear_attn"
            fused_tmp = f"model.language_model.layers.{i}.linear_attn.in_proj_fused_tmp"
            aoa_config["aoa_statements"] += [
                f"{hf_pre}.in_proj_qkv.weight, {hf_pre}.in_proj_z.weight, {hf_pre}.in_proj_b.weight, {hf_pre}.in_proj_a.weight -> {fused_tmp}, axis=0",
                f"{fused_tmp}^T -> {llm_prefix}layers.{i}.self_attn.in_proj.weight",
            ]
        aoa_config["aoa_statements"] += [
            f"model.language_model.layers.{i}.linear_attn.conv1d.weight -> {llm_prefix}layers.{i}.self_attn.conv1d.weight, dtype='bfloat16'"
            for i in linear_attn_layers
        ]
        aoa_config["aoa_statements"] += [
            f"model.language_model.layers.{i}.linear_attn.dt_bias -> {llm_prefix}layers.{i}.self_attn.dt_bias, dtype='bfloat16'"
            for i in linear_attn_layers
        ]
        aoa_config["aoa_statements"] += [
            f"model.language_model.layers.{i}.linear_attn.A_log -> {llm_prefix}layers.{i}.self_attn.A_log, dtype='bfloat16'"
            for i in linear_attn_layers
        ]
        aoa_config["aoa_statements"] += [
            f"model.language_model.layers.{i}.linear_attn.norm.weight -> {llm_prefix}layers.{i}.self_attn.out_norm.weight, dtype='bfloat16'"
            for i in linear_attn_layers
        ]
        aoa_config["aoa_statements"] += [
            f"model.language_model.layers.{i}.linear_attn.out_proj.weight^T -> {llm_prefix}layers.{i}.self_attn.out_proj.weight"
            for i in linear_attn_layers
        ]

        # ── MLP (dense or MoE, depending on model config) ──
        # Qwen3_5TextConfig has num_experts=60 as class default even for dense models,
        # so we use model_type to distinguish: "moe" in model_type means MoE variant
        is_moe = "moe" in getattr(config, "model_type", "")
        num_experts = getattr(text_config, "num_experts", 0) or getattr(text_config, "n_routed_experts", 0)
        if is_moe and num_experts > 0:
            # MoE — router gate
            aoa_config["aoa_statements"] += [
                f"model.language_model.layers.{i}.mlp.gate.weight -> {llm_prefix}layers.{i}.mlp.gate.weight, dtype='float32'"
                for i in range(text_config.num_hidden_layers)
            ]
            # MoE — routed experts
            for i in range(text_config.num_hidden_layers):
                if getattr(config, "moe_grouped_gemm", True):
                    aoa_config["aoa_statements"] += [
                        f'model.language_model.layers.{i}.mlp.experts.gate_up_proj -> {llm_prefix}layers.{i}.mlp.grouped_gemm_experts.weight1, permute="[0, 2, 1]"',
                        f'model.language_model.layers.{i}.mlp.experts.down_proj -> {llm_prefix}layers.{i}.mlp.grouped_gemm_experts.weight2, permute="[0, 2, 1]"',
                    ]
                else:
                    split_experts_up_gate = ""
                    split_experts_down = ""
                    for expert_id in range(num_experts):
                        split_experts_up_gate += f"{llm_prefix}layers.{i}.mlp.experts.{expert_id}.up_gate_proj.weight,"
                        split_experts_down += f"{llm_prefix}layers.{i}.mlp.experts.{expert_id}.down_proj.weight,"
                    split_experts_down += "axis=0"
                    split_experts_up_gate += "axis=0"
                    aoa_config["aoa_statements"] += [
                        f"model.language_model.layers.{i}.mlp.experts.gate_up_proj -> {split_experts_up_gate}",
                        f"model.language_model.layers.{i}.mlp.experts.down_proj -> {split_experts_down}",
                    ]
            # MoE — shared experts
            shared_expert_intermediate_size = getattr(text_config, "shared_expert_intermediate_size", 0)
            if shared_expert_intermediate_size and shared_expert_intermediate_size > 0:
                aoa_config["aoa_statements"] += [
                    f"model.language_model.layers.{i}.mlp.shared_expert.gate_proj.weight^T, model.language_model.layers.{i}.mlp.shared_expert.up_proj.weight^T -> {llm_prefix}layers.{i}.mlp.shared_experts.up_gate_proj.weight, fused_ffn"
                    for i in range(text_config.num_hidden_layers)
                ]
                aoa_config["aoa_statements"] += [
                    f"model.language_model.layers.{i}.mlp.shared_expert.down_proj.weight^T -> {llm_prefix}layers.{i}.mlp.shared_experts.down_proj.weight"
                    for i in range(text_config.num_hidden_layers)
                ]
                aoa_config["aoa_statements"] += [
                    f"model.language_model.layers.{i}.mlp.shared_expert_gate.weight^T -> {llm_prefix}layers.{i}.mlp.shared_experts.gate_weight"
                    for i in range(text_config.num_hidden_layers)
                ]
        else:
            # Dense MLP (SwiGLU: gate_proj + up_proj fused, down_proj)
            aoa_config["aoa_statements"] += [
                f"model.language_model.layers.{i}.mlp.gate_proj.weight^T, model.language_model.layers.{i}.mlp.up_proj.weight^T -> {llm_prefix}layers.{i}.mlp.up_gate_proj.weight, fused_ffn"
                for i in range(text_config.num_hidden_layers)
            ]
            aoa_config["aoa_statements"] += [
                f"model.language_model.layers.{i}.mlp.down_proj.weight^T -> {llm_prefix}layers.{i}.mlp.down_proj.weight"
                for i in range(text_config.num_hidden_layers)
            ]

        # ── visual model — attention qkv ──
        # Fleet sharded_state_dict uses: model.vision_model.layers.{i} (NOT decoder.layers)
        # LayerNorm is remapped: input_layernorm -> self_attn.qkv_proj.layer_norm_*
        #                        post_attention_layernorm -> mlp.up_gate_proj.layer_norm_*
        aoa_config["aoa_statements"] += [
            stmt
            for layer_id in range(vision_config.depth)
            for stmt in (
                f"model.visual.blocks.{layer_id}.attn.qkv.weight -> model.visual.blocks.{layer_id}.attn.q.weight, model.visual.blocks.{layer_id}.attn.k.weight,model.visual.blocks.{layer_id}.attn.v.weight,axis=0",
                f"model.visual.blocks.{layer_id}.attn.q.weight^T, model.visual.blocks.{layer_id}.attn.k.weight^T, model.visual.blocks.{layer_id}.attn.v.weight^T -> {visual_prefix}layers.{layer_id}.self_attn.qkv_proj.weight,fused_qkv, num_heads={vision_config.num_heads}, num_key_value_groups={vision_config.num_heads}",
                f"model.visual.blocks.{layer_id}.attn.qkv.bias -> model.visual.blocks.{layer_id}.attn.q.bias, model.visual.blocks.{layer_id}.attn.k.bias, model.visual.blocks.{layer_id}.attn.v.bias,axis=0",
                f"model.visual.blocks.{layer_id}.attn.q.bias, model.visual.blocks.{layer_id}.attn.k.bias, model.visual.blocks.{layer_id}.attn.v.bias -> {visual_prefix}layers.{layer_id}.self_attn.qkv_proj.bias, fused_qkv, num_heads={vision_config.num_heads}, num_key_value_groups={vision_config.num_heads},axis=0",
            )
        ]
        # visual model — o_proj, mlp, norms, patch_embed, pos_embed, merger
        aoa_config["aoa_statements"] += (
            [
                f"model.visual.blocks.$LAYER_ID.attn.proj.weight^T -> {visual_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
                f"model.visual.blocks.$LAYER_ID.attn.proj.bias -> {visual_prefix}layers.$LAYER_ID.self_attn.o_proj.bias",
            ]
            + [
                f"model.visual.blocks.$LAYER_ID.mlp.{x}.weight^T -> {visual_prefix}layers.$LAYER_ID.mlp.{y}.weight"
                for x, y in (("linear_fc1", "up_gate_proj"), ("linear_fc2", "down_proj"))
            ]
            + [
                f"model.visual.blocks.$LAYER_ID.mlp.{x}.bias -> {visual_prefix}layers.$LAYER_ID.mlp.{y}.bias"
                for x, y in (("linear_fc1", "up_gate_proj"), ("linear_fc2", "down_proj"))
            ]
        )
        aoa_config["aoa_statements"] += [
            f"model.visual.patch_embed.proj.weight -> {visual_prefix}patch_embed.weight",
            f"model.visual.patch_embed.proj.bias -> {visual_prefix}patch_embed.bias",
            f"model.visual.pos_embed.weight -> {visual_prefix}pos_embed.weight",
            f"model.visual.merger.norm.weight -> {visual_prefix}merger.norm.weight",
            f"model.visual.merger.norm.bias -> {visual_prefix}merger.norm.bias",
            # LayerNorm keys are remapped by sharded_state_dict_keys_map:
            f"model.visual.blocks.$LAYER_ID.norm1.weight -> {visual_prefix}layers.$LAYER_ID.input_layernorm.weight",
            f"model.visual.blocks.$LAYER_ID.norm1.bias -> {visual_prefix}layers.$LAYER_ID.input_layernorm.bias",
            f"model.visual.blocks.$LAYER_ID.norm2.weight -> {visual_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
            f"model.visual.blocks.$LAYER_ID.norm2.bias -> {visual_prefix}layers.$LAYER_ID.post_attention_layernorm.bias",
        ]
        aoa_config["aoa_statements"] += [
            f"model.visual.merger.linear_fc1.weight^T -> {visual_prefix}merger.mlp.up_gate_proj.weight",
            f"model.visual.merger.linear_fc1.bias -> {visual_prefix}merger.mlp.up_gate_proj.bias",
            f"model.visual.merger.linear_fc2.weight^T -> {visual_prefix}merger.mlp.down_proj.weight",
            f"model.visual.merger.linear_fc2.bias -> {visual_prefix}merger.mlp.down_proj.bias",
        ]

        # lm_head
        if cls._tied_weights_keys:
            aoa_config["aoa_statements"] += [
                f"{'model.language_model.embed_tokens.weight' if config.tie_word_embeddings else 'lm_head.weight'} -> {llm_prefix}lm_head.weight",
            ]

        return aoa_config

    @classmethod
    def _gen_inv_aoa_config(cls, config):
        mapping = cls._checkpoint_conversion_mapping
        llm_target = next((v for v in mapping.values() if "language_model" in v), "language_model")
        visual_target = "model.vision_model"
        llm_prefix = f"{llm_target}." if not llm_target.endswith(".") else llm_target
        visual_prefix = f"{visual_target}." if not visual_target.endswith(".") else visual_target

        text_config = config.text_config
        vision_config = config.vision_config

        layer_types = getattr(text_config, "layer_types", None)
        if layer_types is None:
            layer_types = ["full_attention"] * text_config.num_hidden_layers
        full_attn_layers = [i for i, lt in enumerate(layer_types) if lt == "full_attention"]
        linear_attn_layers = [i for i, lt in enumerate(layer_types) if lt == "linear_attention"]

        # language model — embedding & final norm
        aoa_config = {
            "aoa_statements": [
                f"{llm_prefix}embedding.embed_tokens.weight -> model.language_model.embed_tokens.weight",
                f"{llm_prefix}norm.weight -> model.language_model.norm.weight",
            ]
        }

        # language model — layer norms (common to all layer types)
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.language_model.layers.$LAYER_ID.input_layernorm.weight",
            f"{llm_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.language_model.layers.$LAYER_ID.post_attention_layernorm.weight",
        ]

        # ── full_attention layers: inverse fused QKV, o_proj, qk norms ──
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.{i}.self_attn.qkv_proj.weight -> model.language_model.layers.{i}.self_attn.q_proj.weight, model.language_model.layers.{i}.self_attn.k_proj.weight, model.language_model.layers.{i}.self_attn.v_proj.weight, fused_qkv, num_heads={text_config.num_attention_heads}, num_key_value_groups={text_config.num_key_value_heads}"
            for i in full_attn_layers
        ]
        if getattr(text_config, "attention_bias", False):
            aoa_config["aoa_statements"] += [
                f"{llm_prefix}layers.{i}.self_attn.qkv_proj.bias -> model.language_model.layers.{i}.self_attn.q_proj.bias, model.language_model.layers.{i}.self_attn.k_proj.bias, model.language_model.layers.{i}.self_attn.v_proj.bias, fused_qkv, num_heads={text_config.num_attention_heads}, num_key_value_groups={text_config.num_key_value_heads}"
                for i in full_attn_layers
            ]
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.{i}.self_attn.{x}_proj.weight^T -> model.language_model.layers.{i}.self_attn.{x}_proj.weight"
            for i in full_attn_layers
            for x in ("q", "k", "v")
        ]
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.{i}.self_attn.o_proj.weight^T -> model.language_model.layers.{i}.self_attn.o_proj.weight"
            for i in full_attn_layers
        ]
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.{i}.self_attn.{x}_norm.weight -> model.language_model.layers.{i}.self_attn.{x}_norm.weight"
            for i in full_attn_layers
            for x in ("q", "k")
        ]

        # ── linear_attention layers: inverse fused in_proj, conv1d, dt_bias, A_log, out_norm, out_proj ──
        # Fleet fused in_proj -> split back to HF's 4 separate projections
        # Fleet split order: [qkv, z(gate), beta, alpha]
        # Step 1: transpose fleet in_proj -> intermediate
        # Step 2: split intermediate along axis=0 into 4 HF weights
        for i in linear_attn_layers:
            hf_pre = f"model.language_model.layers.{i}.linear_attn"
            fused_tmp = f"model.language_model.layers.{i}.linear_attn.in_proj_fused_tmp"
            aoa_config["aoa_statements"] += [
                f"{llm_prefix}layers.{i}.self_attn.in_proj.weight^T -> {fused_tmp}",
                f"{fused_tmp} -> {hf_pre}.in_proj_qkv.weight, {hf_pre}.in_proj_z.weight, {hf_pre}.in_proj_b.weight, {hf_pre}.in_proj_a.weight, axis=0",
            ]
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.{i}.self_attn.conv1d.weight -> model.language_model.layers.{i}.linear_attn.conv1d.weight"
            for i in linear_attn_layers
        ]
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.{i}.self_attn.dt_bias -> model.language_model.layers.{i}.linear_attn.dt_bias"
            for i in linear_attn_layers
        ]
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.{i}.self_attn.A_log -> model.language_model.layers.{i}.linear_attn.A_log"
            for i in linear_attn_layers
        ]
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.{i}.self_attn.out_norm.weight -> model.language_model.layers.{i}.linear_attn.norm.weight"
            for i in linear_attn_layers
        ]
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.{i}.self_attn.out_proj.weight^T -> model.language_model.layers.{i}.linear_attn.out_proj.weight"
            for i in linear_attn_layers
        ]

        # ── MoE — routed experts (all layers) ──
        # Fleet grouped_gemm [num_experts, in_features, out_features] -> HF [num_experts, out_features, in_features]
        aoa_config["aoa_statements"] += [
            state
            for i in range(text_config.num_hidden_layers)
            for state in (
                f'{llm_prefix}layers.{i}.mlp.grouped_gemm_experts.weight1 -> model.language_model.layers.{i}.mlp.experts.gate_up_proj, permute="[0, 2, 1]"',
                f'{llm_prefix}layers.{i}.mlp.grouped_gemm_experts.weight2 -> model.language_model.layers.{i}.mlp.experts.down_proj, permute="[0, 2, 1]"',
            )
        ]

        # ── MoE — router gate (all layers) ──
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.{i}.mlp.gate.weight -> model.language_model.layers.{i}.mlp.gate.weight, dtype='bfloat16'"
            for i in range(text_config.num_hidden_layers)
        ]

        # ── MoE — shared experts (all layers) ──
        shared_expert_intermediate_size = getattr(text_config, "shared_expert_intermediate_size", 0)
        if shared_expert_intermediate_size and shared_expert_intermediate_size > 0:
            aoa_config["aoa_statements"] += [
                f"{llm_prefix}layers.{i}.mlp.shared_experts.{x}_proj.weight^T -> model.language_model.layers.{i}.mlp.shared_expert.{x}_proj.weight"
                for i in range(text_config.num_hidden_layers)
                for x in ("gate", "up", "down")
            ]
            aoa_config["aoa_statements"] += [
                f"{llm_prefix}layers.{i}.mlp.shared_experts.gate_weight^T -> model.language_model.layers.{i}.mlp.shared_expert_gate.weight"
                for i in range(text_config.num_hidden_layers)
            ]

        # ── visual model — attention qkv ──
        # Fleet sharded_state_dict uses: model.vision_model.layers.{i} (NOT decoder.layers)
        aoa_config["aoa_statements"] += [
            stmt
            for layer_id in range(vision_config.depth)
            for stmt in (
                f"{visual_prefix}layers.{layer_id}.self_attn.qkv_proj.weight -> model.visual.blocks.{layer_id}.attn.q.weight, model.visual.blocks.{layer_id}.attn.k.weight, model.visual.blocks.{layer_id}.attn.v.weight, fused_qkv, num_heads={vision_config.num_heads}, num_key_value_groups={vision_config.num_heads}",
                f"model.visual.blocks.{layer_id}.attn.q.weight^T, model.visual.blocks.{layer_id}.attn.k.weight^T, model.visual.blocks.{layer_id}.attn.v.weight^T -> model.visual.blocks.{layer_id}.attn.qkv.weight, axis=0",
                f"{visual_prefix}layers.{layer_id}.self_attn.qkv_proj.bias -> model.visual.blocks.{layer_id}.attn.q.bias, model.visual.blocks.{layer_id}.attn.k.bias, model.visual.blocks.{layer_id}.attn.v.bias, fused_qkv, num_heads={vision_config.num_heads}, num_key_value_groups={vision_config.num_heads},axis=0",
                f"model.visual.blocks.{layer_id}.attn.q.bias, model.visual.blocks.{layer_id}.attn.k.bias, model.visual.blocks.{layer_id}.attn.v.bias -> model.visual.blocks.{layer_id}.attn.qkv.bias, axis=0",
            )
        ]
        # visual model — o_proj, mlp, norms, patch_embed, pos_embed, merger
        aoa_config["aoa_statements"] += (
            [
                f"{visual_prefix}layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.visual.blocks.$LAYER_ID.attn.proj.weight",
                f"{visual_prefix}layers.$LAYER_ID.self_attn.o_proj.bias -> model.visual.blocks.$LAYER_ID.attn.proj.bias",
            ]
            + [
                f"{visual_prefix}layers.$LAYER_ID.mlp.{y}.weight^T -> model.visual.blocks.$LAYER_ID.mlp.{x}.weight"
                for x, y in (("linear_fc1", "up_gate_proj"), ("linear_fc2", "down_proj"))
            ]
            + [
                f"{visual_prefix}layers.$LAYER_ID.mlp.{y}.bias -> model.visual.blocks.$LAYER_ID.mlp.{x}.bias"
                for x, y in (("linear_fc1", "up_gate_proj"), ("linear_fc2", "down_proj"))
            ]
        )
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}patch_embed.weight -> model.visual.patch_embed.proj.weight",
            f"{visual_prefix}patch_embed.bias -> model.visual.patch_embed.proj.bias",
            f"{visual_prefix}pos_embed.weight -> model.visual.pos_embed.weight",
            f"{visual_prefix}merger.norm.weight -> model.visual.merger.norm.weight",
            f"{visual_prefix}merger.norm.bias -> model.visual.merger.norm.bias",
            # LayerNorm keys remapped by sharded_state_dict_keys_map
            f"{visual_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.visual.blocks.$LAYER_ID.norm1.weight",
            f"{visual_prefix}layers.$LAYER_ID.input_layernorm.bias -> model.visual.blocks.$LAYER_ID.norm1.bias",
            f"{visual_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.visual.blocks.$LAYER_ID.norm2.weight",
            f"{visual_prefix}layers.$LAYER_ID.post_attention_layernorm.bias -> model.visual.blocks.$LAYER_ID.norm2.bias",
        ]
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}merger.mlp.up_gate_proj.weight^T -> model.visual.merger.linear_fc1.weight",
            f"{visual_prefix}merger.mlp.up_gate_proj.bias -> model.visual.merger.linear_fc1.bias",
            f"{visual_prefix}merger.mlp.down_proj.weight^T -> model.visual.merger.linear_fc2.weight",
            f"{visual_prefix}merger.mlp.down_proj.bias -> model.visual.merger.linear_fc2.bias",
        ]

        # lm_head
        if cls._tied_weights_keys:
            aoa_config["aoa_statements"] += [
                f"{llm_prefix}lm_head.weight -> {'_' if config.tie_word_embeddings else 'lm_head.weight'}",
            ]

        return aoa_config

    def __new__(cls, config, have_criterion=True):
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)

        criterion = None
        if have_criterion:
            criterion = CriterionLayer(config.text_config)

        qwen3_5_model = build_qwen3_5_model(config, criterion)

        qwen3_5_model._gen_aoa_config = cls._gen_aoa_config
        qwen3_5_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        qwen3_5_model._get_tensor_parallel_mappings = cls._get_tensor_parallel_mappings
        qwen3_5_model.get_hardware_flops = types.MethodType(cls.get_hardware_flops, qwen3_5_model)
        qwen3_5_model.config_to_save = config

        return qwen3_5_model


# Alias to match HF config.json architectures: ["Qwen3_5MoeForConditionalGeneration"]
Qwen3_5MoeForConditionalGeneration = Qwen3_5ForConditionalGeneration

__all__ = [
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
]
