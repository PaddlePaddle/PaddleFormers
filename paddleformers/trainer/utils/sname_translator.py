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

"""
Translate old-code structure names (sname) to new-code structure names.

Background
----------
Old code sname convention:
    ernie.embed_tokens.weight
    scale_embeds.weight
    vision_input_projection.*
    vision_patch_merger.*
    audio_spk_emb_adaptor.*
    ernie.layers.{0~27}.*
    ernie.layers.{8~15}.audio_embeds.weight   (per-layer audio embedding)
    ernie.norm.weight
    lm_head.weight / lm_head.bias

New code sname convention (non-pipeline, pipeline_model_parallel_size=1):
    ernie.embed.embed_tokens.weight            (added ernie.embed. prefix)
    ernie.embed.scale_embeds.weight
    ernie.embed.vision_input_projection.*
    ernie.embed.vision_patch_merger.*
    ernie.embed.audio_spk_emb_adaptor.*
    ernie.layers.{0~27}.*                      (unchanged)
    ernie.audio_embeds.audio_embeds_{0~7}      (global shared, no longer per-layer)
    ernie.norm.weight                          (unchanged)
    lm_head.weight / lm_head.bias             (unchanged)

Key differences
---------------
1. Top-level embed/vision/audio params: moved under ernie.embed.* namespace
2. audio_embeds: from ernie.layers.{8+i}.audio_embeds.weight
                 to   ernie.audio_embeds.audio_embeds_{i}
3. All decoder layer params (ernie.layers.X.*): unchanged
"""

# Old audio layer ids and their corresponding new audio_embeds index
_OLD_AUDIO_LAYER_TO_IDX = {8: 0, 9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6, 15: 7}

# Decoder layer parameter suffixes present in every decoder layer
_DECODER_PARAM_SUFFIXES = [
    "self_attn.qkv_proj.weight",
    "self_attn.qkv_proj.bias",
    "self_attn.o_proj.weight",
    "self_attn.o_proj.bias",
    "mlp.up_gate_proj.weight",
    "mlp.up_gate_proj.bias",
    "mlp.down_proj.weight",
    "mlp.down_proj.bias",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
]

# Audio-head suffixes (only in old layers 8-15)
_AUDIO_DECODER_PARAM_SUFFIXES = [
    "audio_norm.weight",
    "audio_head.weight",
    "audio_head.bias",
]

# Vision-head suffixes (only in old layer 15)
_VISION_DECODER_PARAM_SUFFIXES = [
    "vision_norm.weight",
    "vision_head.weight",
    "vision_head.bias",
]


def build_old_to_new_sname_mapping():
    """
    Return a complete {old_sname: new_sname} dict covering all 334 parameters.
    """
    mapping = {}

    # ------------------------------------------------------------------
    # 1. Top-level params -> ernie.embed.* namespace
    # ------------------------------------------------------------------
    mapping["ernie.embed_tokens.weight"] = "ernie.embed.embed_tokens.weight"
    mapping["scale_embeds.weight"] = "ernie.embed.scale_embeds.weight"

    mapping["vision_input_projection.weight"] = "ernie.embed.vision_input_projection.weight"
    mapping["vision_input_projection.bias"] = "ernie.embed.vision_input_projection.bias"

    for suffix in [
        "norm.weight",
        "vison_fc.weight",
        "vison_fc.bias",
        "out_fc.weight",
        "out_fc.bias",
        "attn.self_attn.qkv_proj.weight",
        "attn.self_attn.qkv_proj.bias",
        "attn.self_attn.o_proj.weight",
        "attn.self_attn.o_proj.bias",
        "attn.input_layernorm.weight",
    ]:
        mapping[f"vision_patch_merger.{suffix}"] = f"ernie.embed.vision_patch_merger.{suffix}"

    mapping["audio_spk_emb_adaptor.weight"] = "ernie.embed.audio_spk_emb_adaptor.weight"
    mapping["audio_spk_emb_adaptor.bias"] = "ernie.embed.audio_spk_emb_adaptor.bias"

    # ------------------------------------------------------------------
    # 2. audio_embeds: per-layer -> global shared
    #    old: ernie.layers.{8+i}.audio_embeds.weight
    #    new: ernie.audio_embeds.audio_embeds_{i}
    # ------------------------------------------------------------------
    for old_layer_id, audio_idx in _OLD_AUDIO_LAYER_TO_IDX.items():
        old_key = f"ernie.layers.{old_layer_id}.audio_embeds.weight"
        new_key = f"ernie.audio_embeds.audio_embeds_{audio_idx}"
        mapping[old_key] = new_key

    # ------------------------------------------------------------------
    # 3. Decoder layer params: unchanged (ernie.layers.X.* -> ernie.layers.X.*)
    # ------------------------------------------------------------------
    for layer_id in range(28):
        old_prefix = f"ernie.layers.{layer_id}"
        new_prefix = f"ernie.layers.{layer_id}"

        for suffix in _DECODER_PARAM_SUFFIXES:
            mapping[f"{old_prefix}.{suffix}"] = f"{new_prefix}.{suffix}"

        if layer_id in _OLD_AUDIO_LAYER_TO_IDX:
            for suffix in _AUDIO_DECODER_PARAM_SUFFIXES:
                mapping[f"{old_prefix}.{suffix}"] = f"{new_prefix}.{suffix}"

        if layer_id == 15:
            for suffix in _VISION_DECODER_PARAM_SUFFIXES:
                mapping[f"{old_prefix}.{suffix}"] = f"{new_prefix}.{suffix}"

    # ------------------------------------------------------------------
    # 4. Tail params: unchanged
    # ------------------------------------------------------------------
    mapping["ernie.norm.weight"] = "ernie.norm.weight"
    mapping["lm_head.weight"] = "lm_head.weight"
    mapping["lm_head.bias"] = "lm_head.bias"

    return mapping


def translate_structure_name_mapping(old_structure_name_mapping):
    """
    Replace the keys (old snames) in a structure_name_mapping dict with
    the corresponding new snames.

    Parameters
    ----------
    old_structure_name_mapping : dict[str, str]
        {old_sname: tname} as loaded from the checkpoint's sharding meta.

    Returns
    -------
    dict[str, str]
        {new_sname: tname}

    Raises
    ------
    KeyError
        If an old sname has no entry in the translation table.
    """
    sname_map = build_old_to_new_sname_mapping()
    result = {}
    for old_sname, tname in old_structure_name_mapping.items():
        if old_sname not in sname_map:
            raise KeyError(
                f"[sname_translator] No new-code mapping found for old sname: '{old_sname}'. "
                f"Please update build_old_to_new_sname_mapping()."
            )
        result[sname_map[old_sname]] = tname
    return result
