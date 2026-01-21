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


import json
import re
import shutil
from pathlib import Path
from typing import List, Optional

from safetensors.torch import load_file, save_file

_LAYER_RE = re.compile(r"^deepseek_v2.layers\.(\d+)\.(.*)$")
_EXPERT_W1_RE = re.compile(r"^mlp\.experts\.(\d+)\.w1(?:\.weight)?$")
_EXPERT_W2_RE = re.compile(r"^mlp\.experts\.(\d+)\.w2(?:\.weight)?$")
_SHARE_EXPERT_W1_RE = re.compile(r"^mlp\.shared_experts\.w1(?:\.weight)?$")
_SHARE_EXPERT_W2_RE = re.compile(r"^mlp\.shared_experts\.w2(?:\.weight)?$")

_EXPERT_W1_RE_v2 = re.compile(r"^mlp\.experts\.(\d+)\.gate_up_fused_proj(?:\.weight)?$")
_SHARE_EXPERT_W1_RE_v2 = re.compile(r"^mlp\.shared_experts\.gate_up_fused_proj(?:\.weight)?$")

custom_name_map = {
    "self_attn.input_layernorm.weight": "input_layernorm.weight",
    "self_attn.fused_rms_norm_linear.rms_norm_weight": "input_layernorm.weight",
    "self_attn.memory_recompute_att.kv_ln_weight": "self_attn.kv_a_layernorm.weight",
    "self_attn.fused_rms_norm_linear.kv_down_weight": "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.memory_recompute_att.kv_up_weight": "self_attn.kv_b_proj.weight",
    "self_attn.memory_recompute_att.q_ln_weight": "self_attn.q_a_layernorm.weight",
    "self_attn.fused_rms_norm_linear.q_down_weight": "self_attn.q_a_proj.weight",
    "self_attn.memory_recompute_att.q_up_weight": "self_attn.q_b_proj.weight",
}


def paddle_name_to_hf_names(paddle_name: str) -> List[str]:
    """
    Convert Paddle model parameter names to Hugging Face format name lists

    Args:
        paddle_name: Parameter name in Paddle format

    Returns:
        List of parameter names in Hugging Face format (may be split into multiple parameters)
    """

    if paddle_name == "deepseek_v2.embed_tokens.weight":
        return ["model.embed_tokens.weight"]

    if paddle_name == "deepseek_v2.norm.weight":
        return ["model.norm.weight"]

    if paddle_name == "lm_head.weight":
        return ["lm_head.weight"]

    if ".norm_weight" in paddle_name:
        return []

    if "router" in paddle_name:
        return []

    if "input_layernorm" in paddle_name:
        return []

    m = _LAYER_RE.match(paddle_name)

    if not m:
        return []
    else:
        rest = m.group(2) or ""

    hf_prefix = "model" + ".layers." + m.group(1)

    # if "self_attn.fused_rms_norm_linear.rms_norm_weight" in paddle_name:
    #     breakpoint()

    if rest in custom_name_map:
        return [f"{hf_prefix}.{custom_name_map[rest]}"]

    if expert_names := _handle_expert_weights(hf_prefix, rest):
        return expert_names

    if shared_mlp_names := _handle_shared_expert_weights(hf_prefix, rest):
        return shared_mlp_names

    if mlp_names := _handle_mlp_weights(hf_prefix, rest):
        return mlp_names

    if rest == "mlp.gate_up_fused_proj.weight" or rest == "mlp.w1":
        return [hf_prefix + ".mlp.gate_proj.weight", hf_prefix + ".mlp.up_proj.weight"]

    if rest == "mlp.w2":
        return [hf_prefix + ".mlp.down_proj.weight"]

    if rest == "mlp.shared_experts.gate_up_fused_proj.weight":
        return [hf_prefix + ".mlp.shared_experts.gate_proj.weight", hf_prefix + ".mlp.shared_experts.up_proj.weight"]

    if m := _EXPERT_W1_RE_v2.match(rest):
        expert_id = m.group(1)
        return [
            hf_prefix + ".mlp.experts." + expert_id + ".gate_proj.weight",
            hf_prefix + ".mlp.experts." + expert_id + ".up_proj.weight",
        ]

    if m := _EXPERT_W1_RE.match(rest):
        expert_id = m.group(1)
        return [
            hf_prefix + ".mlp.experts." + expert_id + ".gate_proj.weight",
            hf_prefix + ".mlp.experts." + expert_id + ".up_proj.weight",
        ]

    if m := _EXPERT_W2_RE.match(rest):
        expert_id = m.group(1)
        return [hf_prefix + ".mlp.experts." + expert_id + ".down_proj.weight"]

    if m := _SHARE_EXPERT_W1_RE.match(rest):
        return [hf_prefix + ".mlp.shared_experts.gate_proj.weight", hf_prefix + ".mlp.shared_experts.up_proj.weight"]

    if m := _SHARE_EXPERT_W2_RE.match(rest):
        return [hf_prefix + ".mlp.shared_experts.down_proj.weight"]

    return [paddle_name.replace("deepseek_v2", "model")]


def _handle_expert_weights(hf_prefix: str, rest: str) -> Optional[List[str]]:
    if m := _EXPERT_W1_RE.match(rest):
        expert_id = int(m.group(1))
        return [
            f"{hf_prefix}.mlp.experts.{expert_id}.gate_proj.weight",
            f"{hf_prefix}.mlp.experts.{expert_id}.up_proj.weight",
        ]

    if m := _EXPERT_W2_RE.match(rest):
        expert_id = int(m.group(1))
        return [f"{hf_prefix}.mlp.experts.{expert_id}.down_proj.weight"]

    return None


def _handle_shared_expert_weights(hf_prefix: str, rest: str) -> Optional[List[str]]:
    if _SHARE_EXPERT_W1_RE.match(rest):
        return [
            f"{hf_prefix}.mlp.shared_experts.gate_proj.weight",
            f"{hf_prefix}.mlp.shared_experts.up_proj.weight",
        ]

    if _SHARE_EXPERT_W2_RE.match(rest):
        return [f"{hf_prefix}.mlp.shared_experts.down_proj.weight"]

    return None


def _handle_mlp_weights(hf_prefix: str, rest: str) -> Optional[List[str]]:
    if rest == "mlp.w1":
        return [f"{hf_prefix}.mlp.gate_proj.weight", f"{hf_prefix}.mlp.up_proj.weight"]

    if rest == "mlp.w2":
        return [f"{hf_prefix}.mlp.down_proj.weight"]

    return None


def _is_need_transpose(key):
    transpose_weight_keys = [
        "fused_rms_norm_linear.kv_down_weight",
        "memory_recompute_att.kv_up_weight",
        "o_proj.weight",
        "fused_rms_norm_linear.q_down_weight",
        "memory_recompute_att.q_up_weight",
        "w1",
        "w2",
        "gate.weight",
        "eh_proj.weight",
        "lm_head.weight",
    ]
    for trans_key in transpose_weight_keys:
        if key.endswith(trans_key):
            return True
    return False


def prepare_tensor(key, value):

    value_size = 0

    new_keys = paddle_name_to_hf_names(key)

    if _is_need_transpose(key):
        value = value.T.contiguous()

    if len(new_keys) == 2:
        new_keys_1 = new_keys[0]
        new_keys_2 = new_keys[1]
        chunks = value.split(value.shape[0] // 2, dim=0)
        value_1, value_2 = chunks[0].contiguous(), chunks[1].contiguous()
        value_size += value_1.numel() * value_1.element_size() + value_2.numel() * value_2.element_size()
        return {new_keys_1: value_1, new_keys_2: value_2}, value_size

    elif len(new_keys) == 1:
        value_size += value.numel() * value.element_size()
        return {new_keys[0]: value}, value_size

    elif len(new_keys) == 0:
        return {}, 0

    raise ValueError("new_key length is not 1 or 2")


def load_pretrained_ckpt(ckpt_path, output_path):
    ckpt_pre = ckpt_path
    shard_dir = Path(ckpt_path)

    # 1. Load parameter file mapping table
    weight_map_path = ckpt_pre + "/model.safetensors.index.json"
    with open(weight_map_path, "r") as f:
        weight_map = json.load(f)

    new_map = {}
    for old_key, shard in weight_map["weight_map"].items():
        new_key = paddle_name_to_hf_names(old_key)
        if len(new_key) == 1:
            new_key = new_key[0]
            new_map[new_key] = shard
            print(f"{old_key} -> {new_key}")
        elif len(new_key) == 2:
            new_key1 = new_key[0]
            new_key2 = new_key[1]
            new_map[new_key1] = shard
            new_map[new_key2] = shard
            print(f"{old_key} -> {new_key1}, {new_key2}")
        elif len(new_key) == 0:
            print(f"the weight {old_key} need to be removed")
            print(f"{old_key} -> {new_key}")
        else:
            print(f"{old_key} -> {new_key}")
            raise ValueError("new_key length is not 1 or 2")

    shard_files = [
        p
        for p in shard_dir.glob("*.safetensors")
        if not p.name.startswith("optimizer.") and not p.name.startswith("master_weights.")
    ]

    total_size = 0
    for shard_file in shard_files:
        new_state = {}
        print(f"Loading shard: {shard_file}")
        state = load_file(shard_file)  # bf16 保持
        for k, v in state.items():
            st, st_size = prepare_tensor(k, v)
            new_state.update(st)
            if "master" not in shard_file.name:
                total_size += st_size
        tmp_file = output_path + "/" + shard_file.name
        print("tmp_file:", tmp_file, flush=True)
        save_file(new_state, tmp_file)

    weight_map["metadata"]["total_size"] = total_size
    weight_map["weight_map"] = new_map
    weight_map_path_new = Path(output_path + "/model.safetensors.index.json")
    weight_map_path_new.write_text(json.dumps(weight_map, indent=2), encoding="utf-8")

    json_files = [p for p in shard_dir.glob("*.json") if not p.name.startswith("model.safetensors.index.")]
    for f in json_files:
        print("copy config file:", f, flush=True)
        shutil.copy(str(f), output_path)


if __name__ == "__main__":
    ckpt_path = "your ckpt path"
    output_path = "output path"
    load_pretrained_ckpt(ckpt_path, output_path)
