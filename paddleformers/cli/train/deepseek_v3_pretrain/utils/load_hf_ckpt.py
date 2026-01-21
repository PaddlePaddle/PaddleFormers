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
import sys
from collections import defaultdict
from typing import List, Optional

import paddle

from paddleformers.utils.log import logger

try:
    from safetensors import safe_open
except:
    safe_open = None

__all__ = [
    "load_huggingface_ckpt",
]

_LAYER_RE = re.compile(r"^_layers\.(\d+)\.(\d+)(?:\.(.*))?$")
_EXPERT_W1_RE = re.compile(r"^mlp\.experts\.(\d+)\.w1(?:\.weight)?$")
_EXPERT_W2_RE = re.compile(r"^mlp\.experts\.(\d+)\.w2(?:\.weight)?$")
_SHARE_EXPERT_W1_RE = re.compile(r"^mlp\.shared_experts\.w1(?:\.weight)?$")
_SHARE_EXPERT_W2_RE = re.compile(r"^mlp\.shared_experts\.w2(?:\.weight)?$")

_EXPERT_W1_RE_v2 = re.compile(r"^mlp\.experts\.(\d+)\.gate_up_fused_proj(?:\.weight)?$")
_SHARE_EXPERT_W1_RE_v2 = re.compile(r"^mlp\.shared_experts\.gate_up_fused_proj(?:\.weight)?$")
_LAYER_RE_v2 = re.compile(r"_layers.deepseek_v2.layers\.(\d+)\.(.*)$")

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


def paddle_name_to_hf_names_ds_v2(paddle_name: str) -> List[str]:
    """
    Convert Paddle model parameter names to Hugging Face format name lists

    Args:
        paddle_name: Parameter name in Paddle format

    Returns:
        List of parameter names in Hugging Face format (may be split into multiple parameters)
    """
    if paddle_name == "_layers.deepseek_v2.embed_tokens.weight":
        return ["model.embed_tokens.weight"]

    if paddle_name == "_layers.deepseek_v2.norm.weight":
        return ["model.norm.weight"]

    if paddle_name == "_layers.lm_head.weight":
        return ["lm_head.weight"]

    m = _LAYER_RE_v2.match(paddle_name)
    if not m:
        logger.warning("not match here !!", paddle_name)
        return []

    rest = m.group(2) or ""
    layer_id = m.group(1)
    if rest in custom_name_map:
        rest = custom_name_map[rest]
    out_name = "model.layers." + layer_id + "." + rest

    if rest == "mlp.gate_up_fused_proj.weight" or rest == "mlp.w1":
        return [
            "model.layers." + layer_id + ".mlp.gate_proj.weight",
            "model.layers." + layer_id + ".mlp.up_proj.weight",
        ]

    if rest == "mlp.w2":
        return ["model.layers." + layer_id + ".mlp.down_proj.weight"]

    if rest == "mlp.shared_experts.gate_up_fused_proj.weight":
        return [
            "model.layers." + layer_id + ".mlp.shared_experts.gate_proj.weight",
            "model.layers." + layer_id + ".mlp.shared_experts.up_proj.weight",
        ]

    if m := _EXPERT_W1_RE_v2.match(rest):
        expert_id = m.group(1)
        return [
            "model.layers." + layer_id + ".mlp.experts." + expert_id + ".gate_proj.weight",
            "model.layers." + layer_id + ".mlp.experts." + expert_id + ".up_proj.weight",
        ]

    if m := _EXPERT_W1_RE.match(rest):
        expert_id = m.group(1)
        return [
            "model.layers." + layer_id + ".mlp.experts." + expert_id + ".gate_proj.weight",
            "model.layers." + layer_id + ".mlp.experts." + expert_id + ".up_proj.weight",
        ]

    if m := _EXPERT_W2_RE.match(rest):
        expert_id = m.group(1)
        return ["model.layers." + layer_id + ".mlp.experts." + expert_id + ".down_proj.weight"]

    if m := _SHARE_EXPERT_W1_RE.match(rest):
        return [
            "model.layers." + layer_id + ".mlp.shared_experts.gate_proj.weight",
            "model.layers." + layer_id + ".mlp.shared_experts.up_proj.weight",
        ]

    if m := _SHARE_EXPERT_W2_RE.match(rest):
        return ["model.layers." + layer_id + ".mlp.shared_experts.down_proj.weight"]

    return [out_name]


def paddle_name_to_hf_names(paddle_name: str) -> List[str]:
    if paddle_name == "_layers.local_shared_layers.DeepseekV2_shared_weight.embed_tokens.weight":
        return ["model.embed_tokens.weight"]

    if paddle_name == "_layers.deepseek_v2.embed_tokens.weight":
        return ["model.embed_tokens.weight"]

    m = _LAYER_RE.match(paddle_name)

    if not m:
        logger.warning("not match here !!", paddle_name)
        return []
    else:
        rest = m.group(3) or ""

    segment_id = int(m.group(1))
    id_in_segment = int(m.group(2))

    hf_prefix = _get_hf_prefix(segment_id, id_in_segment)

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

    return [f"{hf_prefix}.{rest}"] if rest else [hf_prefix]


def _get_hf_prefix(segment_id: int, id_in_segment: int) -> str:
    special_cases = {(0, 0): "model", (60, 2): "model.layers.61", (60, 3): "model", (60, 4): "lm_head"}

    if (segment_id, id_in_segment) in special_cases:
        return special_cases[(segment_id, id_in_segment)]

    layer_idx = segment_id + id_in_segment - 1
    return f"model.layers.{layer_idx}"


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


def prepare_tensor(tensor, dst_shape, *, force_transpose=False):
    if isinstance(tensor, list):
        t = paddle.concat(
            [
                paddle.transpose(tensor[0], perm=[1, 0]).contiguous(),
                paddle.transpose(tensor[1], perm=[1, 0]).contiguous(),
            ],
            axis=-1,
        )
        if t.shape != dst_shape:
            logger.warning(
                f"Prepare_tensor: shape not match. base tensor shape: {tensor[0].shape}, {tensor[1].shape}, t.shape: {t.shape}, dst_shape: {dst_shape}"
            )
            sys.exit()
        return t

    if force_transpose:
        return tensor.T.contiguous()

    if tensor.shape == dst_shape:
        if len(tensor.shape) != 1:
            logger.warning("attention same shape not transpose !!!")
        return tensor
    if len(tensor.shape) == 2 and paddle.transpose(tensor, perm=[1, 0]).contiguous().shape == dst_shape:
        return paddle.transpose(tensor, perm=[1, 0]).contiguous()

    logger.warning("Prepare_tensor: shape not match.")
    sys.exit()


def load_huggingface_ckpt(model, huggingface_ckpt_path):
    ckpt_pre = huggingface_ckpt_path

    # 1. Load parameter file mapping table
    weight_map_path = ckpt_pre + "/model.safetensors.index.json"
    with open(weight_map_path, "r") as f:
        weight_map = json.load(f)["weight_map"]

    # 2. Create inverse index: file -> parameter list
    file_to_params = defaultdict(list)
    for param_name, filename in weight_map.items():
        file_to_params[filename].append(param_name)

    # 3. Collect file list that model needs
    required_files = set()
    file_to_pd_param_name = defaultdict(list)
    pd_param_name_to_file = defaultdict(list)
    for pd_name, p in model.named_parameters():
        hf_name = paddle_name_to_hf_names(pd_name)
        if hf_name[0] in weight_map:
            filename = weight_map[hf_name[0]]
            required_files.add(filename)
            file_to_pd_param_name[filename].append(pd_name)
            pd_param_name_to_file[pd_name].append(filename)
        else:
            logger.warning(f"Warning: {pd_name} -> {hf_name[0]} not found in weight map")
            import sys

            sys.exit()

        if len(hf_name) > 1:
            if hf_name[1] in weight_map:
                filename = weight_map[hf_name[1]]
                required_files.add(filename)
                file_to_pd_param_name[filename].append(pd_name)
                if filename != pd_param_name_to_file[pd_name][0]:
                    pd_param_name_to_file[pd_name].append(filename)
            else:
                logger.warning(f"Warning: {pd_name} -> {hf_name[1]} not found in weight map")

    # 4. Group file and load
    check_list = []
    logger.info("Start load huggingface ckpt")
    for i, filename in enumerate(required_files):
        try:
            with safe_open(ckpt_pre + filename, framework="paddle", device="cpu") as f:
                # Load all parameters in file
                pd_params = file_to_pd_param_name[filename]
                for pd_param in pd_params:
                    if pd_param in check_list:
                        continue

                    hf_name = paddle_name_to_hf_names(pd_param)
                    if len(hf_name) == 1:
                        tensor = f.get_tensor(hf_name[0])

                        force_transpose = False

                        model.state_dict()[pd_param].set_value(
                            paddle.cast(
                                prepare_tensor(
                                    tensor, model.state_dict()[pd_param].shape, force_transpose=force_transpose
                                ),
                                model.state_dict()[pd_param].dtype,
                            )
                        )
                    else:
                        files = pd_param_name_to_file[pd_param]
                        if len(files) == 1:
                            tensor0 = f.get_tensor(hf_name[0])
                            tensor1 = f.get_tensor(hf_name[1])
                            target_shape = model.state_dict()[pd_param].shape
                            prepared_tensor = prepare_tensor([tensor0, tensor1], target_shape)
                            model.state_dict()[pd_param].set_value(prepared_tensor)
                        else:
                            if weight_map[hf_name[0]] == filename:
                                tensor0 = f.get_tensor(hf_name[0])
                                with safe_open(
                                    ckpt_pre + weight_map[hf_name[1]], framework="paddle", device="cpu"
                                ) as f_other:
                                    tensor1 = f_other.get_tensor(hf_name[1])
                                    target_shape = model.state_dict()[pd_param].shape
                                    prepared_tensor = prepare_tensor([tensor0, tensor1], target_shape)
                                    model.state_dict()[pd_param].set_value(prepared_tensor)
                            else:
                                with safe_open(
                                    ckpt_pre + weight_map[hf_name[0]], framework="paddle", device="cpu"
                                ) as f_other:
                                    tensor0 = f_other.get_tensor(hf_name[0])
                                    tensor1 = f.get_tensor(hf_name[1])
                                    model.state_dict()[pd_param].set_value(
                                        prepare_tensor([tensor0, tensor1], model.state_dict()[pd_param].shape)
                                    )
                    check_list.append(pd_param)

        except Exception as e:
            logger.warning(f"Error loading {filename}: {str(e)}")
            raise
