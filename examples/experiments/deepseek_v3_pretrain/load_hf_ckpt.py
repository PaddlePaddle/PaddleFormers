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
    "self_attn.input_layernorm.weight": "input_layernorm.weight",
    "mlp.gate.norm_weight": "post_attention_layernorm.weight",
    "mlp.router.weight": "mlp.gate.weight",
    "mlp.router.e_score_correction_bias": "mlp.gate.e_score_correction_bias",
    "mlp.router.norm_weight": "post_attention_layernorm.weight",
    "mlp.shared_experts.norm_weight": "post_attention_layernorm.weight",
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

    m = _LAYER_RE.match(paddle_name)

    if not m:
        return []
    else:
        rest = m.group(2) or ""

    hf_prefix = "model" + ".layers." + m.group(1)

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


def prepare_tensor(tensor, dst_shape, *, force_transpose=False):
    if isinstance(tensor, list):
        t = paddle.cat(
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
    for pd_name, p in model.state_dict().items():
        hf_name = paddle_name_to_hf_names(pd_name)
        if len(hf_name) == 0:
            logger.warning(f"the weight {pd_name} does not need to be loaded")
        elif hf_name[0] in weight_map:
            filename = weight_map[hf_name[0]]
            required_files.add(filename)
            file_to_pd_param_name[filename].append(pd_name)
            pd_param_name_to_file[pd_name].append(filename)
        else:
            logger.warning(f"Warning: {pd_name} -> {hf_name[0]} not found in weight map")

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

                        force_transpose = _is_need_transpose(hf_name[0])

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
                        else:
                            if weight_map[hf_name[0]] == filename:
                                tensor0 = f.get_tensor(hf_name[0])
                                with safe_open(
                                    ckpt_pre + weight_map[hf_name[1]], framework="paddle", device="cpu"
                                ) as f_other:
                                    tensor1 = f_other.get_tensor(hf_name[1])
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
