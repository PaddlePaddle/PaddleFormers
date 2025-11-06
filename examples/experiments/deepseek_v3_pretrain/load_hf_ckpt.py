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

"""
Model checkpoint conversion utilities for PaddlePaddle to HuggingFace format.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import paddle

from paddleformers.utils.log import logger

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None
    logger.warning("safetensors not installed. Some functionality may be limited.")

# Regular expressions for pattern matching
_LAYER_RE = re.compile(r"^_layers\.(\d+)\.(\d+)(?:\.(.*))?$")
_LAYER_RE_V2 = re.compile(r"_layers\.deepseek_v2\.layers\.(\d+)\.(.*)$")

# Expert weight patterns
_EXPERT_W1_RE = re.compile(r"^mlp\.experts\.(\d+)\.w1(?:\.weight)?$")
_EXPERT_W2_RE = re.compile(r"^mlp\.experts\.(\d+)\.w2(?:\.weight)?$")
_EXPERT_W1_RE_V2 = re.compile(r"^mlp\.experts\.(\d+)\.gate_up_fused_proj(?:\.weight)?$")

# Shared expert patterns
_SHARE_EXPERT_W1_RE = re.compile(r"^mlp\.shared_experts\.w1(?:\.weight)?$")
_SHARE_EXPERT_W2_RE = re.compile(r"^mlp\.shared_experts\.w2(?:\.weight)?$")
_SHARE_EXPERT_W1_RE_V2 = re.compile(r"^mlp\.shared_experts\.gate_up_fused_proj(?:\.weight)?$")

# Custom name mappings
CUSTOM_NAME_MAP = {
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
    Convert Paddle model parameter names to HuggingFace format for DeepSeek V2.

    Args:
        paddle_name: Parameter name in Paddle format

    Returns:
        List of parameter names in HuggingFace format (may be split into multiple parameters)
    """
    # Handle special cases
    special_mappings = {
        "_layers.deepseek_v2.embed_tokens.weight": ["model.embed_tokens.weight"],
        "_layers.deepseek_v2.norm.weight": ["model.norm.weight"],
        "_layers.lm_head.weight": ["lm_head.weight"],
    }
    
    if paddle_name in special_mappings:
        return special_mappings[paddle_name]

    # Match layer pattern
    match = _LAYER_RE_V2.match(paddle_name)
    if not match:
        logger.warning(f"Pattern not matched for: {paddle_name}")
        return []

    layer_id = match.group(1)
    rest = match.group(2) or ""
    
    # Apply custom name mapping if exists
    if rest in CUSTOM_NAME_MAP:
        rest = CUSTOM_NAME_MAP[rest]
    
    base_prefix = f"model.layers.{layer_id}"
    
    # Handle various weight patterns
    if rest in ("mlp.gate_up_fused_proj.weight", "mlp.w1"):
        return [
            f"{base_prefix}.mlp.gate_proj.weight",
            f"{base_prefix}.mlp.up_proj.weight",
        ]
    
    if rest == "mlp.w2":
        return [f"{base_prefix}.mlp.down_proj.weight"]
    
    if rest == "mlp.shared_experts.gate_up_fused_proj.weight":
        return [
            f"{base_prefix}.mlp.shared_experts.gate_proj.weight",
            f"{base_prefix}.mlp.shared_experts.up_proj.weight",
        ]
    
    # Handle expert weights
    for pattern, proj_types in [
        (_EXPERT_W1_RE_V2, ["gate_proj", "up_proj"]),
        (_EXPERT_W1_RE, ["gate_proj", "up_proj"]),
        (_EXPERT_W2_RE, ["down_proj"]),
    ]:
        if expert_match := pattern.match(rest):
            expert_id = expert_match.group(1)
            return [
                f"{base_prefix}.mlp.experts.{expert_id}.{proj_type}.weight"
                for proj_type in proj_types
            ]
    
    # Handle shared expert weights
    if _SHARE_EXPERT_W1_RE.match(rest):
        return [
            f"{base_prefix}.mlp.shared_experts.gate_proj.weight",
            f"{base_prefix}.mlp.shared_experts.up_proj.weight",
        ]
    
    if _SHARE_EXPERT_W2_RE.match(rest):
        return [f"{base_prefix}.mlp.shared_experts.down_proj.weight"]
    
    return [f"{base_prefix}.{rest}"]


def paddle_name_to_hf_names(paddle_name: str) -> List[str]:
    """
    Convert Paddle model parameter names to HuggingFace format.

    Args:
        paddle_name: Parameter name in Paddle format

    Returns:
        List of parameter names in HuggingFace format (may be split into multiple parameters)
    """
    # Handle special embeddings
    special_embeddings = {
        "_layers.local_shared_layers.DeepseekV2_shared_weight.embed_tokens.weight",
        "_layers.deepseek_v2.embed_tokens.weight",
    }
    if paddle_name in special_embeddings:
        return ["model.embed_tokens.weight"]

    # Match layer pattern
    match = _LAYER_RE.match(paddle_name)
    if not match:
        logger.warning(f"Pattern not matched for: {paddle_name}")
        return []

    segment_id = int(match.group(1))
    id_in_segment = int(match.group(2))
    rest = match.group(3) or ""

    hf_prefix = _get_hf_prefix(segment_id, id_in_segment)

    # Apply custom name mapping
    if rest in CUSTOM_NAME_MAP:
        return [f"{hf_prefix}.{CUSTOM_NAME_MAP[rest]}"]

    # Try different handlers in order
    handlers = [
        _handle_expert_weights,
        _handle_shared_expert_weights,
        _handle_mlp_weights,
    ]
    
    for handler in handlers:
        if result := handler(hf_prefix, rest):
            return result

    # Handle remaining patterns
    if rest in ("mlp.gate_up_fused_proj.weight", "mlp.w1"):
        return [f"{hf_prefix}.mlp.gate_proj.weight", f"{hf_prefix}.mlp.up_proj.weight"]
    
    if rest == "mlp.w2":
        return [f"{hf_prefix}.mlp.down_proj.weight"]
    
    if rest == "mlp.shared_experts.gate_up_fused_proj.weight":
        return [
            f"{hf_prefix}.mlp.shared_experts.gate_proj.weight",
            f"{hf_prefix}.mlp.shared_experts.up_proj.weight",
        ]

    # Handle expert patterns with V2 regex
    for pattern, proj_types in [
        (_EXPERT_W1_RE_V2, ["gate_proj", "up_proj"]),
        (_EXPERT_W1_RE, ["gate_proj", "up_proj"]),
        (_EXPERT_W2_RE, ["down_proj"]),
    ]:
        if expert_match := pattern.match(rest):
            expert_id = expert_match.group(1)
            return [
                f"{hf_prefix}.mlp.experts.{expert_id}.{proj_type}.weight"
                for proj_type in proj_types
            ]

    # Handle shared expert patterns
    if _SHARE_EXPERT_W1_RE.match(rest):
        return [
            f"{hf_prefix}.mlp.shared_experts.gate_proj.weight",
            f"{hf_prefix}.mlp.shared_experts.up_proj.weight",
        ]
    
    if _SHARE_EXPERT_W2_RE.match(rest):
        return [f"{hf_prefix}.mlp.shared_experts.down_proj.weight"]

    return [f"{hf_prefix}.{rest}"] if rest else [hf_prefix]


def _get_hf_prefix(segment_id: int, id_in_segment: int) -> str:
    """Generate HuggingFace format layer prefix."""
    # Special layer mappings
    special_cases = {
        (0, 0): "model",
        (60, 2): "model.layers.61",
        (60, 3): "model",
        (60, 4): "lm_head",
    }

    if (segment_id, id_in_segment) in special_cases:
        return special_cases[(segment_id, id_in_segment)]

    # General layer calculation
    layer_idx = segment_id + id_in_segment - 1
    return f"model.layers.{layer_idx}"


def _handle_expert_weights(hf_prefix: str, rest: str) -> Optional[List[str]]:
    """Handle expert weight patterns."""
    if match := _EXPERT_W1_RE.match(rest):
        expert_id = match.group(1)
        return [
            f"{hf_prefix}.mlp.experts.{expert_id}.gate_proj.weight",
            f"{hf_prefix}.mlp.experts.{expert_id}.up_proj.weight",
        ]

    if match := _EXPERT_W2_RE.match(rest):
        expert_id = match.group(1)
        return [f"{hf_prefix}.mlp.experts.{expert_id}.down_proj.weight"]

    return None


def _handle_shared_expert_weights(hf_prefix: str, rest: str) -> Optional[List[str]]:
    """Handle shared expert weight patterns."""
    if _SHARE_EXPERT_W1_RE.match(rest):
        return [
            f"{hf_prefix}.mlp.shared_experts.gate_proj.weight",
            f"{hf_prefix}.mlp.shared_experts.up_proj.weight",
        ]

    if _SHARE_EXPERT_W2_RE.match(rest):
        return [f"{hf_prefix}.mlp.shared_experts.down_proj.weight"]

    return None


def _handle_mlp_weights(hf_prefix: str, rest: str) -> Optional[List[str]]:
    """Handle MLP weight patterns."""
    mlp_mappings = {
        "mlp.w1": ["mlp.gate_proj.weight", "mlp.up_proj.weight"],
        "mlp.w2": ["mlp.down_proj.weight"],
    }
    
    if rest in mlp_mappings:
        return [f"{hf_prefix}.{name}" for name in mlp_mappings[rest]]
    
    return None


def prepare_tensor(
    tensor: paddle.Tensor | List[paddle.Tensor],
    dst_shape: Tuple[int, ...],
    *,
    force_transpose: bool = False
) -> paddle.Tensor:
    """
    Prepare tensor for model loading with proper shape and layout.
    
    Args:
        tensor: Input tensor or list of tensors
        dst_shape: Destination shape
        force_transpose: Force transpose operation
        
    Returns:
        Prepared tensor with correct shape
        
    Raises:
        SystemExit: If shape mismatch cannot be resolved
    """
    if isinstance(tensor, list):
        # Concatenate transposed tensors
        concatenated = paddle.concat(
            [t.T.contiguous() for t in tensor],
            axis=-1,
        )
        if concatenated.shape != dst_shape:
            logger.error(
                f"Shape mismatch after concatenation. "
                f"Source shapes: {[t.shape for t in tensor]}, "
                f"Result shape: {concatenated.shape}, "
                f"Expected shape: {dst_shape}"
            )
            sys.exit(1)
        return concatenated

    if force_transpose:
        return tensor.T.contiguous()

    # Check if shapes match
    if tensor.shape == dst_shape:
        if len(tensor.shape) != 1:
            logger.warning("Tensor shapes match without transpose (non-1D tensor)")
        return tensor
    
    # Try transposing for 2D tensors
    if len(tensor.shape) == 2 and tensor.T.shape == dst_shape:
        return tensor.T.contiguous()

    logger.error(f"Cannot match tensor shape {tensor.shape} to destination shape {dst_shape}")
    sys.exit(1)


def load_huggingface_checkpoint(
    model: paddle.nn.Layer,
    checkpoint_path: str
) -> None:
    """
    Load HuggingFace checkpoint into PaddlePaddle model.
    
    Args:
        model: PaddlePaddle model
        checkpoint_path: Path to HuggingFace checkpoint directory
        
    Raises:
        RuntimeError: If safetensors is not available
        FileNotFoundError: If checkpoint files are missing
        ValueError: If weight mapping fails
    """
    if safe_open is None:
        raise RuntimeError("safetensors is required but not installed")
    
    checkpoint_path = Path(checkpoint_path)
    
    # Load weight mapping
    weight_map_path = checkpoint_path / "model.safetensors.index.json"
    if not weight_map_path.exists():
        raise FileNotFoundError(f"Weight map not found: {weight_map_path}")
    
    with open(weight_map_path, "r") as f:
        weight_map = json.load(f)["weight_map"]

    # Build file to parameters mapping
    file_to_params = defaultdict(list)
    for param_name, filename in weight_map.items():
        file_to_params[filename].append(param_name)

    # Collect required files and mappings
    required_files = set()
    file_to_pd_params = defaultdict(list)
    pd_param_to_files = defaultdict(list)
    
    for pd_name, _ in model.named_parameters():
        hf_names = paddle_name_to_hf_names(pd_name)
        
        for hf_name in hf_names:
            if hf_name not in weight_map:
                logger.error(f"Missing weight mapping: {pd_name} -> {hf_name}")
                sys.exit(1)
            
            filename = weight_map[hf_name]
            required_files.add(filename)
            file_to_pd_params[filename].append(pd_name)
            
            if filename not in pd_param_to_files[pd_name]:
                pd_param_to_files[pd_name].append(filename)

    # Load weights
    loaded_params = set()
    logger.info(f"Loading HuggingFace checkpoint from {checkpoint_path}")
    logger.info(f"Total files to load: {len(required_files)}")
    
    for i, filename in enumerate(required_files, 1):
        logger.info(f"Loading file {i}/{len(required_files)}: {filename}")
        
        file_path = checkpoint_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {file_path}")
        
        try:
            with safe_open(str(file_path), framework="paddle", device="cpu") as f:
                _load_file_weights(
                    f, model, file_to_pd_params[filename],
                    loaded_params, pd_param_to_files, weight_map,
                    checkpoint_path
                )
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            raise

    logger.info("Checkpoint loading completed successfully")


def _load_file_weights(
    file_handle,
    model: paddle.nn.Layer,
    pd_params: List[str],
    loaded_params: Set[str],
    pd_param_to_files: Dict[str, List[str]],
    weight_map: Dict[str, str],
    checkpoint_path: Path
) -> None:
    """Helper function to load weights from a single file."""
    for pd_param in pd_params:
        if pd_param in loaded_params:
            continue
        
        hf_names = paddle_name_to_hf_names(pd_param)
        model_param = model.state_dict()[pd_param]
        
        if len(hf_names) == 1:
            # Single tensor case
            tensor = file_handle.get_tensor(hf_names[0])
            prepared_tensor = prepare_tensor(tensor, model_param.shape)
            model_param.set_value(
                paddle.cast(prepared_tensor, model_param.dtype)
            )
        else:
            # Multiple tensors case
            tensors = _load_multiple_tensors(
                file_handle, hf_names, pd_param,
                pd_param_to_files, weight_map, checkpoint_path
            )
            prepared_tensor = prepare_tensor(tensors, model_param.shape)
            model_param.set_value(prepared_tensor)
        
        loaded_params.add(pd_param)


def _load_multiple_tensors(
    file_handle,
    hf_names: List[str],
    pd_param: str,
    pd_param_to_files: Dict[str, List[str]],
    weight_map: Dict[str, str],
    checkpoint_path: Path
) -> List[paddle.Tensor]:
    """Load multiple tensors that may be in different files."""
    files = pd_param_to_files[pd_param]
    
    if len(files) == 1:
        # All tensors in the same file
        return [file_handle.get_tensor(name) for name in hf_names]
    
    # Tensors in different files
    tensors = []
    current_filename = file_handle._filename.split("/")[-1]
    
    for hf_name in hf_names:
        if weight_map[hf_name] == current_filename:
            tensors.append(file_handle.get_tensor(hf_name))
        else:
            # Load from another file
            other_file_path = checkpoint_path / weight_map[hf_name]
            with safe_open(str(other_file_path), framework="paddle", device="cpu") as other_file:
                tensors.append(other_file.get_tensor(hf_name))
    
    return tensors


# Maintain backward compatibility
load_huggingface_ckpt = load_huggingface_checkpoint