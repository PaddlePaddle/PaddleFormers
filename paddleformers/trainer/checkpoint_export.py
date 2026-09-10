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
"""Oracle-facing safetensors helpers and HF cadence path resolution.

Default layout is unchanged: ``{output_dir}/hf_checkpoint-{step}``.
``save_hf_output_dir`` is an opt-in override so a formal oracle directory
can stay a unique safetensors root without moving cadence for every job.

``mrk checkpoint`` rglob's the formal ``output_dir`` and rejects duplicate
tensor names. Nested cadence copies of the same names are illegal *in that
oracle directory*; they remain valid for ordinary training.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Dict, Iterable, List, Optional, Tuple

HF_CHECKPOINT_PREFIX = "hf_checkpoint"


def hf_export_provenance(config, aoa_config, output_dir, global_step=None):
    """Snapshot the live provider inputs used for this export, not config_to_save.

    This describes export preparation, not successful checkpoint completion.
    Source identity and terminal status belong to the enclosing invocation.
    """
    fields = (
        "num_hidden_layers",
        "num_nextn_predict_layers",
        "mtp_num_layers",
        "n_routed_experts",
        "num_experts",
        "n_shared_experts",
        "multi_latent_attention",
        "index_n_heads",
        "index_head_dim",
        "index_topk",
        "indexer_types",
        "dsa_index_n_heads",
        "dsa_index_head_dim",
        "dsa_index_topk",
        "dsa_indexer_types",
        "dsa_index_share_for_mtp_iteration",
        "mtp_loss_scaling_factor",
        "moe_expert_fusion",
        "using_sonic_moe",
        "first_k_dense_replace",
        "num_attention_heads",
        "num_key_value_heads",
        "hidden_size",
        "moe_intermediate_size",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "sequence_parallel",
        "params_dtype",
        "dtype",
        "use_bias",
        "use_qk_norm",
        "gpt_model_use_experimental_version",
        "moe_routed_expert_use_bias",
    )

    def json_value(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [json_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): json_value(item) for key, item in value.items()}
        return str(value)

    values = {}
    missing = []
    for field in fields:
        if hasattr(config, field):
            values[field] = json_value(getattr(config, field))
        else:
            missing.append(field)
    return {
        "schema": "paddleformers-hf-export/v1",
        "stage": "prepared",
        "provider_class": f"{type(config).__module__}.{type(config).__qualname__}",
        "provider_config": values,
        "missing_provider_fields": missing,
        "aoa_config": json_value(aoa_config),
        "output_dir": os.path.abspath(output_dir),
        "global_step": global_step,
    }


def resolve_hf_checkpoint_dir(
    output_dir: str,
    global_step: int,
    save_hf_output_dir: Optional[str] = None,
    prefix: str = HF_CHECKPOINT_PREFIX,
) -> str:
    """Return the cadence snapshot directory for one step.

    Default (``save_hf_output_dir is None``): nested under ``output_dir``,
    matching historical resume/rotation/latest-discovery consumers.
    Opt-in override: nest under ``save_hf_output_dir`` instead.
    """
    root = save_hf_output_dir if save_hf_output_dir else output_dir
    return os.path.join(root, f"{prefix}-{int(global_step)}")


def _header_tensor_names(path: str) -> List[str]:
    with open(path, "rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"truncated safetensors header: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        raw_header = stream.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError(f"truncated safetensors header: {path}")
    header = json.loads(raw_header)
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not an object: {path}")
    return [name for name in header if name != "__metadata__"]


def iter_safetensors_files(directory: str) -> Iterable[str]:
    for root, _dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".safetensors"):
                yield os.path.join(root, filename)


def collect_safetensors_names(directory: str) -> List[str]:
    names: List[str] = []
    for path in iter_safetensors_files(directory):
        names.extend(_header_tensor_names(path))
    return names


def assert_unique_safetensors_names(directory: str) -> None:
    """Fail closed on empty or duplicate tensor names under ``directory``."""
    seen = set()
    for name in collect_safetensors_names(directory):
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError(f"invalid or duplicate tensor name: {name!r}")
        seen.add(name)


def write_tiny_safetensors(path: str, tensors: Dict[str, Tuple[str, List[int], bytes]]) -> None:
    """Write a tiny safetensors file for focused tests."""
    header = {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw)) % 8)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as stream:
        stream.write(struct.pack("<Q", len(raw)))
        stream.write(raw)
        stream.write(payload)
