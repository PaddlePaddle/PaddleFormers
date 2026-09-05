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
