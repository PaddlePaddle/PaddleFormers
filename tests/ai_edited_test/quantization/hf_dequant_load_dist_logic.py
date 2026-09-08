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
"""Per-rank worker for test_ai_hf_dequant_load_dist.py.

Loads one fp8_block weight into a ``Shard(0)`` target on two cards.  Each rank
holds two of the four logical rows, which is block-aligned, so the transform
plans a ``"local"`` read: the physical qweight/scale slices this rank reads
cover its rows only, and the dequantized output is written without slicing.
"""

import json
import os

import numpy as np
import paddle
import paddle.distributed as dist

from paddleformers.quantization.hf_checkpoint import build_hf_dequant_load_transform

WEIGHT_KEY = "layers.0.attn.wq_a.weight"
SCALE_KEY = "layers.0.attn.wq_a.scale"
SAFETENSORS_FILE = "model-00001-of-00001.safetensors"

LOGICAL_SHAPE = (4, 4)
BLOCK_SHAPE = (2, 2)
# e4m3 codes for 1, 2, 4 and 8: exponent bias 7, mantissa zero.  One code per
# logical row, so a wrong row mapping shows up as a wrong magnitude.
E4M3_ROW_CODES = (0x38, 0x40, 0x48, 0x50)
# ue8m0 codes are biased exponents, so 127 is 2**0 and 128 is 2**1.  The grid is
# asymmetric on purpose: reading the wrong scale block cannot cancel out.
SCALE_GRID = ((127, 128), (128, 127))
UE8M0_BIAS = 127


def descriptor():
    """The quantization rules a model would return from _gen_hf_quan_config()."""
    return {
        "schema_version": 1,
        "component_pairing": {"weight_suffix": ".weight", "scale_suffix": ".scale"},
        "logic_name_suffix": ".weight",
        "groups": [
            {
                "name": "fp8",
                "targets": [r"re:.*\.attn\.wq_a\.weight$"],
                "quant_method": "fp8_block",
                "value_format": "e4m3",
                "scale_format": "ue8m0",
                "block_shape": list(BLOCK_SHAPE),
            }
        ],
    }


def expected_logical_weight():
    """Reference dequantization; every value is a power of two, so it is exact."""
    values = np.empty(LOGICAL_SHAPE, dtype="float32")
    for row in range(LOGICAL_SHAPE[0]):
        value = float(2**row)
        for col in range(LOGICAL_SHAPE[1]):
            code = SCALE_GRID[row // BLOCK_SHAPE[0]][col // BLOCK_SHAPE[1]]
            values[row, col] = value * float(2 ** (code - UE8M0_BIAS))
    return values


def write_checkpoint(path):
    """Write a real-payload HF checkpoint: raw e4m3 codes plus a ue8m0 scale grid."""
    arrays = [
        (WEIGHT_KEY, "F8_E4M3", np.array([[code] * LOGICAL_SHAPE[1] for code in E4M3_ROW_CODES], dtype=np.uint8)),
        (SCALE_KEY, "F8_E8M0", np.array(SCALE_GRID, dtype=np.uint8)),
    ]
    header = {}
    payload = b""
    for key, storage_format, array in arrays:
        raw = array.tobytes()
        header[key] = {
            "dtype": storage_format,
            "shape": list(array.shape),
            "data_offsets": [len(payload), len(payload) + len(raw)],
        }
        payload += raw
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw_header += b" " * ((8 - len(raw_header) % 8) % 8)
    with open(os.path.join(path, SAFETENSORS_FILE), "wb") as file:
        file.write(len(raw_header).to_bytes(8, byteorder="little"))
        file.write(raw_header)
        file.write(payload)


def main():
    dist.init_parallel_env()
    ckpt_path = os.environ["ckpt_path"]
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        write_checkpoint(ckpt_path)
    dist.barrier()

    # Collective: every rank scans its share of the safetensors headers.
    transform = build_hf_dequant_load_transform(ckpt_path, descriptor())

    mesh = dist.ProcessMesh(list(range(world_size)))
    target = dist.shard_tensor(
        paddle.zeros(list(LOGICAL_SHAPE), dtype="bfloat16"),
        mesh,
        [dist.Shard(0)],
    )
    dist.load_state_dict(
        {WEIGHT_KEY: target},
        ckpt_path,
        safetensors=True,
        load_transform=transform,
    )

    rows = LOGICAL_SHAPE[0] // world_size
    start = rank * rows

    # The plan proves the local path ran: a global plan would have read the whole
    # qweight and sliced afterwards.
    plan = transform._read_plans[WEIGHT_KEY]
    assert plan.mode == "local", f"rank {rank} planned a {plan.mode!r} read"
    assert plan.logical_local_shape == (rows, LOGICAL_SHAPE[1]), f"rank {rank} planned for {plan.logical_local_shape}"
    assert plan.logical_global_offset == (start, 0), f"rank {rank} planned at offset {plan.logical_global_offset}"
    qweight_slice = plan.source_slices[WEIGHT_KEY]
    assert tuple(qweight_slice.global_offset) == (start, 0)
    assert tuple(qweight_slice.local_shape) == (rows, LOGICAL_SHAPE[1])
    scale_slice = plan.source_slices[SCALE_KEY]
    assert tuple(scale_slice.global_offset) == (start // BLOCK_SHAPE[0], 0)
    assert tuple(scale_slice.local_shape) == (rows // BLOCK_SHAPE[0], 2)

    expected = expected_logical_weight()[start : start + rows]
    local = target._local_value().astype("float32").numpy()
    assert local.shape == expected.shape, f"rank {rank} got local shape {local.shape}, expected {expected.shape}"
    np.testing.assert_allclose(local, expected)


if __name__ == "__main__":
    main()
