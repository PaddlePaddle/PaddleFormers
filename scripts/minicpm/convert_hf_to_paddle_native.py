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

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-npy")
    export_parser.add_argument("--hf-dir", required=True)
    export_parser.add_argument("--output-dir", required=True)

    import_parser = subparsers.add_parser("import-paddle")
    import_parser.add_argument("--npy-dir", required=True)
    import_parser.add_argument("--hf-dir", required=True)
    import_parser.add_argument("--output-dir", required=True)
    import_parser.add_argument("--dtype", default="float32", choices=["float32"])
    return parser.parse_args()


def save_array(output_dir, manifest, name, array):
    path = f"{name}.npy"
    np.save(output_dir / path, array.astype("float32", copy=False))
    manifest[name] = path


def fuse_qkv(q_weight, k_weight, v_weight, num_heads, num_key_value_heads):
    q = q_weight.float().cpu().numpy().T
    k = k_weight.float().cpu().numpy().T
    v = v_weight.float().cpu().numpy().T
    hidden_size = q.shape[0]
    head_dim = q.shape[1] // num_heads
    num_key_value_groups = num_heads // num_key_value_heads
    q = q.reshape(hidden_size, num_key_value_heads, num_key_value_groups, head_dim)
    k = k.reshape(hidden_size, num_key_value_heads, 1, head_dim)
    v = v.reshape(hidden_size, num_key_value_heads, 1, head_dim)
    return np.concatenate([q, k, v], axis=2).reshape(
        hidden_size, num_key_value_heads * (num_key_value_groups + 2) * head_dim
    )


def export_npy(args):
    import torch

    hf_dir = Path(args.hf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((hf_dir / "config.json").read_text())
    num_layers = config["num_hidden_layers"]
    num_heads = config["num_attention_heads"]
    num_key_value_heads = config["num_key_value_heads"]

    state = torch.load(hf_dir / "pytorch_model.bin", map_location="cpu")
    manifest = {}

    save_array(
        output_dir, manifest, "model.embed_tokens.weight", state["model.embed_tokens.weight"].float().cpu().numpy()
    )
    save_array(output_dir, manifest, "model.norm.weight", state["model.norm.weight"].float().cpu().numpy())
    save_array(output_dir, manifest, "lm_head.weight", state["model.embed_tokens.weight"].float().cpu().numpy())

    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"
        save_array(
            output_dir,
            manifest,
            f"{prefix}.input_layernorm.weight",
            state[f"{prefix}.input_layernorm.weight"].float().cpu().numpy(),
        )
        save_array(
            output_dir,
            manifest,
            f"{prefix}.post_attention_layernorm.weight",
            state[f"{prefix}.post_attention_layernorm.weight"].float().cpu().numpy(),
        )
        save_array(
            output_dir,
            manifest,
            f"{prefix}.self_attn.o_proj.weight",
            state[f"{prefix}.self_attn.o_proj.weight"].float().cpu().numpy().T,
        )
        save_array(
            output_dir,
            manifest,
            f"{prefix}.mlp.down_proj.weight",
            state[f"{prefix}.mlp.down_proj.weight"].float().cpu().numpy().T,
        )
        save_array(
            output_dir,
            manifest,
            f"{prefix}.self_attn.qkv_proj.weight",
            fuse_qkv(
                state[f"{prefix}.self_attn.q_proj.weight"],
                state[f"{prefix}.self_attn.k_proj.weight"],
                state[f"{prefix}.self_attn.v_proj.weight"],
                num_heads,
                num_key_value_heads,
            ),
        )
        gate = state[f"{prefix}.mlp.gate_proj.weight"].float().cpu().numpy().T
        up = state[f"{prefix}.mlp.up_proj.weight"].float().cpu().numpy().T
        save_array(output_dir, manifest, f"{prefix}.mlp.up_gate_proj.weight", np.concatenate([gate, up], axis=1))

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(manifest)} tensors to {output_dir}")


def import_paddle(args):
    import paddle

    from paddleformers.transformers import MiniCPMConfig
    from paddleformers.transformers import (
        MiniCPMForCausalLMDeprecated as MiniCPMForCausalLM,
    )

    npy_dir = Path(args.npy_dir)
    hf_dir = Path(args.hf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = MiniCPMConfig.from_pretrained(str(hf_dir))
    config.architectures = ["MiniCPMForCausalLMDeprecated"]
    model = MiniCPMForCausalLM(config)
    state_dict = model.state_dict()
    manifest = json.loads((npy_dir / "manifest.json").read_text())
    converted = {}
    for key in state_dict:
        if key not in manifest:
            raise KeyError(f"Missing converted tensor for {key}")
        array = np.load(npy_dir / manifest[key])
        converted[key] = paddle.to_tensor(array, dtype=args.dtype)
    model.set_state_dict(converted)
    paddle.save(converted, str(output_dir / "model_state.pdparams"))
    config.save_pretrained(str(output_dir))

    for name in [
        "tokenizer.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ]:
        src = hf_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
    print(f"Saved Paddle checkpoint to {output_dir}")


def main():
    args = parse_args()
    if args.command == "export-npy":
        export_npy(args)
    elif args.command == "import-paddle":
        import_paddle(args)


if __name__ == "__main__":
    main()
