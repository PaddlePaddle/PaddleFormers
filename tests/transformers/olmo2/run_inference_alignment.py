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

"""Run an official-weight single-card inference alignment test for Olmo2.

This script covers the 2.5 interface path and first-20-token logits alignment:

1. Paddle interface smoke test:
   AutoConfig / AutoTokenizer / AutoModelForCausalLM.from_pretrained(convert_from_hf=True)
2. Torch/Paddle first-20-token logits comparison under official weights

Example:
  source /root/venvs/transformers_cu126/bin/activate
  CUDA_VISIBLE_DEVICES=1 python tests/transformers/olmo2/run_inference_alignment.py --backend torch

  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/root/PaddleFormers \
    python tests/transformers/olmo2/run_inference_alignment.py --backend paddle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "allenai/OLMo-2-0425-1B"
DEFAULT_WORK_DIR = Path("/tmp/olmo2_inference_alignment")
DEFAULT_PROMPT = (
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?\nAnswer:"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["torch", "paddle"], required=True)
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--first-n-tokens", type=int, default=20)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--verify-save-to-hf", action="store_true")
    return parser.parse_args()


def resolve_model_dir(model_name_or_path):
    path = Path(model_name_or_path)
    if path.exists():
        return path

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model_name_or_path,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "model*.safetensors",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
            ],
        )
    )


def run_torch(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    args.work_dir.mkdir(parents=True, exist_ok=True)
    model_dir = resolve_model_dir(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32, device_map=None)
    model.eval().cuda()

    model_inputs = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        outputs = model(**model_inputs, use_cache=False)

    first_n = min(args.first_n_tokens, outputs.logits.shape[1])
    ref = {
        "input_ids": model_inputs["input_ids"].detach().cpu().numpy().astype("int64"),
        "logits_first_n": outputs.logits[:, :first_n, :].detach().cpu().float().numpy(),
        "first_n": np.array(first_n, dtype="int64"),
    }
    np.savez(args.work_dir / "torch_inference_ref.npz", **ref)
    (args.work_dir / "model_dir.txt").write_text(str(model_dir), encoding="utf-8")
    (args.work_dir / "prompt.txt").write_text(args.prompt, encoding="utf-8")

    print(f"model_dir: {model_dir}")
    print(f"prompt_len: {model_inputs['input_ids'].shape[-1]}")
    print(f"first_n_tokens: {first_n}")
    print(f"torch_logits_first_n_shape: {ref['logits_first_n'].shape}")

    if args.verify_save_to_hf:
        export_dir = args.work_dir / "paddle_hf_export"
        export_model = AutoModelForCausalLM.from_pretrained(export_dir, torch_dtype=torch.float32, device_map=None)
        export_model.eval().cuda()
        export_inputs = {
            "input_ids": torch.as_tensor(ref["input_ids"], dtype=torch.long, device=export_model.device),
        }
        with torch.no_grad():
            export_outputs = export_model(**export_inputs, use_cache=False)

        export_logits = export_outputs.logits[:, :first_n, :].detach().cpu().float().numpy()
        export_abs_diff = np.abs(export_logits - ref["logits_first_n"].astype("float32"))
        export_metrics = {
            "first_n_tokens": first_n,
            "export_global_mean_abs_diff": float(export_abs_diff.mean()),
            "export_global_max_abs_diff": float(export_abs_diff.max()),
        }
        export_metrics_path = args.work_dir / "torch_export_metrics.json"
        export_metrics_path.write_text(json.dumps(export_metrics, indent=2), encoding="utf-8")
        print(f"export_global_mean_abs_diff: {export_metrics['export_global_mean_abs_diff']}")
        print(f"export_global_max_abs_diff: {export_metrics['export_global_max_abs_diff']}")


def run_paddle(args):
    import paddle

    from paddleformers.transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    paddle.seed(2026)
    paddle.set_default_dtype("float32")
    paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")

    model_dir = resolve_model_dir(args.model_name_or_path)
    config = AutoConfig.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        convert_from_hf=True,
        dtype=args.dtype,
        _attn_implementation="eager",
    ).eval()

    smoke_inputs = tokenizer("Hello world", return_tensors="pd")
    smoke_inputs.pop("attention_mask", None)
    smoke_outputs = model.generate(**smoke_inputs, max_new_tokens=4, do_sample=False)
    generated_ids = smoke_outputs[0][0].tolist() if isinstance(smoke_outputs, tuple) else smoke_outputs[0].tolist()
    print(f"paddle_interface_smoke_model_type: {config.model_type}")
    print(f"paddle_interface_smoke_generate: {tokenizer.decode(generated_ids)!r}")

    input_features = tokenizer(args.prompt, return_tensors="pd", add_special_tokens=False)
    with paddle.no_grad():
        outputs = model(**input_features, use_cache=False, return_dict=True)

    ref_path = args.work_dir / "torch_inference_ref.npz"
    ref = np.load(ref_path)
    first_n = int(ref["first_n"])
    paddle_logits = outputs.logits[:, :first_n, :].numpy().astype("float32")
    torch_logits = ref["logits_first_n"].astype("float32")
    abs_diff = np.abs(paddle_logits - torch_logits)

    token_mean_abs_diff = abs_diff.mean(axis=(0, 2))
    token_max_abs_diff = abs_diff.max(axis=(0, 2))

    metrics = {
        "first_n_tokens": first_n,
        "global_mean_abs_diff": float(abs_diff.mean()),
        "global_max_abs_diff": float(abs_diff.max()),
        "token_mean_abs_diff": token_mean_abs_diff.tolist(),
        "token_max_abs_diff": token_max_abs_diff.tolist(),
    }
    metrics_path = args.work_dir / "paddle_inference_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"prompt_len: {input_features['input_ids'].shape[-1]}")
    print(f"first_n_tokens: {first_n}")
    print(f"global_mean_abs_diff: {metrics['global_mean_abs_diff']}")
    print(f"global_max_abs_diff: {metrics['global_max_abs_diff']}")
    for idx, (mean_diff, max_diff) in enumerate(zip(token_mean_abs_diff.tolist(), token_max_abs_diff.tolist())):
        print(f"token_{idx:02d}_mean_abs_diff: {mean_diff}")
        print(f"token_{idx:02d}_max_abs_diff: {max_diff}")

    export_dir = args.work_dir / "paddle_hf_export"
    model.save_pretrained(export_dir, save_to_hf=True)
    print(f"save_to_hf_export_dir: {export_dir}")


def main():
    args = parse_args()
    if args.backend == "torch":
        run_torch(args)
    else:
        run_paddle(args)


if __name__ == "__main__":
    main()
