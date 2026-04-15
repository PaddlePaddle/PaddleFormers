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

"""Run a one-step GSM8K training alignment test for Olmo2.

This script is intentionally split by backend because the local Torch and Paddle
stacks live in separate virtual environments.

Example:
  source /root/venvs/transformers_cu126/bin/activate
  CUDA_VISIBLE_DEVICES=1 python tests/transformers/olmo2/run_gsm8k_training_alignment.py --backend torch

  source /root/venvs/paddleformers_env/bin/activate
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/root/PaddleFormers \
    python tests/transformers/olmo2/run_gsm8k_training_alignment.py --backend paddle
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


DEFAULT_MODEL = "allenai/OLMo-2-0425-1B"
DEFAULT_WORK_DIR = Path("/tmp/olmo2_gsm8k_training_alignment")
DEFAULT_PROMPT_TEMPLATE = "Question: {question}\nAnswer:"
DEFAULT_GRAD_KEYS = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.q_norm.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "lm_head.weight",
]
LINEAR_SUFFIXES = (
    ".q_proj.weight",
    ".k_proj.weight",
    ".v_proj.weight",
    ".o_proj.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["torch", "paddle"], required=True)
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--dataset-name", default="openai/gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="train[:1]")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-7)
    parser.add_argument("--torch-backend", choices=["transformers", "ms-swift"], default="transformers")
    return parser.parse_args()


def format_gsm8k_example(example):
    prompt = DEFAULT_PROMPT_TEMPLATE.format(question=example["question"].strip())
    answer = " " + example["answer"].strip()
    return prompt, answer


def resolve_model_dir_torch(model_name_or_path):
    from huggingface_hub import snapshot_download

    path = Path(model_name_or_path)
    if path.exists():
        return path
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


def build_gsm8k_batch(tokenizer, dataset_name, dataset_config, split, max_length):
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    if len(dataset) < 1:
        raise ValueError(f"Empty dataset split: {dataset_name}/{dataset_config} {split}")

    prompt, answer = format_gsm8k_example(dataset[0])
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        answer_ids = answer_ids + [eos_token_id]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        if all(label == -100 for label in labels):
            raise ValueError("max_length truncated all answer labels; increase --max-length")

    attention_mask = [1] * len(input_ids)
    ignore_index = -100
    paddle_labels = labels[1:] + [ignore_index]

    return {
        "input_ids": np.array([input_ids], dtype=np.int64),
        "labels": np.array([labels], dtype=np.int64),
        "paddle_labels": np.array([paddle_labels], dtype=np.int64),
        "attention_mask": np.array([attention_mask], dtype=np.int64),
        "prompt": prompt,
        "answer": answer,
    }


def run_torch(args):
    if args.torch_backend == "ms-swift":
        raise NotImplementedError(
            "ms-swift is not installed in the current transformers environment. "
            "Install ms-swift and use its CLI for end-to-end SFT; this alignment script uses Transformers "
            "for deterministic tensor-level loss/gradient comparison."
        )

    import torch
    from transformers import AutoTokenizer, Olmo2ForCausalLM

    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    args.work_dir.mkdir(parents=True, exist_ok=True)
    model_dir = resolve_model_dir_torch(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    batch = build_gsm8k_batch(tokenizer, args.dataset_name, args.dataset_config, args.split, args.max_length)

    np.savez(
        args.work_dir / "gsm8k_batch.npz",
        input_ids=batch["input_ids"],
        labels=batch["labels"],
        paddle_labels=batch["paddle_labels"],
        attention_mask=batch["attention_mask"],
    )
    (args.work_dir / "sample.json").write_text(
        json.dumps({"prompt": batch["prompt"], "answer": batch["answer"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.work_dir / "model_dir.txt").write_text(str(model_dir), encoding="utf-8")

    model = Olmo2ForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32, device_map=None)
    model.config.use_cache = False
    model.train().cuda()

    input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device="cuda")
    labels = torch.tensor(batch["labels"], dtype=torch.long, device="cuda")
    attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long, device="cuda")

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
    outputs.loss.backward()

    ref = {
        "loss_before": np.array(outputs.loss.detach().cpu().float().numpy()),
        "logits_before": outputs.logits.detach().cpu().float().numpy(),
    }
    named_params = dict(model.named_parameters())
    for name in DEFAULT_GRAD_KEYS:
        ref[f"grad::{name}"] = named_params[name].grad.detach().cpu().float().numpy()

    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param -= args.learning_rate * param.grad
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        after = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
    ref["loss_after"] = np.array(after.loss.detach().cpu().float().numpy())
    ref["logits_after"] = after.logits.detach().cpu().float().numpy()
    np.savez(args.work_dir / "torch_ref.npz", **ref)

    print(f"model_dir: {model_dir}")
    print(f"batch_shape: {batch['input_ids'].shape}")
    print(f"loss_before: {float(ref['loss_before'])}")
    print(f"loss_after: {float(ref['loss_after'])}")


def build_olmo2_config_paddle(config_path):
    from paddleformers.transformers.olmo2 import Olmo2Config

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    return Olmo2Config(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        max_position_embeddings=cfg["max_position_embeddings"],
        initializer_range=cfg["initializer_range"],
        rms_norm_eps=cfg["rms_norm_eps"],
        use_cache=False,
        attention_dropout=cfg["attention_dropout"],
        attention_bias=cfg["attention_bias"],
        tie_word_embeddings=cfg["tie_word_embeddings"],
        pad_token_id=cfg["pad_token_id"],
        eos_token_id=cfg["eos_token_id"],
        rope_theta=cfg["rope_theta"],
        _attn_implementation="eager",
        fuse_rms_norm=False,
    )


def load_hf_safetensors_into_paddle(model, model_dir):
    import paddle
    from safetensors import safe_open

    with open(model_dir / "model.safetensors.index.json", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    opened = {}

    def get_array(name):
        filename = weight_map[name]
        if filename not in opened:
            opened[filename] = safe_open(str(model_dir / filename), framework="np")
        return opened[filename].get_tensor(name)

    state = {}
    for name, tensor in model.state_dict().items():
        arr = get_array(name)
        if name.endswith(LINEAR_SUFFIXES):
            arr = arr.T
        state[name] = paddle.to_tensor(np.ascontiguousarray(arr), dtype=tensor.dtype)
    model.set_state_dict(state)


def run_paddle(args):
    import paddle
    from paddleformers.transformers.olmo2 import Olmo2ForCausalLM

    paddle.seed(2026)
    paddle.set_default_dtype("float32")
    paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")

    model_dir_file = args.work_dir / "model_dir.txt"
    if not model_dir_file.exists():
        raise FileNotFoundError(f"Run --backend torch first to create {model_dir_file}")
    model_dir = Path(model_dir_file.read_text(encoding="utf-8").strip())
    batch_path = args.work_dir / "gsm8k_batch.npz"
    ref_path = args.work_dir / "torch_ref.npz"
    batch = np.load(batch_path)
    ref = np.load(ref_path)

    model = Olmo2ForCausalLM(build_olmo2_config_paddle(model_dir / "config.json"))
    load_hf_safetensors_into_paddle(model, model_dir)
    model.train()

    input_ids = paddle.to_tensor(batch["input_ids"], dtype="int64")
    if "paddle_labels" in batch:
        paddle_labels = batch["paddle_labels"]
    else:
        ignore_index = -100
        paddle_labels = np.concatenate(
            [
                batch["labels"][..., 1:],
                np.full(batch["labels"][..., :1].shape, ignore_index, dtype=batch["labels"].dtype),
            ],
            axis=-1,
        )
    labels = paddle.to_tensor(paddle_labels, dtype="int64")
    attention_mask = paddle.to_tensor(batch["attention_mask"], dtype="int64")

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False, return_dict=True)
    outputs.loss.backward()

    named_params = dict(model.named_parameters())
    metrics = {
        "loss_before_abs_diff": abs(float(outputs.loss.numpy()) - float(ref["loss_before"])),
        "logits_before_max_abs_diff": float(np.max(np.abs(outputs.logits.numpy() - ref["logits_before"]))),
        "logits_before_mean_abs_diff": float(np.mean(np.abs(outputs.logits.numpy() - ref["logits_before"]))),
    }
    for name in DEFAULT_GRAD_KEYS:
        grad = named_params[name].grad.numpy()
        ref_grad = ref[f"grad::{name}"]
        if name.endswith(LINEAR_SUFFIXES):
            ref_grad = ref_grad.T
        metrics[f"grad::{name}::max_abs_diff"] = float(np.max(np.abs(grad - ref_grad)))

    with paddle.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.set_value(param - args.learning_rate * param.grad)
    model.clear_gradients()
    with paddle.no_grad():
        after = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False, return_dict=True)

    metrics["loss_after_abs_diff"] = abs(float(after.loss.numpy()) - float(ref["loss_after"]))
    metrics["logits_after_max_abs_diff"] = float(np.max(np.abs(after.logits.numpy() - ref["logits_after"])))
    metrics["logits_after_mean_abs_diff"] = float(np.mean(np.abs(after.logits.numpy() - ref["logits_after"])))

    for key, value in metrics.items():
        print(f"{key}: {value}")
    (args.work_dir / "paddle_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    if args.backend == "torch":
        run_torch(args)
    else:
        run_paddle(args)


if __name__ == "__main__":
    main()
