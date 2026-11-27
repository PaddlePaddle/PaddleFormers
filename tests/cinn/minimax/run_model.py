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

"""Tiny MiniMax default CINN smoke; not full-model convergence acceptance."""
import argparse
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import random

import numpy as np
import paddle
from transformers import AutoTokenizer

from paddleformers.transformers import MiniMaxConfig, MiniMaxForCausalLM
from tests.cinn.checkpoint_utils import strict_set_state_dict


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dynamic", "cinn"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--static-config-get", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument(
        "--resume",
        type=Path,
        help="Training-state checkpoint; output must be a new segment directory",
    )
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument(
        "--probe-gradients",
        action="store_true",
        help="Save first-step gradients before optimizer update",
    )
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.inference_only and (args.initial_checkpoint is None or not args.initial_checkpoint.is_file()):
        raise ValueError("Inference requires an existing --initial-checkpoint file")
    return args


class Net(paddle.nn.Layer):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids):
        return self.model(ids, use_cache=False, return_dict=False)[0]


def benchmark_forward(net, x, output, scope, save_logits=False):
    durations = []
    with paddle.no_grad():
        for j in range(40):
            paddle.device.synchronize()
            start = time.perf_counter()
            out = net(x)
            paddle.device.synchronize()
            if j >= 10:
                durations.append(1000 * (time.perf_counter() - start))
        if save_logits:
            np.save(output / "logits.npy", out.numpy())
    (output / "inference.json").write_text(
        json.dumps(
            {
                "median_ms": float(np.median(durations)),
                "times_ms": durations,
                "warmup": 10,
                "iterations": 30,
                "scope": scope,
            },
            indent=2,
        )
    )


def run_inference(args, net, tokens):
    net.eval()
    benchmark_forward(
        net,
        paddle.to_tensor([tokens[0]], dtype="int64"),
        args.output,
        "independent checkpoint reload; fixed-input forward, not generation",
        save_logits=True,
    )
    print("INFERENCE_FINISHED", flush=True)
    return


def run_training(args, model, net, tokens, labels, meta):
    optimizer = paddle.optimizer.AdamW(learning_rate=2e-5, weight_decay=0.01, parameters=model.parameters())

    start_step = 0
    if args.resume:
        saved = paddle.load(str(args.resume))
        for key in (
            "mode",
            "source_commit",
            "initial_checkpoint_sha256",
            "static_config_get",
            "moe_implementation",
        ):
            if saved["meta"].get(key) != meta.get(key):
                raise ValueError(f"Resume mismatch: {key}")
        if saved["tokens"] != tokens or saved["labels"] != labels:
            raise ValueError("Resume data mismatch")
        strict_set_state_dict(model, saved["model"])
        optimizer.set_state_dict(saved["optimizer"])
        start_step = saved["next_step"]
        paddle.set_rng_state(saved["cpu_rng"])
        paddle.set_cuda_rng_state(saved["cuda_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if args.steps <= start_step:
            raise ValueError("--steps must exceed checkpoint step")
    meta.update(
        start_step=start_step,
        resume=str(args.resume) if args.resume else None,
        save_every=args.save_every,
    )
    (args.output / "run.json").write_text(json.dumps(meta, indent=2))

    def save_training_state(step):
        payload = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            next_step=step,
            cpu_rng=paddle.get_rng_state(),
            cuda_rng=paddle.get_cuda_rng_state(),
            numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),
            tokens=tokens,
            labels=labels,
            meta=meta,
        )
        temporary = args.output / "training_state.tmp"
        paddle.save(payload, str(temporary))
        os.replace(temporary, args.output / "training_state.pdstate")

    def evaluate(step):
        net.eval()
        with paddle.no_grad():
            values = [
                float(
                    paddle.nn.functional.cross_entropy(
                        net(paddle.to_tensor([tokens[j]], dtype="int64")),
                        paddle.to_tensor([labels[j]], dtype="int64"),
                    )
                )
                for j in range(20, 24)
            ]
        with (args.output / "validation.jsonl").open("a") as vf:
            vf.write(json.dumps({"step": step, "loss": float(np.mean(values))}) + "\n")
        net.train()

    if args.extended:
        evaluate(start_step)
    net.train()
    with (args.output / "loss.jsonl").open("w") as f:
        for i in range(start_step, args.steps):
            x = paddle.to_tensor([tokens[i % 20]], dtype="int64")
            y = paddle.to_tensor([labels[i % 20]], dtype="int64")
            paddle.device.synchronize()
            t = time.perf_counter()
            out = net(x)
            if i == start_step:
                np.save(args.output / "initial_logits.npy", out.detach().numpy())
            loss = paddle.nn.functional.cross_entropy(out, y)
            loss_value = float(loss)
            if not np.isfinite(loss_value):
                raise FloatingPointError(f"Non-finite training loss at step {i}: {loss_value}")
            loss.backward()
            if args.probe_gradients and i == start_step:
                paddle.save(
                    {n: v.grad for n, v in model.named_parameters() if v.grad is not None},
                    str(args.output / "first_gradients.pdparams"),
                )
            optimizer.step()
            optimizer.clear_grad()
            paddle.device.synchronize()
            r = {"step": i, "loss": loss_value, "ms": 1000 * (time.perf_counter() - t)}
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r, flush=True)
            if args.extended and (i + 1) % 100 == 0:
                evaluate(i + 1)
            if args.save_every > 0 and (i + 1) % args.save_every == 0:
                save_training_state(i + 1)
    save_training_state(args.steps)
    net.eval()
    with paddle.no_grad():
        np.save(
            args.output / "final_logits.npy",
            net(paddle.to_tensor([tokens[0]], dtype="int64")).numpy(),
        )
    if args.extended:
        paddle.save(model.state_dict(), str(args.output / "final.pdparams"))
        benchmark_forward(
            net,
            paddle.to_tensor([tokens[0]], dtype="int64"),
            args.output,
            "in-process final-weight fixed-input forward, not generation or independent reload",
        )
    print("FINISHED", flush=True)


def main(argv=None):
    args = parse_args(argv)
    repo = str(Path(__file__).resolve().parents[3])
    args.output.mkdir(parents=True, exist_ok=False)
    paddle.set_device("gpu:0")
    paddle.seed(20260922)
    source = os.environ["CINN_ASSET_DIR"]
    config = MiniMaxConfig.from_pretrained(source)
    config._attn_implementation = "eager"
    config.use_cache = False
    model = MiniMaxForCausalLM(config)
    if args.initial_checkpoint:
        if args.initial_checkpoint.exists():
            strict_set_state_dict(model, paddle.load(str(args.initial_checkpoint)))
        else:
            raise FileNotFoundError(args.initial_checkpoint)
    config.save_pretrained(str(args.output / "config"))
    if args.static_config_get:

        @paddle.jit.not_to_static
        def config_get(self, key, default=None):
            return getattr(self, key, default)

        config.get = types.MethodType(config_get, config)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    rows = [json.loads(s) for s in Path(str(Path(__file__).parent / "data/gsm8k_24.jsonl")).read_text().splitlines()][
        : 24 if args.extended else 20
    ]
    tokens = []
    labels = []
    for row in rows:
        ids = tokenizer.encode(row["question"] + "\n" + row["answer"], add_special_tokens=False)[:65]
        n = len(ids)
        ids = ids + [tokenizer.eos_token_id] * (65 - n)
        lab = ids[1:]
        lab[max(n - 1, 0) :] = [-100] * (64 - max(n - 1, 0))
        tokens.append(ids[:-1])
        labels.append(lab)
    np.savez(args.output / "inputs.npz", tokens=tokens, labels=labels)

    net = Net(model)
    source_commit = (
        (Path(repo) / "SOURCE_COMMIT").read_text().strip()
        if (Path(repo) / "SOURCE_COMMIT").exists()
        else subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    )
    meta = {
        "mode": args.mode,
        "seed": 20260922,
        "parameters": sum(int(v.numel()) for v in model.parameters()),
        "paddle": paddle.__version__,
        "paddle_commit": paddle.version.commit,
        "source_commit": source_commit,
        "precision": "FP32 TF32 off",
        "steps": 20,
        "sequence_length": 64,
        "blacklist": [],
        "initialization": (
            f"loaded checkpoint {args.initial_checkpoint.name}"
            if args.initial_checkpoint
            else "fresh fixed-seed random; tiny CE config; original tokenizer"
        ),
    }
    meta["static_config_get"] = args.static_config_get
    meta["moe_implementation"] = "production_functional_accumulation"
    meta["steps"] = args.steps
    meta["training_samples"] = 20
    meta["validation_samples"] = 4 if args.extended else 0
    if args.initial_checkpoint:
        import hashlib

        meta["initial_checkpoint_sha256"] = hashlib.sha256(args.initial_checkpoint.read_bytes()).hexdigest()
    (args.output / "run.json").write_text(json.dumps(meta, indent=2))
    print(meta, flush=True)
    if args.mode == "cinn":
        net = paddle.jit.to_static(
            net,
            full_graph=True,
            backend="CINN",
            input_spec=[paddle.static.InputSpec([1, 64], "int64")],
        )
    if args.inference_only:
        run_inference(args, net, tokens)
    else:
        run_training(args, model, net, tokens, labels, meta)


if __name__ == "__main__":
    main()
