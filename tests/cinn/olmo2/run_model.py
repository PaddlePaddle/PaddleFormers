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

"""Original-width two-layer OLMo2; both modes reload a shared FP32 checkpoint."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import types
from contextlib import nullcontext
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dynamic", "cinn"], required=True)
    parser.add_argument("--task", choices=["train", "infer"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20, help="Total steps INCLUDING warmup updates")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0 or args.warmup < 0:
        parser.error("steps must be positive; warmup must be nonnegative")
    for key in os.environ:
        if key.startswith("FLAGS_prim") or "blacklist" in key.lower():
            raise RuntimeError(f"Unexpected inherited compiler policy: {key}")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    os.environ["FLAGS_use_cinn"] = "true" if args.mode == "cinn" else "false"
    os.environ["ENABLE_CINN_IN_DY2ST"] = "1" if args.mode == "cinn" else "0"
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo))
    import numpy as np
    import paddle
    import transformers
    from transformers import AutoTokenizer

    from paddleformers.transformers import Olmo2Config, Olmo2ForCausalLM
    from tests.cinn.checkpoint_utils import strict_set_state_dict
    from tests.cinn.olmo2.delivery_utils import (
        INITIAL_HASH,
        PARAMETERS,
        SEED,
        make_inputs,
        parameter_hash,
        sha256_file,
        validate_config,
    )
    from tests.cinn.olmo2.verify_assets import verify_assets

    entry = Path(__file__).parent
    verify_assets(
        json.loads((entry / "asset_manifest.json").read_text()),
        {"checkpoint": args.initial_checkpoint.parent, "assets": args.assets, "repo": entry},
    )
    if args.initial_checkpoint.name != "initial.pdparams":
        raise ValueError("Both tasks must load the manifest's initial.pdparams")
    raw = json.loads((args.assets / "config.json").read_text())
    validate_config(raw)
    args.output.mkdir(parents=True, exist_ok=False)
    paddle.set_device("gpu:0")
    paddle.seed(SEED)
    config = Olmo2Config(**raw)
    config._attn_implementation = "eager"
    model = Olmo2ForCausalLM(config)
    state = paddle.load(str(args.initial_checkpoint), return_numpy=True)
    if any(value.dtype != np.float32 or not np.isfinite(value).all() for value in state.values()):
        raise ValueError("Checkpoint must contain finite FP32 parameters")
    strict_set_state_dict(model, state)
    del state
    initial_hash = parameter_hash(model)
    if initial_hash != INITIAL_HASH or sum(int(v.numel()) for v in model.parameters()) != PARAMETERS:
        raise ValueError("Reloaded parameters differ from the frozen depth-only benchmark")

    @paddle.jit.not_to_static
    def config_get(self, key, default=None):
        return Olmo2Config.get(self, key, default)

    config.get = types.MethodType(config_get, config)
    tokenizer = AutoTokenizer.from_pretrained(str(args.assets), local_files_only=True)
    data = entry / "data/gsm8k_24.jsonl"
    tokens, labels, input_hash = make_inputs(tokenizer, data)
    np.savez(args.output / "inputs.npz", tokens=tokens, labels=labels)

    class Net(paddle.nn.Layer):
        def __init__(self, language_model):
            super().__init__()
            self.model = language_model

        def forward(self, ids):
            return self.model(ids, use_cache=False, return_dict=False)[0]

    net = Net(model)
    net.train() if args.task == "train" else net.eval()
    if args.mode == "cinn":
        net = paddle.jit.to_static(
            net, full_graph=True, backend="CINN", input_spec=[paddle.static.InputSpec([1, 128], "int64")]
        )
    optimizer = (
        paddle.optimizer.AdamW(learning_rate=2e-5, weight_decay=0.01, parameters=model.parameters())
        if args.task == "train"
        else None
    )
    xs = [paddle.to_tensor([x], dtype="int64") for x in tokens]
    ys = [paddle.to_tensor([y], dtype="int64") for y in labels]
    metadata = dict(
        mode=args.mode,
        task=args.task,
        seed=SEED,
        parameters=PARAMETERS,
        config=raw,
        initial_parameter_hash=initial_hash,
        initial_checkpoint_sha256=sha256_file(args.initial_checkpoint),
        input_sha256=input_hash,
        data_sha256=sha256_file(data),
        source_commit=subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        source_diff_sha256=hashlib.sha256(
            subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD"])
        ).hexdigest(),
        entry_sha256={p.name: sha256_file(p) for p in sorted(entry.glob("*.py"))},
        acceptance_sha256=sha256_file(entry / "acceptance.json"),
        shell_sha256={p.name: sha256_file(p) for p in sorted(entry.glob("*.sh"))},
        model_sha256=sha256_file(repo / "paddleformers/transformers/olmo2/modeling.py"),
        paddle=paddle.__version__,
        paddle_commit=paddle.version.commit,
        python=platform.python_version(),
        numpy=np.__version__,
        transformers=transformers.__version__,
        cuda=paddle.version.cuda(),
        cudnn=paddle.version.cudnn(),
        gpu=paddle.device.cuda.get_device_name(),
        precision="FP32; TF32 disabled; no AMP",
        batch=1,
        sequence_length=128,
        steps=args.steps,
        warmup=args.warmup,
        learning_rate=2e-5,
        weight_decay=0.01,
        training_samples=20,
        initialization="shared random depth-only checkpoint; NOT pretrained",
        boundary="full forward wrapper; loss and optimizer outside; no manual layer selection or blacklist",
        scope="fixed-input forward, not generation" if args.task == "infer" else "forward/loss/backward/AdamW",
    )
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)
    durations = []
    with (paddle.no_grad() if args.task == "infer" else nullcontext()):
        with (args.output / "steps.jsonl").open("w") as log:
            for step in range(args.steps):
                index = step % 20 if optimizer else 0
                paddle.device.synchronize()
                start = time.perf_counter()
                logits = net(xs[index])
                if optimizer:
                    loss = paddle.nn.functional.cross_entropy(
                        logits.reshape([-1, raw["vocab_size"]]), ys[index].reshape([-1]), ignore_index=-100
                    )
                    loss.backward()
                    if step == 0:
                        # All tensors are audited; save deterministic samples, not GB-sized gradient copies.
                        audit, samples = {}, {}
                        for name, parameter in model.named_parameters():
                            if parameter.stop_gradient:
                                continue
                            if parameter.grad is None:
                                raise RuntimeError(f"Missing gradient: {name}")
                            array = parameter.grad.numpy()
                            if not np.isfinite(array).all():
                                raise FloatingPointError(f"Nonfinite gradient: {name}")
                            flat = array.reshape(-1)
                            audit[name] = dict(max_abs=float(np.abs(flat).max()), nonzero=int(np.count_nonzero(flat)))
                            samples[name] = flat[np.linspace(0, len(flat) - 1, min(1024, len(flat)), dtype=int)].copy()
                        np.savez(args.output / "gradient_samples.npz", **samples)
                        (args.output / "gradient_audit.json").write_text(json.dumps(audit, indent=2))
                    optimizer.step()
                    optimizer.clear_grad()
                paddle.device.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                row = dict(
                    step=step, ms=elapsed, warmup=step < args.warmup, gradient_diagnostic=step == 0 and bool(optimizer)
                )
                if optimizer:
                    row["loss"] = float(loss)
                    if not np.isfinite(row["loss"]):
                        raise FloatingPointError(row)
                if step == 0:
                    array = logits.numpy()
                    if not np.isfinite(array).all():
                        raise FloatingPointError("Nonfinite initial logits")
                    np.save(args.output / "initial_logits.npy", array)
                log.write(json.dumps(row) + "\n")
                log.flush()
                if step % 100 == 0 or step + 1 == args.steps:
                    print(json.dumps(row), flush=True)
                if not row["warmup"] and not row["gradient_diagnostic"]:
                    durations.append(elapsed)
    for name, parameter in model.state_dict().items():
        if not bool(paddle.isfinite(parameter).all()):
            raise FloatingPointError(f"Nonfinite final parameter: {name}")
    if optimizer:
        for name, value in optimizer.state_dict().items():
            if isinstance(value, paddle.Tensor) and not bool(paddle.isfinite(value).all()):
                raise FloatingPointError(f"Nonfinite optimizer state: {name}")
    result = dict(
        metadata,
        status="completed",
        timed_steps=len(durations),
        median_ms=float(np.median(durations)) if durations else None,
        peak_allocated_bytes=paddle.device.cuda.max_memory_allocated(),
        final_parameter_hash=parameter_hash(model),
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print("COMPLETED", flush=True)


if __name__ == "__main__":
    main()
