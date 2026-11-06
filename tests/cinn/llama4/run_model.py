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

"""Separate tiny Llama4 MoE text and dense Llama Guard4 image CINN probes."""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

repo = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, repo)
import paddle

from paddleformers.transformers import Llama4ForCausalLM, Llama4TextConfig

p = argparse.ArgumentParser()
p.add_argument("--kind", choices=["text"], default="text")
p.add_argument("--mode", choices=["dynamic", "cinn"], required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--adapt", action="store_true")
p.add_argument("--compile-vision", action="store_true")
p.add_argument("--real", action="store_true")
p.add_argument("--steps", type=int, default=20)
p.add_argument("--checkpoint", type=Path)
p.add_argument("--infer", action="store_true")
p.add_argument("--warmup", type=int, default=50)
p.add_argument("--iterations", type=int, default=200)
p.add_argument("--resume", type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
if a.infer and (a.checkpoint is None or not a.checkpoint.is_file()):
    raise ValueError("Independent inference requires an existing checkpoint")
paddle.set_device("gpu:0")
paddle.seed(20260922)
c = Llama4TextConfig.from_pretrained(str(Path(os.environ["CINN_RESOURCE_DIR"]) / "assets"))
c._attn_implementation = "eager"
c.use_cache = False
model = Llama4ForCausalLM(c)
if a.checkpoint:
    if a.checkpoint.exists():
        model.set_state_dict(paddle.load(str(a.checkpoint)))
    else:
        raise FileNotFoundError(a.checkpoint)
c.save_pretrained(str(a.output / "config"))
if a.adapt:
    import types

    @paddle.jit.not_to_static
    def config_get(self, key, default=None):
        return getattr(self, key, default)

    tc = c if a.kind == "text" else c.text_config
    tc.get = types.MethodType(config_get, tc)
state = model.state_dict()
import hashlib

digest = hashlib.sha256()
for name in sorted(state):
    digest.update(name.encode())
    digest.update(state[name].numpy().tobytes())
source_commit = (
    (Path(repo) / "SOURCE_COMMIT").read_text().strip()
    if (Path(repo) / "SOURCE_COMMIT").exists()
    else subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
)
(a.output / "run.json").write_text(
    json.dumps(
        {
            "kind": a.kind,
            "mode": a.mode,
            "initial_parameter_hash": digest.hexdigest(),
            "parameters": sum(int(v.numel()) for v in model.parameters()),
            "source_commit": source_commit,
            "paddle": paddle.__version__,
            "paddle_commit": paddle.version.commit,
            "scope": "independent inference"
            if a.infer
            else ("real-data tiny validation" if a.real else "synthetic smoke"),
            "steps": 0 if a.infer else a.steps,
        },
        indent=2,
    )
)
ids = paddle.to_tensor([[1, 100, 2, 3, 4, 5, 6, 7]], dtype="int64")
pixels = paddle.randn([1, 3, 28, 28])
samples = []
if a.real:
    from PIL import Image
    from transformers import AutoTokenizer

    token_path = str(Path(os.environ["CINN_RESOURCE_DIR"]) / "assets") if a.kind == "text" else ""
    tok = AutoTokenizer.from_pretrained(token_path, local_files_only=True)
    if a.kind == "text":
        data = [
            json.loads(s)
            for s in Path(str(Path(os.environ["CINN_RESOURCE_DIR"]) / "data/gsm8k_24.jsonl")).read_text().splitlines()
        ][:24]
        texts = [v["question"] + "\n" + v["answer"] for v in data]
    else:
        data = np.load("")
        texts = ["Image class ID: " + str(int(v)) + "." for v in data["labels"]]
    for i, text in enumerate(texts):
        values = tok.encode(text, add_special_tokens=False)
        if a.kind == "guard":
            values = [c.image_token_index] + values
        values = values[:65]
        n = len(values)
        padded = values + [0] * (65 - n)
        x = paddle.to_tensor([padded[:-1]], dtype="int64")
        y = paddle.to_tensor([padded[1:n] + [-100] * (65 - n)], dtype="int64")
        if a.kind == "guard":
            im = np.asarray(Image.fromarray(data["images"][i]).resize((28, 28)), dtype="float32") / 127.5 - 1
            sample = [x, paddle.to_tensor(im.transpose(2, 0, 1)[None])]
        else:
            sample = [x]
        samples.append((sample, y))
    ids = samples[0][0][0]
    if a.kind == "guard":
        pixels = samples[0][0][1]
    (a.output / "experiment.json").write_text(
        json.dumps(
            {
                "steps": a.steps,
                "train_samples": 20,
                "heldout_samples": 4,
                "sequence_length": 64,
                "tokenizer": token_path,
                "data": "GSM8K first24" if a.kind == "text" else "ImageNet20train4val with class-ID text",
                "image_preprocessing": "RGB resize28 PIL default, /127.5-1; fixed test protocol, not original high-res tiling"
                if a.kind == "guard"
                else None,
                "precision": "FP32 TF32 off",
                "checkpoint": str(a.checkpoint),
                "adapt": a.adapt,
                "compile_vision": a.compile_vision,
            },
            indent=2,
        )
    )


class TextNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.model = model

    def forward(self, ids):
        return self.model(input_ids=ids, use_cache=False, return_dict=False)[0]


class GuardNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.model = model

    def forward(self, ids, pixels):
        return self.model(input_ids=ids, pixel_values=pixels, use_cache=False, return_dict=False)[0]


net = TextNet() if a.kind == "text" else GuardNet()
args = [ids] if a.kind == "text" else [ids, pixels]
np.savez(a.output / "inputs.npz", **{str(i): v.numpy() for i, v in enumerate(args)})
if a.mode == "cinn":
    if a.adapt and a.kind == "guard":
        if a.compile_vision:
            from paddleformers.transformers.model_outputs import (
                BaseModelOutputWithPooling,
            )

            class VisionTensor(paddle.nn.Layer):
                def __init__(self, vision):
                    super().__init__()
                    self.vision = vision

                def forward(self, pixels):
                    return self.vision(pixels, return_dict=False)[0]

            class VisionBoundary(paddle.nn.Layer):
                def __init__(self, vision):
                    super().__init__()
                    self.compiled = paddle.jit.to_static(
                        VisionTensor(vision),
                        full_graph=True,
                        backend="CINN",
                        input_spec=[paddle.static.InputSpec([1, 3, 28, 28], "float32")],
                    )

                def forward(self, pixel_values, **kwargs):
                    return BaseModelOutputWithPooling(last_hidden_state=self.compiled(pixel_values))

            model.vision_model = VisionBoundary(model.vision_model)
        for layer in model.language_model.model.layers:
            layer.forward = paddle.jit.to_static(layer.forward, full_graph=True, backend="CINN")
    else:
        net = paddle.jit.to_static(
            net,
            full_graph=True,
            backend="CINN",
            input_spec=[paddle.static.InputSpec(v.shape, v.dtype) for v in args],
        )
if a.infer:
    net.eval()
    durations = []
    with paddle.no_grad():
        for i in range(a.warmup + a.iterations):
            paddle.device.synchronize()
            start = time.perf_counter()
            out = net(*args)
            paddle.device.synchronize()
            if i >= a.warmup:
                durations.append(1000 * (time.perf_counter() - start))
        np.save(a.output / "logits.npy", out.numpy())
    (a.output / "inference.json").write_text(
        json.dumps(
            {
                "median_ms": float(np.median(durations)),
                "warmup": a.warmup,
                "iterations": a.iterations,
                "mean_ms": float(np.mean(durations)),
                "times_ms": durations,
                "scope": "independent checkpoint reload; fixed-input forward, not generation",
            },
            indent=2,
        )
    )
    print("INFERENCE_FINISHED", flush=True)
    sys.exit(0)
net.train()
opt = paddle.optimizer.AdamW(learning_rate=2e-5, parameters=model.parameters())
import random

start_step = 0
if a.resume:
    # Trusted local legacy training states contain PosixPath CLI values.
    from paddle.framework import restricted_unpickler

    previous_allowed = restricted_unpickler._ALLOWED_CLASSES.get("pathlib")
    restricted_unpickler._ALLOWED_CLASSES["pathlib"] = {"PosixPath"}
    try:
        saved = paddle.load(str(a.resume))
    finally:
        if previous_allowed is None:
            restricted_unpickler._ALLOWED_CLASSES.pop("pathlib", None)
        else:
            restricted_unpickler._ALLOWED_CLASSES["pathlib"] = previous_allowed
    for key in ["kind", "mode", "adapt", "compile_vision", "real"]:
        if saved["arguments"].get(key) != getattr(a, key):
            raise ValueError(f"Resume mismatch: {key}")
    if saved["source_commit"] != source_commit or saved["initial_parameter_hash"] != digest.hexdigest():
        raise ValueError("Resume source/initial weights mismatch")
    model.set_state_dict(saved["model"])
    opt.set_state_dict(saved["optimizer"])
    start_step = saved["next_step"]
    if a.steps <= start_step:
        raise ValueError("--steps must exceed checkpoint step")
    paddle.set_rng_state(saved["cpu_rng"])
    paddle.set_cuda_rng_state(saved["cuda_rng"])
    np.random.set_state(saved["numpy_rng"])
    random.setstate(saved["python_rng"])
    (a.output / "resume.json").write_text(
        json.dumps(
            dict(checkpoint=str(a.resume), start_step=start_step, target_step=a.steps),
            indent=2,
        )
    )


def save_training_state(step):
    import random

    temporary = a.output / "training_state.tmp"
    paddle.save(
        dict(
            model=model.state_dict(),
            optimizer=opt.state_dict(),
            next_step=step,
            cpu_rng=paddle.get_rng_state(),
            cuda_rng=paddle.get_cuda_rng_state(),
            numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
            source_commit=source_commit,
            initial_parameter_hash=digest.hexdigest(),
        ),
        str(temporary),
    )
    os.replace(temporary, a.output / "training_state.pdstate")


def evaluate(step):
    net.eval()
    with paddle.no_grad():
        values = [float(paddle.nn.functional.cross_entropy(net(*s), y)) for s, y in samples[20:]]
    with (a.output / "validation.jsonl").open("a") as f:
        f.write(json.dumps({"step": step, "loss": float(np.mean(values))}) + "\n")
    net.train()


if samples:
    evaluate(start_step)
with (a.output / "loss.jsonl").open("w") as f:
    for i in range(start_step, a.steps):
        if samples:
            args, labels = samples[i % 20]
        paddle.device.synchronize()
        t = time.perf_counter()
        out = net(*args)
        if i == 0:
            np.save(a.output / "initial.npy", out.detach().numpy())
        loss = (
            paddle.nn.functional.cross_entropy(out, labels)
            if samples
            else paddle.nn.functional.cross_entropy(out[:, :-1, :], ids[:, 1:])
        )
        loss.backward()
        if i == 0:
            np.savez(
                a.output / "grads.npz",
                **{n: v.grad.numpy() for n, v in model.named_parameters() if v.grad is not None},
            )
        opt.step()
        opt.clear_grad()
        paddle.device.synchronize()
        row = {"step": i, "loss": float(loss), "ms": 1000 * (time.perf_counter() - t)}
        assert np.isfinite(row["loss"])
        f.write(json.dumps(row) + "\n")
        f.flush()
        print(row, flush=True)
        if samples and (i + 1) % 100 == 0:
            evaluate(i + 1)
        if (i + 1) % 500 == 0:
            save_training_state(i + 1)
save_training_state(a.steps)
if samples:
    paddle.save(model.state_dict(), str(a.output / "final.pdparams"))
print("FINISHED", flush=True)
