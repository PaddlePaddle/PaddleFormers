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

"""Fixed-shape synthetic tiny Phi4 text/image compiler probe, not convergence."""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import paddle
from PIL import Image

from paddleformers.transformers import (
    AutoConfig,
    AutoProcessor,
    Phi4MultimodalForCausalLM,
)
from tests.cinn.checkpoint_utils import strict_set_state_dict

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["dynamic", "cinn"], required=True)
p.add_argument("--kind", choices=["text", "image"], required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--static-config-get", action="store_true")
p.add_argument("--split-components", action="store_true")
p.add_argument("--vision-only", action="store_true")
p.add_argument("--steps", type=int, default=20)
p.add_argument("--dataset", type=Path)
p.add_argument("--checkpoint", type=Path)
p.add_argument("--infer", action="store_true")
p.add_argument("--resume", type=Path)
p.add_argument("--save-every", type=int, default=500)
a = p.parse_args()
a.output.mkdir(exist_ok=False, parents=True)
paddle.set_device("gpu:0")
paddle.seed(20260922)
source = str(Path(os.environ["CINN_RESOURCE_DIR"]) / "assets")
c = AutoConfig.from_pretrained(source)
c._attn_implementation = "eager"
c.use_cache = False
model = Phi4MultimodalForCausalLM(c)
if a.checkpoint:
    if a.checkpoint.exists():
        strict_set_state_dict(model, paddle.load(str(a.checkpoint)))
    else:
        raise FileNotFoundError(a.checkpoint)
c.save_pretrained(str(a.output / "config"))
if a.static_config_get:
    import types

    @paddle.jit.not_to_static
    def config_get(self, key, default=None):
        return getattr(self, key, default)

    c.get = types.MethodType(config_get, c)
processor = AutoProcessor.from_pretrained(source)
if a.kind == "image":
    batch = processor(
        text=processor.image_token + " Describe this image.",
        images=[Image.fromarray(np.arange(28 * 28 * 3, dtype=np.uint8).reshape(28, 28, 3))],
        return_tensors="pd",
    )
else:
    batch = processor(text="Count to two. One, two.", return_tensors="pd")
print(
    {k: list(v.shape) for k, v in batch.items() if isinstance(v, paddle.Tensor)},
    flush=True,
)


class TextNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.model = model

    def forward(self, ids, mask):
        return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]


class ImageNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.model = model

    def forward(self, ids, mask, pixels, sizes, image_mask):
        return self.model(
            input_ids=ids,
            attention_mask=mask,
            image_pixel_values=pixels,
            image_sizes=sizes,
            image_attention_mask=image_mask,
            use_cache=False,
            return_dict=False,
        )[0]


args = [batch["input_ids"], batch["attention_mask"]]
if a.kind == "image":
    args += [
        batch["image_pixel_values"],
        batch["image_sizes"],
        batch["image_attention_mask"],
    ]
samples = []
if a.dataset:
    data = np.load(a.dataset)
    for im, label in zip(data["images"], data["labels"]):
        b = processor(
            text=processor.image_token + " Image class ID: " + str(int(label)) + ".",
            images=[Image.fromarray(im).resize((28, 28))],
            return_tensors="pd",
        )
        ids = b["input_ids"]
        mask = b["attention_mask"]
        pad = 32 - ids.shape[1]
        if pad < 0:
            raise ValueError(f"Image sample produces {ids.shape[1]} tokens; the benchmark limit is 32")
        ids = paddle.concat([ids, paddle.full([1, pad], c.pad_token_id, dtype="int64")], axis=1)
        mask = paddle.concat([mask, paddle.zeros([1, pad], dtype=mask.dtype)], axis=1)
        samples.append(
            [
                ids,
                mask,
                b["image_pixel_values"],
                b["image_sizes"],
                b["image_attention_mask"],
            ]
        )
    args = samples[0]
import hashlib
import subprocess

meta = {
    "source_commit": subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"], text=True
    ).strip(),
    "mode": a.mode,
    "steps": a.steps,
    "parameters": sum(int(v.numel()) for v in model.parameters()),
    "seed": 20260922,
    "paddle": paddle.__version__,
    "paddle_commit": paddle.version.commit,
    "precision": "FP32 TF32 off",
    "split_components": a.split_components,
    "position_implementation": "production_where_assignment",
    "dataset_sha256": hashlib.sha256(a.dataset.read_bytes()).hexdigest() if a.dataset else None,
    "checkpoint_sha256": hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() if a.checkpoint else None,
}
(a.output / "run.json").write_text(json.dumps(meta, indent=2))
net = TextNet() if a.kind == "text" else ImageNet()
np.savez(a.output / "inputs.npz", **{str(i): t.numpy() for i, t in enumerate(args)})
if a.mode == "cinn":
    if a.split_components:
        image_embed = model.model.embed_tokens_extend.image_embed
        image_embed.get_img_features = paddle.jit.to_static(
            image_embed.get_img_features, full_graph=True, backend="CINN"
        )
        if not a.vision_only:
            for layer in model.model.layers:
                layer.forward = paddle.jit.to_static(layer.forward, full_graph=True, backend="CINN")
    else:
        net = paddle.jit.to_static(
            net,
            full_graph=True,
            backend="CINN",
            input_spec=[paddle.static.InputSpec(t.shape, t.dtype) for t in args],
        )


def sample_loss(out, values):
    labels = values[0][:, 1:]
    labels = paddle.where(
        (values[1][:, 1:] == 0) | (labels >= 200000),
        paddle.full_like(labels, -100),
        labels,
    )
    return paddle.nn.functional.cross_entropy(out[:, :-1, :], labels)


def evaluate(step):
    net.eval()
    with paddle.no_grad():
        values = [float(sample_loss(net(*s), s)) for s in samples[20:]]
    with (a.output / "validation.jsonl").open("a") as f:
        f.write(json.dumps({"step": step, "loss": float(np.mean(values))}) + "\n")
    net.train()


if a.infer:
    net.eval()
    times = []
    with paddle.no_grad():
        for i in range(40):
            paddle.device.synchronize()
            t = time.perf_counter()
            out = net(*args)
            paddle.device.synchronize()
            if i >= 10:
                times.append(1000 * (time.perf_counter() - t))
        np.save(a.output / "logits.npy", out.numpy())
    (a.output / "inference.json").write_text(
        json.dumps(
            {
                "median_ms": float(np.median(times)),
                "times_ms": times,
                "warmup": 10,
                "repeats": 30,
            }
        )
    )
    print("INFERENCE_FINISHED", flush=True)
    sys.exit(0)
net.train()
opt = paddle.optimizer.AdamW(learning_rate=2e-5, parameters=model.parameters())
import random

start_step = 0
if a.resume:
    saved = paddle.load(str(a.resume))
    for key in [
        "mode",
        "dataset_sha256",
        "checkpoint_sha256",
        "position_implementation",
        "source_commit",
        "split_components",
        "paddle_commit",
    ]:
        if saved["meta"].get(key) != meta.get(key):
            raise ValueError(f"Resume mismatch: {key}")
    strict_set_state_dict(model, saved["model"])
    opt.set_state_dict(saved["optimizer"])
    start_step = saved["next_step"]
    paddle.set_rng_state(saved["cpu_rng"])
    paddle.set_cuda_rng_state(saved["cuda_rng"])
    np.random.set_state(saved["numpy_rng"])
    random.setstate(saved["python_rng"])
    if a.steps <= start_step:
        raise ValueError("--steps must exceed checkpoint step")
meta.update(start_step=start_step, resume=str(a.resume) if a.resume else None)
(a.output / "run.json").write_text(json.dumps(meta, indent=2))


def save_training_state(step):
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
            meta=meta,
        ),
        str(temporary),
    )
    os.replace(temporary, a.output / "training_state.pdstate")


if samples:
    evaluate(start_step)
with (a.output / "loss.jsonl").open("w") as f:
    for i in range(start_step, a.steps):
        if samples:
            args = samples[i % 20]
        paddle.device.synchronize()
        start = time.perf_counter()
        out = net(*args)
        if i == start_step:
            np.save(a.output / "initial.npy", out.detach().numpy())
        loss = sample_loss(out, args)
        loss.backward()
        if i == start_step:
            np.savez(
                a.output / "gradients.npz",
                **{n: v.grad.numpy() for n, v in model.named_parameters() if v.grad is not None},
            )
        opt.step()
        opt.clear_grad()
        paddle.device.synchronize()
        row = {
            "step": i,
            "loss": float(loss),
            "ms": 1000 * (time.perf_counter() - start),
        }
        if not np.isfinite(row["loss"]):
            raise FloatingPointError(f"Non-finite training loss at step {i}: {row['loss']}")
        f.write(json.dumps(row) + "\n")
        f.flush()
        print(row, flush=True)
        if samples and (i + 1) % 100 == 0:
            evaluate(i + 1)
        if a.save_every > 0 and (i + 1) % a.save_every == 0:
            save_training_state(i + 1)
save_training_state(a.steps)
if samples:
    paddle.save(model.state_dict(), str(a.output / "final.pdparams"))
print("FINISHED", flush=True)
