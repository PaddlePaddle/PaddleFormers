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

"""Tiny FastVLM real-image subset fitting; class-ID digit tokens, not caption quality."""
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

from paddleformers.transformers.fastvlm.configuration import FastVLMConfig
from paddleformers.transformers.fastvlm.modeling import FastVLMForConditionalGeneration

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["dynamic", "cinn"], required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--split", action="store_true")
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--infer", action="store_true")
p.add_argument("--checkpoint", type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
paddle.set_device("gpu:0")
paddle.seed(20260930)
c = FastVLMConfig(
    vocab_size=128,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=2,
    mm_vision_tower="mobileclip_l_64",
    mm_hidden_size=128,
    vision_config={"layers": [1, 1, 1, 1], "embed_dims": [8, 16, 32, 64]},
    tie_word_embeddings=False,
    use_cache=False,
)
c._attn_implementation = "eager"
model = FastVLMForConditionalGeneration(c)
if a.checkpoint:
    model.set_state_dict(paddle.load(str(a.checkpoint)))
c.save_pretrained(str(a.output / "config"))
paddle.save(model.state_dict(), str(a.output / "initial.pdparams"))


class Boundary(paddle.nn.Layer):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids, pixels, labels):
        if a.infer:
            return self.model(input_ids=ids, pixel_values=pixels, use_cache=False, return_dict=False)[0]
        return self.model(
            input_ids=ids,
            pixel_values=pixels,
            labels=labels,
            use_cache=False,
            return_dict=False,
        )[0]


net = Boundary(model)
if a.mode == "cinn" and a.split:
    model.model.vision_tower.forward = paddle.jit.to_static(
        model.model.vision_tower.forward, full_graph=True, backend="CINN"
    )
    for layer in model.model.layers:
        layer.forward = paddle.jit.to_static(layer.forward, full_graph=True, backend="CINN")
elif a.mode == "cinn":
    net = paddle.jit.to_static(net, full_graph=True, backend="CINN")
net.train()
ids = paddle.to_tensor([[1, -200, 2, 3]], dtype="int64")
labels = paddle.to_tensor([[1, -100, 2, 3]], dtype="int64")
from PIL import Image

raw = np.load(str(Path(os.environ["CINN_RESOURCE_DIR"]) / "data/images24.npz"))
samples = []
for image, label in zip(raw["images"], raw["labels"]):
    # BOS, image placeholder, three decimal digit tokens, EOS; fixed 6 tokens.
    tokens = [1, -200] + [10 + int(d) for d in f"{int(label):03d}"] + [2]
    target = [-100, -100] + tokens[2:]
    pixels_np = np.asarray(Image.fromarray(image).resize((64, 64)), dtype="float32")
    pixels_np = (pixels_np / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    samples.append(
        (
            paddle.to_tensor([tokens], dtype="int64"),
            paddle.to_tensor(pixels_np.transpose(2, 0, 1)[None].astype("float32")),
            paddle.to_tensor([target], dtype="int64"),
        )
    )
ids, pixels, labels = samples[0]
(a.output / "experiment.json").write_text(
    json.dumps(
        dict(
            initialization="random tiny",
            train_samples=20,
            heldout_samples=4,
            steps=a.steps,
            precision="FP32 TF32 off",
            seed=20260930,
            task="image class-ID decimal digits; not natural captioning",
            split=a.split,
            source_commit=__import__("subprocess")
            .check_output(["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"], text=True)
            .strip(),
        ),
        indent=2,
    )
)
np.savez(
    a.output / "inputs.npz",
    ids=ids.numpy(),
    labels=labels.numpy(),
    pixels=pixels.numpy(),
)
if a.infer:
    if not a.checkpoint:
        raise ValueError("Inference requires a shared checkpoint")
    net.eval()
    times = []
    with paddle.no_grad():
        for i in range(40):
            paddle.device.synchronize()
            start = time.perf_counter()
            out = net(ids, pixels, labels)
            paddle.device.synchronize()
            if i >= 10:
                times.append(1000 * (time.perf_counter() - start))
    np.save(a.output / "logits.npy", out.numpy())
    (a.output / "inference.json").write_text(
        json.dumps(
            dict(
                median_ms=float(np.median(times)),
                times_ms=times,
                scope="fixed-image forward logits, not generation",
                checkpoint=str(a.checkpoint),
            ),
            indent=2,
        )
    )
    print("INFERENCE_FINISHED", flush=True)
    sys.exit(0)
opt = paddle.optimizer.AdamW(learning_rate=2e-5, parameters=model.parameters())


def evaluate(step):
    net.eval()
    with paddle.no_grad():
        losses = [float(net(*sample)) for sample in samples[20:]]
    net.train()
    with (a.output / "validation.jsonl").open("a") as f:
        f.write(json.dumps(dict(step=step, loss=float(np.mean(losses)))) + "\n")


# Evaluate after training begins too, to exercise train/eval graph transitions.
evaluate(0)
with (a.output / "loss.jsonl").open("w") as log:
    for step in range(a.steps):
        ids, pixels, labels = samples[step % 20]
        paddle.device.synchronize()
        start = time.perf_counter()
        loss = net(ids, pixels, labels)
        loss.backward()
        if step == 0:
            np.savez(
                a.output / "gradients.npz",
                **{n: v.grad.numpy() for n, v in model.named_parameters() if v.grad is not None},
            )
        opt.step()
        opt.clear_grad()
        paddle.device.synchronize()
        value = float(loss)
        if not np.isfinite(value):
            raise RuntimeError(f"Nonfinite loss: {value}")
        row = dict(step=step, loss=value, ms=1000 * (time.perf_counter() - start))
        log.write(json.dumps(row) + "\n")
        log.flush()
        if step % 100 == 0:
            print(json.dumps(row), flush=True)
        if (step + 1) % 500 == 0:
            evaluate(step + 1)
            paddle.save(
                dict(model=model.state_dict(), optimizer=opt.state_dict(), step=step + 1),
                str(a.output / "training_state.pdstate"),
            )
paddle.save(model.state_dict(), str(a.output / "final.pdparams"))
print("FINISHED", a.steps, flush=True)
