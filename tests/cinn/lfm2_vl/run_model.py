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

"""Tiny LFM2-VL image-subset fitting; not full model quality evaluation."""
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

from paddleformers.transformers.lfm2_vl.configuration import (
    Lfm2Config,
    Lfm2VlConfig,
    Siglip2VisionConfig,
)
from paddleformers.transformers.lfm2_vl.modeling import Lfm2VlForConditionalGeneration
from tests.cinn.checkpoint_utils import strict_set_state_dict

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["dynamic", "cinn"], required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--split", action="store_true")
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--infer", action="store_true")
p.add_argument("--checkpoint", type=Path)
p.add_argument("--resume", type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
paddle.set_device("gpu:0")
paddle.seed(20260930)
text = Lfm2Config(
    vocab_size=128,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    layer_types=["conv", "full_attention", "conv"],
    block_auto_adjust_ff_dim=False,
    pad_token_id=0,
    use_cache=False,
)
text._attn_implementation = "eager"
vision = Siglip2VisionConfig(
    hidden_size=24,
    intermediate_size=48,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_patches=4,
    patch_size=8,
)
c = Lfm2VlConfig(
    text_config=text,
    vision_config=vision,
    image_token_id=7,
    projector_hidden_size=32,
    downsample_factor=2,
)
model = Lfm2VlForConditionalGeneration(c)
if a.checkpoint:
    strict_set_state_dict(model, paddle.load(str(a.checkpoint)))
resume_state = paddle.load(str(a.resume)) if a.resume else None
start_step = resume_state["step"] if resume_state else 0
if resume_state:
    strict_set_state_dict(model, resume_state["model"])
c.save_pretrained(str(a.output / "config"))
paddle.save(model.state_dict(), str(a.output / "initial.pdparams"))


class Boundary(paddle.nn.Layer):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids, mask, pixels, pixel_mask, shapes, labels):
        outputs = self.model(
            input_ids=ids,
            attention_mask=mask,
            pixel_values=pixels,
            pixel_attention_mask=pixel_mask,
            spatial_shapes=shapes,
            labels=None if a.infer else labels,
            use_cache=False,
        )
        return outputs.logits if a.infer else outputs.loss


net = Boundary(model)
if a.mode == "cinn" and a.split:
    for layer in model.model.vision_tower.vision_model.encoder.layers:
        layer.forward = paddle.jit.to_static(layer.forward, full_graph=True, backend="CINN")
    for layer in model.model.language_model.layers:
        layer.forward = paddle.jit.to_static(layer.forward, full_graph=True, backend="CINN")
elif a.mode == "cinn":
    net = paddle.jit.to_static(net, full_graph=True, backend="CINN")
net.train()
from PIL import Image

raw = np.load(str(Path(os.environ["CINN_RESOURCE_DIR"]) / "data/images24.npz"))
samples = []
for image, label in zip(raw["images"], raw["labels"]):
    tokens = [1, 7] + [10 + int(d) for d in f"{int(label):03d}"] + [2]
    ids = paddle.to_tensor([tokens], dtype="int64")
    im = np.asarray(Image.fromarray(image).resize((16, 16)), dtype="float32") / 127.5 - 1
    patches = im.reshape(2, 8, 2, 8, 3).transpose(0, 2, 1, 3, 4).reshape(1, 4, 192)
    samples.append(
        (
            ids,
            paddle.ones_like(ids),
            paddle.to_tensor(patches),
            paddle.ones([1, 4], dtype="int64"),
            paddle.to_tensor([[2, 2]], dtype="int64"),
            paddle.to_tensor([[-100, -100] + tokens[2:]], dtype="int64"),
        )
    )
inputs = samples[0]
(a.output / "experiment.json").write_text(
    json.dumps(
        dict(
            model="LFM2-VL random tiny",
            train_images=20,
            heldout_images=4,
            task="decimal class-ID tokens",
            image_size=16,
            patch_size=8,
            steps=a.steps,
            precision="FP32 TF32 off",
            split=a.split,
        ),
        indent=2,
    )
)
if a.infer:
    if not a.checkpoint:
        raise ValueError("shared checkpoint required")
    net.eval()
    times = []
    with paddle.no_grad():
        for i in range(40):
            paddle.device.synchronize()
            t = time.perf_counter()
            logits = net(*inputs)
            paddle.device.synchronize()
            if i >= 10:
                times.append(1000 * (time.perf_counter() - t))
    np.save(a.output / "logits.npy", logits.numpy())
    (a.output / "inference.json").write_text(
        json.dumps(
            dict(
                median_ms=float(np.median(times)),
                times_ms=times,
                checkpoint=str(a.checkpoint),
            )
        )
    )
    print("INFERENCE_FINISHED", flush=True)
    sys.exit(0)


def evaluate(step):
    net.eval()
    with paddle.no_grad():
        loss = float(np.mean([float(net(*x)) for x in samples[20:]]))
    net.train()
    with (a.output / "validation.jsonl").open("a") as f:
        f.write(json.dumps(dict(step=step, loss=loss)) + "\n")


evaluate(start_step)
opt = paddle.optimizer.AdamW(learning_rate=2e-5, parameters=model.parameters())
if resume_state:
    opt.set_state_dict(resume_state["optimizer"])
with (a.output / "loss.jsonl").open("w") as f:
    for step in range(start_step, a.steps):
        inputs = samples[step % 20]
        paddle.device.synchronize()
        t = time.perf_counter()
        loss = net(*inputs)
        loss.backward()
        if step == 0:
            np.savez(
                a.output / "gradients.npz",
                **{n: v.grad.numpy() for n, v in model.named_parameters() if v.grad is not None},
            )
        opt.step()
        opt.clear_grad()
        value = float(loss)
        if not np.isfinite(value):
            raise FloatingPointError(f"Non-finite training loss: {value}")
        paddle.device.synchronize()
        row = dict(step=step, loss=value, ms=1000 * (time.perf_counter() - t))
        f.write(json.dumps(row) + "\n")
        f.flush()
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
