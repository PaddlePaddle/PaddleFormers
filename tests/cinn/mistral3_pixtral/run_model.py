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

"""Tiny Mistral3/Pixtral image-subset fitting; not full model quality evaluation."""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
import argparse
import faulthandler
import json
import time

faulthandler.enable()
faulthandler.dump_traceback_later(90, repeat=True)
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import paddle

from paddleformers.transformers import Mistral3Config, Mistral3ForConditionalGeneration

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
c = Mistral3Config(
    text_config=dict(
        model_type="mistral",
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        rope_theta=1e9,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=4,
    ),
    vision_config=dict(
        model_type="pixtral",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_channels=3,
        image_size=24,
        patch_size=6,
        hidden_act="gelu",
    ),
    image_token_index=1,
    spatial_merge_size=2,
    vision_feature_layer=-1,
)
c.text_config._attn_implementation = "eager"
c.text_config.use_cache = False
model = Mistral3ForConditionalGeneration(c)
if a.checkpoint:
    model.set_state_dict(paddle.load(str(a.checkpoint)))
resume_state = paddle.load(str(a.resume)) if a.resume else None
start_step = resume_state["step"] if resume_state else 0
if resume_state:
    model.set_state_dict(resume_state["model"])
c.save_pretrained(str(a.output / "config"))
paddle.save(model.state_dict(), str(a.output / "initial.pdparams"))


class Boundary(paddle.nn.Layer):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids, mask, pixels, shapes, labels):
        return self.model(
            input_ids=ids,
            attention_mask=mask,
            pixel_values=pixels,
            image_sizes=shapes,
            labels=None if a.infer else labels,
            use_cache=False,
            return_dict=False,
        )[0]


net = Boundary(model)
if a.mode == "cinn" and a.split:
    for layer in model.model.vision_tower.transformer.layers:
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
    tokens = [2, 1, 1, 1, 1] + [10 + int(d) for d in f"{int(label):03d}"] + [3]
    ids = paddle.to_tensor([tokens[:-1]], dtype="int64")
    targets = [-100 if x == 1 else x for x in tokens[1:]]
    im = np.asarray(Image.fromarray(image).resize((24, 24)), dtype="float32") / 127.5 - 1
    samples.append(
        (
            ids,
            paddle.ones_like(ids),
            paddle.to_tensor(im.transpose(2, 0, 1)[None]),
            paddle.to_tensor([[24, 24]], dtype="int64"),
            paddle.to_tensor([targets], dtype="int64"),
        )
    )

inputs = samples[0]
(a.output / "experiment.json").write_text(
    json.dumps(
        dict(
            model="Mistral3/Pixtral random tiny",
            train_images=20,
            heldout_images=4,
            task="decimal class-ID tokens",
            image_size=24,
            patch_size=6,
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
        assert np.isfinite(value)
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
