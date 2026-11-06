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

"""Tiny DeepSeek-OCR-2 image-subset fitting; not full model quality evaluation."""
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

from paddleformers.transformers import (
    DeepseekOCR2Config,
    DeepseekOCR2ForConditionalGeneration,
)

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["dynamic", "cinn"], required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--split", action="store_true")
p.add_argument("--keep-moe-dynamic", action="store_true")
p.add_argument("--static-config-get", action="store_true")
p.add_argument("--no-vision-cache", action="store_true")
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--infer", action="store_true")
p.add_argument("--checkpoint", type=Path)
p.add_argument("--resume", type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
paddle.set_device("gpu:0")
paddle.seed(20260930)
c = DeepseekOCR2Config(
    vocab_size=128,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=8,
    max_position_embeddings=512,
    moe_intermediate_size=16,
    n_routed_experts=4,
    n_shared_experts=1,
    num_experts_per_tok=2,
    first_k_dense_replace=1,
    n_group=1,
    topk_group=1,
    topk_method="greedy",
    use_mla=False,
    kv_lora_rank=None,
    q_lora_rank=None,
    qk_rope_head_dim=0,
    qk_nope_head_dim=0,
    v_head_dim=0,
    pad_token_id=0,
    bos_token_id=2,
    eos_token_id=3,
    vision_config=dict(
        encoder_embed_dim=16,
        encoder_depth=1,
        encoder_num_heads=2,
        encoder_global_attn_indexes=[],
        prompt_embed_dim=8,
        image_size=192,
        vit_patch_size=4,
        window_size=12,
        mlp_ratio=2,
        decoder_layer=1,
        hidden_dimension=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        _attn_implementation="eager",
    ),
)
c._attn_implementation = "eager"
c.use_cache = False
model = DeepseekOCR2ForConditionalGeneration(c)
if a.no_vision_cache:
    model.model.qwen2_model.model.model.config.use_cache = False
if a.checkpoint:
    model.set_state_dict(paddle.load(str(a.checkpoint)))
resume_state = paddle.load(str(a.resume)) if a.resume else None
start_step = resume_state["step"] if resume_state else 0
if resume_state:
    model.set_state_dict(resume_state["model"])
c.save_pretrained(str(a.output / "config"))
if a.static_config_get:
    import types

    @paddle.jit.not_to_static
    def config_get(self, key, default=None):
        return getattr(self, key, default)

    c.get = types.MethodType(config_get, c)
paddle.save(model.state_dict(), str(a.output / "initial.pdparams"))


class Boundary(paddle.nn.Layer):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids, mask, pixels, shapes, labels):
        return self.model(
            input_ids=ids,
            attention_mask=mask,
            images=[(paddle.zeros_like(pixels), pixels)],
            images_spatial_crop=shapes,
            images_seq_mask=ids == 1,
            labels=None if a.infer else labels,
            use_cache=False,
            return_dict=False,
        )[0]


net = Boundary(model)
if a.mode == "cinn" and a.split:
    for layer in model.model.qwen2_model.model.model.layers:
        layer.forward = paddle.jit.to_static(layer.forward, full_graph=True, backend="CINN")
    for layer in model.model.layers:
        if a.keep_moe_dynamic:
            layer.self_attn.forward = paddle.jit.to_static(layer.self_attn.forward, full_graph=True, backend="CINN")
            if not hasattr(layer.mlp, "gate"):
                layer.mlp.forward = paddle.jit.to_static(layer.mlp.forward, full_graph=True, backend="CINN")
        else:
            layer.forward = paddle.jit.to_static(layer.forward, full_graph=True, backend="CINN")
elif a.mode == "cinn":
    net = paddle.jit.to_static(net, full_graph=True, backend="CINN")
net.train()
from PIL import Image

raw = np.load(str(Path(os.environ["CINN_RESOURCE_DIR"]) / "data/images24.npz"))
samples = []
for image, label in zip(raw["images"], raw["labels"]):
    tokens = [2] + [1] * 145 + [10 + int(d) for d in f"{int(label):03d}"] + [3]
    ids = paddle.to_tensor([tokens[:-1]], dtype="int64")
    targets = [-100 if x == 1 else x for x in tokens[1:]]
    im = np.asarray(Image.fromarray(image).resize((192, 192)), dtype="float32") / 127.5 - 1
    samples.append(
        (
            ids,
            paddle.ones_like(ids),
            paddle.to_tensor(im.transpose(2, 0, 1)[None]),
            paddle.to_tensor([[1, 1]], dtype="int64"),
            paddle.to_tensor([targets], dtype="int64"),
        )
    )

inputs = samples[0]
(a.output / "experiment.json").write_text(
    json.dumps(
        dict(
            model="DeepSeek-OCR-2 random tiny",
            train_images=20,
            heldout_images=4,
            task="decimal class-ID tokens",
            image_size=192,
            patch_size=4,
            steps=a.steps,
            precision="FP32 TF32 off",
            split=a.split,
            keep_moe_dynamic=a.keep_moe_dynamic,
            static_config_get=a.static_config_get,
            no_vision_cache=a.no_vision_cache,
            vision_gradients="disabled by original model no_grad",
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
