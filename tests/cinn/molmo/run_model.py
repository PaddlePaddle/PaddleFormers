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

"""Small real-image/class-ID test; not full Molmo quality or convergence acceptance."""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
import argparse
import hashlib
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
from PIL import Image

repo = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, repo)
import paddle
from tokenizers import Tokenizer

from paddleformers.transformers import MolmoForCausalLM
from tests.transformers.molmo.test_modeling import MolmoModelTester

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["dynamic", "cinn"], required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--steps", type=int, default=300)
p.add_argument("--checkpoint", type=Path)
p.add_argument(
    "--resume",
    type=Path,
    help="Full training state; write continuation into a new output directory",
)
p.add_argument("--infer", action="store_true")
p.add_argument(
    "--split",
    action="store_true",
    help="Compile vision and decoder blocks; retain outer validation/merge in dynamic mode",
)
p.add_argument(
    "--split-blocks",
    action="store_true",
    help="Narrow vision boundary to individual transformer blocks",
)
p.add_argument("--trace-boundaries", action="store_true")
p.add_argument("--split-text-parts", action="store_true")
p.add_argument(
    "--split-safe",
    action="store_true",
    help="Vision blocks and text MLP only; text attention remains dynamic",
)
p.add_argument(
    "--split-rope",
    action="store_true",
    help="Also compile full text attention, with RoPE coefficient expansion outside",
)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
paddle.set_device("gpu:0")
paddle.seed(20260923)
tok = Tokenizer.from_file(str(Path(os.environ["CINN_RESOURCE_DIR"]) / "assets/tokenizer.json"))
c = MolmoModelTester(None, batch_size=1, vocab_size=tok.get_vocab_size(), max_position_embeddings=64).get_config(
    with_vision=True
)
c.vision_backbone.update(image_default_input_size=(28, 28), image_patch_size=14, image_pos_patch_size=14)
model = MolmoForCausalLM(c)
if a.infer and a.checkpoint is None:
    raise ValueError("Inference requires checkpoint")
if a.checkpoint:
    model.set_state_dict(paddle.load(str(a.checkpoint)))
c.save_pretrained(str(a.output / "config"))


@paddle.jit.not_to_static
def config_get(self, key, default=None):
    return getattr(self, key, default)


c.get = types.MethodType(config_get, c)
h = hashlib.sha256()
for n, v in sorted(model.state_dict().items()):
    h.update(n.encode())
    h.update(v.numpy().tobytes())
data = np.load(str(Path(os.environ["CINN_RESOURCE_DIR"]) / "data/images24.npz"))
samples = []
for im, label in zip(data["images"], data["labels"]):
    tokens = tok.encode("Image class ID: " + str(int(label)) + ".", add_special_tokens=False).ids
    values = ([1] + [0] * 4 + tokens + [2])[:33]
    size = len(values)
    x = np.array([values[:-1] + [0] * (33 - size)], dtype="int64")
    y = np.array([values[1:] + [-100] * (33 - size)], dtype="int64")
    y[:, :4] = -100
    pixels = np.asarray(Image.fromarray(im).resize((28, 28)), dtype="float32") / 127.5 - 1
    patches = pixels.reshape(2, 14, 2, 14, 3).transpose(0, 2, 1, 3, 4).reshape(1, 1, 4, 588)
    args = [
        paddle.to_tensor(x),
        paddle.to_tensor(patches),
        paddle.ones([1, 1, 4]),
        paddle.to_tensor([[[1, 2, 3, 4]]], dtype="int64"),
    ]
    samples.append((args, paddle.to_tensor(y)))
meta = {
    "mode": a.mode,
    "seed": 20260923,
    "initial_parameter_hash": h.hexdigest(),
    "parameters": sum(int(v.numel()) for v in model.parameters()),
    "paddle": paddle.__version__,
    "paddle_commit": paddle.version.commit,
    "source_commit": subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip(),
    "steps": a.steps,
    "precision": "FP32 TF32 off",
    "batch_size": 1,
    "optimizer": "AdamW lr=2e-5 default other parameters",
    "data": "20 ImageNet train + 4 official validation images, generated class-ID text",
    "tokenizer": "existing tiny CE Molmo tokenizer, not full upstream vocabulary",
    "preprocessing": "existing RGB224 center crops resized28 PIL; /127.5-1; four14x14 patches; sequence32; image/pad targets masked",
    "scope": "tiny random model, not full-scale task convergence",
    "adaptation": "config.get Python-only boundary; no tensor changes or blacklists",
}
(a.output / "run.json").write_text(json.dumps(meta, indent=2))
meta["compile_scope"] = (
    "vision backbone and decoder blocks; outer merge/checks/LM head dynamic" if a.split else "whole forward"
)
if a.split_blocks:
    meta["compile_scope"] = "individual vision residual blocks and decoder blocks; other operations dynamic"
if a.split_text_parts:
    meta[
        "compile_scope"
    ] = "vision residual blocks, text self-attention and MLP; text residual/norm and outer operations dynamic"
if a.split_safe:
    meta[
        "compile_scope"
    ] = "vision residual blocks and text MLP only; text attention, norms and outer operations dynamic"
if a.split_rope:
    meta[
        "compile_scope"
    ] = "vision residual blocks, text MLP, full text attention core; RoPE coefficient head-axis expansion and outer residual/norm/merge dynamic"
(a.output / "run.json").write_text(json.dumps(meta, indent=2))
np.savez(a.output / "inputs.npz", **{str(i): v.numpy() for i, v in enumerate(samples[0][0])})


class Net(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.model = model

    def forward(self, ids, images, masks, indices):
        return self.model(
            input_ids=ids,
            images=images,
            image_masks=masks,
            image_input_idx=indices,
            use_cache=False,
            return_dict=False,
        )[0]


net = Net()
if a.split_rope:
    from molmo_attention_boundary import install

    install(model, compile_core=a.mode == "cinn")
if a.trace_boundaries:

    def before(layer, args):
        print("ENTER", layer.full_name(), flush=True)

    def after(layer, args, out):
        print("EXIT", layer.full_name(), flush=True)

    for layer in [
        *model.model.vision_backbone.image_vit.transformer.resblocks,
        *model.model.layers,
        *[part for layer in model.model.layers for part in [layer.self_attn, layer.mlp]],
    ]:
        layer.register_forward_pre_hook(before)
        layer.register_forward_post_hook(after)
if a.mode == "cinn":
    if a.split or a.split_blocks or a.split_text_parts or a.split_safe or a.split_rope:
        if a.split_blocks or a.split_text_parts or a.split_safe or a.split_rope:
            for block in model.model.vision_backbone.image_vit.transformer.resblocks:
                block.forward = paddle.jit.to_static(block.forward, full_graph=True, backend="CINN")
        else:
            model.model.vision_backbone.forward = paddle.jit.to_static(
                model.model.vision_backbone.forward, full_graph=True, backend="CINN"
            )
        for layer in model.model.layers:
            if a.split_text_parts or a.split_safe or a.split_rope:
                if a.split_text_parts:
                    layer.self_attn.forward = paddle.jit.to_static(
                        layer.self_attn.forward, full_graph=True, backend="CINN"
                    )
                layer.mlp.forward = paddle.jit.to_static(layer.mlp.forward, full_graph=True, backend="CINN")
            else:
                layer.forward = paddle.jit.to_static(layer.forward, full_graph=True, backend="CINN")
    else:
        net = paddle.jit.to_static(
            net,
            full_graph=True,
            backend="CINN",
            input_spec=[paddle.static.InputSpec(v.shape, v.dtype) for v in samples[0][0]],
        )
if a.infer:
    net.eval()
    times = []
    with paddle.no_grad():
        for i in range(40):
            paddle.device.synchronize()
            t = time.perf_counter()
            out = net(*samples[20][0])
            paddle.device.synchronize()
            if i >= 10:
                times.append((time.perf_counter() - t) * 1000)
    np.save(a.output / "logits.npy", out.numpy())
    (a.output / "inference.json").write_text(
        json.dumps(
            {
                "median_ms": float(np.median(times)),
                "samples_ms": times,
                "warmup": 10,
                "scope": "checkpoint reload, fixed heldout image forward; not generation",
            },
            indent=2,
        )
    )
    print("INFERENCE_FINISHED", flush=True)
    sys.exit(0)
opt = paddle.optimizer.AdamW(learning_rate=2e-5, parameters=model.parameters())
import random

start_step = 0
if a.resume:
    # Locally produced legacy checkpoints contain PosixPath in CLI arguments.
    # Permit only that class during loading; retain the restricted unpickler.
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
    for key in [
        "mode",
        "seed",
        "initial_parameter_hash",
        "source_commit",
        "paddle_commit",
        "compile_scope",
    ]:
        if saved["meta"].get(key) != meta.get(key):
            raise ValueError(f"Resume mismatch: {key}")
    for key in [
        "split",
        "split_blocks",
        "split_text_parts",
        "split_safe",
        "split_rope",
    ]:
        if saved["arguments"].get(key) != getattr(a, key):
            raise ValueError(f"Resume boundary mismatch: {key}")
    model.set_state_dict(saved["model"])
    opt.set_state_dict(saved["optimizer"])
    start_step = saved["next_step"]
    if a.steps <= start_step:
        raise ValueError("--steps must exceed checkpoint step")
    paddle.set_rng_state(saved["cpu_rng"])
    paddle.set_cuda_rng_state(saved["cuda_rng"])
    np.random.set_state(saved["numpy_rng"])
    random.setstate(saved["python_rng"])
meta.update(start_step=start_step, resume=str(a.resume) if a.resume else None)
(a.output / "run.json").write_text(json.dumps(meta, indent=2))


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
            meta=meta,
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
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


evaluate(start_step)
with (a.output / "loss.jsonl").open("w") as f:
    for i in range(start_step, a.steps):
        args, y = samples[i % 20]
        paddle.device.synchronize()
        t = time.perf_counter()
        out = net(*args)
        if i == 0:
            np.save(a.output / "initial.npy", out.detach().numpy())
        loss = paddle.nn.functional.cross_entropy(out, y)
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
        if (i + 1) % 100 == 0:
            evaluate(i + 1)
        if (i + 1) % 500 == 0:
            save_training_state(i + 1)
save_training_state(a.steps)
paddle.save(model.state_dict(), str(a.output / "final.pdparams"))
print("FINISHED", flush=True)
