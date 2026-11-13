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

"""Shared Llama4/Guard4 CINN runner; import does not start a GPU run."""

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
import types
from pathlib import Path

import numpy as np
import paddle

from tests.cinn.checkpoint_utils import strict_set_state_dict


def parse_args(kind, argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=[kind], default=kind)
    parser.add_argument("--mode", choices=["dynamic", "cinn"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapt", action="store_true")
    parser.add_argument("--compile-vision", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--warmup", type=int, default=50 if kind == "text" else 10)
    parser.add_argument("--iterations", type=int, default=200 if kind == "text" else 30)
    parser.add_argument("--resume", type=Path)
    options = parser.parse_args(argv)
    if options.kind == "text" and options.compile_vision:
        parser.error("--compile-vision only applies to Llama Guard4")
    if options.steps < 1 or options.warmup < 0 or options.iterations < 1:
        parser.error("steps/iterations must be positive and warmup must be nonnegative")
    if options.infer and (options.checkpoint is None or not options.checkpoint.is_file()):
        raise ValueError("Independent inference requires an existing checkpoint")
    return options


def run_inference(options, net, args):
    net.eval()
    durations = []
    with paddle.no_grad():
        for i in range(options.warmup + options.iterations):
            paddle.device.synchronize()
            start = time.perf_counter()
            out = net(*args)
            paddle.device.synchronize()
            if i >= options.warmup:
                durations.append(1000 * (time.perf_counter() - start))
        np.save(options.output / "logits.npy", out.numpy())
    (options.output / "inference.json").write_text(
        json.dumps(
            {
                "median_ms": float(np.median(durations)),
                "warmup": options.warmup,
                "iterations": options.iterations,
                "mean_ms": float(np.mean(durations)),
                "times_ms": durations,
                "scope": "independent checkpoint reload; fixed-input forward, not generation",
            },
            indent=2,
        )
    )
    print("INFERENCE_FINISHED", flush=True)
    return


def run_training(options, model, net, args, samples, source_commit, initial_hash):
    net.train()
    opt = paddle.optimizer.AdamW(learning_rate=2e-5, parameters=model.parameters())

    start_step = 0
    if options.resume:
        saved = paddle.load(str(options.resume))
        for key in ["kind", "mode", "adapt", "compile_vision", "real"]:
            if saved["arguments"].get(key) != getattr(options, key):
                raise ValueError(f"Resume mismatch: {key}")
        if saved["source_commit"] != source_commit or saved["initial_parameter_hash"] != initial_hash:
            raise ValueError("Resume source/initial weights mismatch")
        strict_set_state_dict(model, saved["model"])
        opt.set_state_dict(saved["optimizer"])
        start_step = saved["next_step"]
        if options.steps <= start_step:
            raise ValueError("--steps must exceed checkpoint step")
        paddle.set_rng_state(saved["cpu_rng"])
        paddle.set_cuda_rng_state(saved["cuda_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        (options.output / "resume.json").write_text(
            json.dumps(
                dict(checkpoint=str(options.resume), start_step=start_step, target_step=options.steps),
                indent=2,
            )
        )

    def save_training_state(step):
        temporary = options.output / "training_state.tmp"
        paddle.save(
            dict(
                model=model.state_dict(),
                optimizer=opt.state_dict(),
                next_step=step,
                cpu_rng=paddle.get_rng_state(),
                cuda_rng=paddle.get_cuda_rng_state(),
                numpy_rng=np.random.get_state(),
                python_rng=random.getstate(),
                arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()},
                source_commit=source_commit,
                initial_parameter_hash=initial_hash,
            ),
            str(temporary),
        )
        os.replace(temporary, options.output / "training_state.pdstate")

    def evaluate(step):
        net.eval()
        with paddle.no_grad():
            values = [float(paddle.nn.functional.cross_entropy(net(*s), y)) for s, y in samples[20:]]
        with (options.output / "validation.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, "loss": float(np.mean(values))}) + "\n")
        net.train()

    if samples:
        evaluate(start_step)
    with (options.output / "loss.jsonl").open("w") as f:
        for i in range(start_step, options.steps):
            if samples:
                args, labels = samples[i % 20]
            paddle.device.synchronize()
            t = time.perf_counter()
            out = net(*args)
            if i == start_step:
                np.save(options.output / "initial.npy", out.detach().numpy())
            loss = (
                paddle.nn.functional.cross_entropy(out, labels)
                if samples
                else paddle.nn.functional.cross_entropy(out[:, :-1, :], args[0][:, 1:])
            )
            loss_value = float(loss)
            if not np.isfinite(loss_value):
                raise FloatingPointError(f"Non-finite training loss at step {i}: {loss_value}")
            loss.backward()
            if i == start_step:
                np.savez(
                    options.output / "grads.npz",
                    **{n: v.grad.numpy() for n, v in model.named_parameters() if v.grad is not None},
                )
            opt.step()
            opt.clear_grad()
            paddle.device.synchronize()
            row = {"step": i, "loss": loss_value, "ms": 1000 * (time.perf_counter() - t)}
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(row, flush=True)
            if samples and (i + 1) % 100 == 0:
                evaluate(i + 1)
            if (i + 1) % 500 == 0:
                save_training_state(i + 1)
    save_training_state(options.steps)
    if samples:
        paddle.save(model.state_dict(), str(options.output / "final.pdparams"))
    print("FINISHED", flush=True)


def main(kind, argv=None):
    options = parse_args(kind, argv)
    # Loading model classes imports Fleet GPU operators. Keep CLI inspection
    # and the CPU runner unit tests independent of CUDA device discovery.
    from paddleformers.transformers import (
        Llama4Config,
        Llama4ForCausalLM,
        Llama4ForConditionalGeneration,
        Llama4TextConfig,
    )

    repo = str(Path(__file__).resolve().parents[2])
    options.output.mkdir(parents=True, exist_ok=False)
    paddle.set_device("gpu:0")
    paddle.seed(20260922)
    if options.kind == "text":
        c = Llama4TextConfig.from_pretrained(str(Path(os.environ["CINN_RESOURCE_DIR"]) / "assets"))
        c._attn_implementation = "eager"
        c.use_cache = False
        model = Llama4ForCausalLM(c)
    else:
        c = Llama4Config.from_pretrained(str(Path(__file__).parent / "llama_guard4" / "tiny_config.json"))
        c.text_config._attn_implementation = "eager"
        c.text_config.use_cache = False
        if options.real:
            c.text_config.vocab_size = 202048
            c.image_token_index = 200092
        else:
            c.text_config.vocab_size = 128
            c.image_token_index = 100
        model = Llama4ForConditionalGeneration(c)
    if options.checkpoint:
        if options.checkpoint.exists():
            strict_set_state_dict(model, paddle.load(str(options.checkpoint)))
        else:
            raise FileNotFoundError(options.checkpoint)
    c.save_pretrained(str(options.output / "config"))
    if options.adapt:

        @paddle.jit.not_to_static
        def config_get(self, key, default=None):
            return getattr(self, key, default)

        tc = c if options.kind == "text" else c.text_config
        tc.get = types.MethodType(config_get, tc)
    state = model.state_dict()

    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].numpy().tobytes())
    source_commit = (
        (Path(repo) / "SOURCE_COMMIT").read_text().strip()
        if (Path(repo) / "SOURCE_COMMIT").exists()
        else subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    )
    (options.output / "run.json").write_text(
        json.dumps(
            {
                "kind": options.kind,
                "mode": options.mode,
                "initial_parameter_hash": digest.hexdigest(),
                "parameters": sum(int(v.numel()) for v in model.parameters()),
                "source_commit": source_commit,
                "paddle": paddle.__version__,
                "paddle_commit": paddle.version.commit,
                "scope": "independent inference"
                if options.infer
                else ("real-data tiny validation" if options.real else "synthetic smoke"),
                "steps": 0 if options.infer else options.steps,
            },
            indent=2,
        )
    )
    ids = paddle.to_tensor([[1, 100, 2, 3, 4, 5, 6, 7]], dtype="int64")
    pixels = paddle.randn([1, 3, 28, 28])
    samples = []
    if options.real:
        from PIL import Image
        from transformers import AutoTokenizer

        token_path = str(Path(os.environ["CINN_RESOURCE_DIR"]) / "assets")
        tok = AutoTokenizer.from_pretrained(token_path, local_files_only=True)
        if options.kind == "text":
            data = [
                json.loads(s)
                for s in (Path(os.environ["CINN_RESOURCE_DIR"]) / "data/gsm8k_24.jsonl").read_text().splitlines()
            ][:24]
            texts = [v["question"] + "\n" + v["answer"] for v in data]
        else:
            data = np.load(str(Path(os.environ["CINN_RESOURCE_DIR"]) / "data/images24.npz"))
            texts = ["Image class ID: " + str(int(v)) + "." for v in data["labels"]]
        if len(texts) < 24:
            raise ValueError("Real-data runs require 20 training and 4 held-out samples")
        texts = texts[:24]
        for i, text in enumerate(texts):
            values = tok.encode(text, add_special_tokens=False)
            if options.kind == "guard":
                values = [c.image_token_index] + values
            values = values[:65]
            n = len(values)
            padded = values + [0] * (65 - n)
            x = paddle.to_tensor([padded[:-1]], dtype="int64")
            y = paddle.to_tensor([padded[1:n] + [-100] * (65 - n)], dtype="int64")
            if options.kind == "guard":
                im = np.asarray(Image.fromarray(data["images"][i]).resize((28, 28)), dtype="float32") / 127.5 - 1
                sample = [x, paddle.to_tensor(im.transpose(2, 0, 1)[None])]
            else:
                sample = [x]
            samples.append((sample, y))
        ids = samples[0][0][0]
        if options.kind == "guard":
            pixels = samples[0][0][1]
        (options.output / "experiment.json").write_text(
            json.dumps(
                {
                    "steps": options.steps,
                    "train_samples": 20,
                    "heldout_samples": 4,
                    "sequence_length": 64,
                    "tokenizer": token_path,
                    "data": "GSM8K first24" if options.kind == "text" else "ImageNet20train4val with class-ID text",
                    "image_preprocessing": "RGB resize28 PIL default, /127.5-1; fixed test protocol, not original high-res tiling"
                    if options.kind == "guard"
                    else None,
                    "precision": "FP32 TF32 off",
                    "checkpoint": str(options.checkpoint),
                    "adapt": options.adapt,
                    "compile_vision": options.compile_vision,
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

    net = TextNet() if options.kind == "text" else GuardNet()
    args = [ids] if options.kind == "text" else [ids, pixels]
    np.savez(options.output / "inputs.npz", **{str(i): v.numpy() for i, v in enumerate(args)})
    if options.mode == "cinn":
        if options.adapt and options.kind == "guard":
            if options.compile_vision:
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
    if options.infer:
        run_inference(options, net, args)
    else:
        run_training(options, model, net, args, samples, source_commit, digest.hexdigest())
