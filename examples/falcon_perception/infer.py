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

"""Run native Paddle single-image segmentation and optionally export alignment tensors."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ["FLAGS_allow_tf32_cublas"] = "0"
os.environ["FLAGS_allow_tf32_cudnn"] = "0"
import numpy as np
import paddle
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from trace_utils import InferenceTrace

from paddleformers.transformers.falcon_perception import FalconPerceptionForSegmentation


def save_outputs(image, query, predictions, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pixels = np.asarray(image).astype(np.float32).copy()
    colors = [(255, 70, 70), (40, 180, 255), (80, 220, 100), (220, 120, 255), (255, 190, 50)]
    for index, prediction in enumerate(predictions):
        rle = prediction["mask_rle"]
        mask = mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode()}).astype(bool)
        Image.fromarray(mask.astype(np.uint8) * 255).save(directory / f"mask_{index:03d}.png")
        pixels[mask] = 0.55 * pixels[mask] + 0.45 * np.asarray(colors[index % len(colors)])
    overlay = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(overlay)
    for index, prediction in enumerate(predictions):
        x, y = prediction["xy"]["x"] * image.width, prediction["xy"]["y"] * image.height
        w, h = prediction["hw"]["w"] * image.width, prediction["hw"]["h"] * image.height
        color = colors[index % len(colors)]
        draw.rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], outline=color, width=3)
        draw.text(
            (max(0, x - w / 2), max(0, y - h / 2 - 14)),
            f"{query} #{index + 1}",
            fill=color,
            stroke_width=1,
            stroke_fill="black",
        )
    overlay.save(directory / "overlay.png")
    (directory / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--query", default="cat")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-dimension", type=int, default=1024)
    parser.add_argument("--min-dimension", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="float32 is validated; bfloat16 is experimental and has not passed mask alignment.",
    )
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    paddle.set_device(args.device)
    paddle.seed(42)
    image = Image.open(args.image).convert("RGB")
    start = time.perf_counter()
    model = FalconPerceptionForSegmentation.from_pretrained(args.model, dtype=args.dtype)
    load_seconds = time.perf_counter() - start
    print("Loaded Paddle model:", model.weight_loading_info, flush=True)
    trace = InferenceTrace(args.output)
    if args.trace:
        trace.install(model)
    start = time.perf_counter()
    with paddle.no_grad():
        predictions = model.generate(
            image,
            args.query,
            min_dimension=args.min_dimension,
            max_dimension=args.max_dimension,
            max_new_tokens=args.max_new_tokens,
        )[0]
    if args.device.startswith("gpu"):
        paddle.device.synchronize()
    elapsed = time.perf_counter() - start
    trace.finish(
        dict(
            framework="paddle",
            version=paddle.__version__,
            dtype=args.dtype,
            image=str(Path(args.image).resolve()),
            query=args.query,
            min_dimension=args.min_dimension,
            max_dimension=args.max_dimension,
            max_new_tokens=args.max_new_tokens,
            inference_seconds=elapsed,
            load_seconds=load_seconds,
            gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
            weight_loading=model.weight_loading_info,
        ),
        predictions,
    )
    save_outputs(image, args.query, predictions, args.output)
    print(f"Paddle: {len(predictions)} instances in {elapsed:.2f}s; outputs={args.output}", flush=True)


if __name__ == "__main__":
    main()
