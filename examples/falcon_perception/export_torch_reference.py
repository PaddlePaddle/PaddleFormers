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

"""Export the original model's tensors without importing Paddle/PaddleFormers."""
import argparse
import os
import time
from pathlib import Path

import torch
from PIL import Image
from trace_utils import InferenceTrace
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--query", default="cat")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-dimension", type=int, default=1024)
    parser.add_argument("--min-dimension", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    image = Image.open(args.image).convert("RGB")
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        dtype=getattr(torch, args.dtype),
        device_map={"": "cuda:0"},
    ).eval()
    # RTX 3090 needs smaller FP32 FlexAttention tiles than the default autotuned
    # config; changing tile sizes leaves the source computation unchanged.
    import importlib
    from functools import partial

    module = importlib.import_module(type(model).__module__)
    module.compiled_flex_attn_prefill = partial(
        module.compiled_flex_attn_prefill, kernel_options={"BLOCK_M": 32, "BLOCK_N": 32}
    )
    module.compiled_flex_attn_decode = partial(
        module.compiled_flex_attn_decode, kernel_options={"BLOCK_M": 32, "BLOCK_N": 32}
    )
    print("Loaded original PyTorch model", flush=True)
    trace = InferenceTrace(args.output)
    if args.trace:
        trace.install(model)
    start = time.perf_counter()
    predictions = model.generate(
        image,
        args.query,
        min_dimension=args.min_dimension,
        max_dimension=args.max_dimension,
        max_new_tokens=args.max_new_tokens,
        compile=False,
    )[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    trace.finish(
        dict(
            framework="torch",
            version=torch.__version__,
            dtype=args.dtype,
            image=str(Path(args.image).resolve()),
            query=args.query,
            min_dimension=args.min_dimension,
            max_dimension=args.max_dimension,
            max_new_tokens=args.max_new_tokens,
            inference_seconds=elapsed,
            gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        ),
        predictions,
    )
    print(f"Torch: {len(predictions)} instances in {elapsed:.2f}s; outputs={args.output}", flush=True)


if __name__ == "__main__":
    main()
