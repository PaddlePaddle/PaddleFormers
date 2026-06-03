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

import argparse
import json
import time
from pathlib import Path

import numpy as np
import paddle
from paddle import nn

from paddleformers.transformers import (
    MiniCPMForCausalLMDeprecated as MiniCPMForCausalLM,
)


class ForwardWrapper(nn.Layer):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.model
        self.lm_head = model.lm_head
        self.logit_scale = model.config.hidden_size / model.config.dim_model_base

    def forward(self, input_ids, attention_mask, position_ids):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=False,
        )
        hidden_states = outputs[0] / self.logit_scale
        logits = paddle.matmul(hidden_states, self.lm_head.weight, transpose_y=True)
        if self.lm_head.bias is not None:
            logits += self.lm_head.bias
        return logits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Paddle native MiniCPM checkpoint directory.")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--load-checkpoint-format", default="sharding_io")
    parser.add_argument("--backend", default="CINN")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--full-graph", action="store_true", help="Enable full graph conversion for to_static.")
    parser.add_argument(
        "--disable-fused-rms-norm",
        action="store_true",
        help="Disable fused RMSNorm in both dynamic and static runs.",
    )
    return parser.parse_args()


def disable_fused_rms_norm(layer):
    for sublayer in layer.sublayers():
        if hasattr(sublayer, "fuse_rms_norm"):
            sublayer.fuse_rms_norm = False


def synchronize():
    if paddle.device.is_compiled_with_cuda() and str(paddle.get_device()).startswith("gpu"):
        paddle.device.synchronize()


def time_forward(layer, input_ids, attention_mask, position_ids, warmup, steps):
    with paddle.no_grad():
        for _ in range(warmup):
            _ = layer(input_ids, attention_mask, position_ids)
        synchronize()

        start = time.perf_counter()
        for _ in range(steps):
            output = layer(input_ids, attention_mask, position_ids)
        synchronize()
        elapsed = time.perf_counter() - start

    avg_latency = elapsed / steps
    tokens_per_step = 1
    for dim in input_ids.shape:
        tokens_per_step *= dim
    tokens_per_second = tokens_per_step * steps / elapsed
    return output, avg_latency, tokens_per_second


def main():
    args = parse_args()
    paddle.set_device(args.device)
    paddle.seed(args.seed)
    np.random.seed(args.seed)

    model = MiniCPMForCausalLM.from_pretrained(
        args.model,
        dtype=args.dtype,
        load_checkpoint_format=args.load_checkpoint_format,
        convert_from_hf=False,
    )
    if args.disable_fused_rms_norm:
        disable_fused_rms_norm(model)
    model.eval()

    vocab_size = int(model.config.vocab_size)
    input_ids_np = np.random.randint(0, vocab_size, size=(args.batch_size, args.seq_len), dtype="int64")
    attention_mask_np = np.ones((args.batch_size, args.seq_len), dtype="int64")
    position_ids_np = np.broadcast_to(np.arange(args.seq_len, dtype="int64"), (args.batch_size, args.seq_len)).copy()

    input_ids = paddle.to_tensor(input_ids_np, dtype="int64")
    attention_mask = paddle.to_tensor(attention_mask_np, dtype="int64")
    position_ids = paddle.to_tensor(position_ids_np, dtype="int64")

    dynamic_layer = ForwardWrapper(model)
    dynamic_output, dynamic_latency, dynamic_tps = time_forward(
        dynamic_layer, input_ids, attention_mask, position_ids, args.warmup, args.steps
    )

    static_layer = paddle.jit.to_static(
        ForwardWrapper(model),
        input_spec=[
            paddle.static.InputSpec(shape=[args.batch_size, args.seq_len], dtype="int64", name="input_ids"),
            paddle.static.InputSpec(shape=[args.batch_size, args.seq_len], dtype="int64", name="attention_mask"),
            paddle.static.InputSpec(shape=[args.batch_size, args.seq_len], dtype="int64", name="position_ids"),
        ],
        backend=args.backend,
        full_graph=args.full_graph,
    )
    static_layer.eval()
    static_output, static_latency, static_tps = time_forward(
        static_layer, input_ids, attention_mask, position_ids, args.warmup, args.steps
    )

    diff = paddle.abs(dynamic_output.astype("float32") - static_output.astype("float32"))
    speedup_percent = (dynamic_latency / static_latency - 1.0) * 100.0
    result = {
        "model": args.model,
        "dtype": args.dtype,
        "device": args.device,
        "backend": args.backend,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "warmup": args.warmup,
        "steps": args.steps,
        "full_graph": args.full_graph,
        "disable_fused_rms_norm": args.disable_fused_rms_norm,
        "dynamic_avg_latency_sec": dynamic_latency,
        "static_avg_latency_sec": static_latency,
        "dynamic_tokens_per_second": dynamic_tps,
        "static_tokens_per_second": static_tps,
        "speedup_percent": speedup_percent,
        "max_diff": float(diff.max().numpy()),
        "mean_diff": float(diff.mean().numpy()),
    }

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
