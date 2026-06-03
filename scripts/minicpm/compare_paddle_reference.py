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
from pathlib import Path

import numpy as np
import paddle
from transformers import AutoTokenizer

from paddleformers.transformers import (
    MiniCPMForCausalLMDeprecated as MiniCPMForCausalLM,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--load-checkpoint-format", default="flex_checkpoint")
    parser.add_argument("--convert-from-hf", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    paddle.set_device(args.device)

    reference = np.load(args.reference)
    input_ids_np = reference["input_ids"]
    attention_mask_np = reference["attention_mask"]
    hf_logits = reference["logits"]
    hf_generated = reference["generated"]

    attention_mask = None
    if attention_mask_np.size:
        attention_mask = paddle.to_tensor(attention_mask_np, dtype="int64")
    input_ids = paddle.to_tensor(input_ids_np, dtype="int64")

    if args.load_checkpoint_format == "direct":
        from paddleformers.transformers import MiniCPMConfig

        config = MiniCPMConfig.from_pretrained(args.model)
        model = MiniCPMForCausalLM(config)
        state_dict = paddle.load(str(Path(args.model) / "model_state.pdparams"))
        model.set_state_dict(state_dict)
    else:
        model = MiniCPMForCausalLM.from_pretrained(
            args.model,
            dtype=args.dtype,
            load_checkpoint_format=args.load_checkpoint_format,
            convert_from_hf=args.convert_from_hf,
        )
    model.eval()

    with paddle.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits.astype("float32").numpy()
        generated = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=args.max_new_tokens
        )[0]

    diff = np.abs(logits - hf_logits)
    prompt_len = input_ids_np.shape[1]
    hf_new_tokens = hf_generated[:, prompt_len : prompt_len + generated.shape[1]]
    first_10_match = np.array_equal(generated[:, : hf_new_tokens.shape[1]], hf_new_tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    result = {
        "max_diff": float(diff.max()),
        "mean_diff": float(diff.mean()),
        "last_max_diff": float(np.abs(logits[:, -1, :] - hf_logits[:, -1, :]).max()),
        "last_mean_diff": float(np.abs(logits[:, -1, :] - hf_logits[:, -1, :]).mean()),
        "first_10_tokens_match": bool(first_10_match),
        "paddle_generated_ids": generated.tolist(),
        "hf_generated_ids": hf_generated.tolist(),
        "hf_new_token_ids": hf_new_tokens.tolist(),
        "paddle_generated_text": tokenizer.decode(generated[0], skip_special_tokens=False),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
