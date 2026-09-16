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
import torch
import transformers.utils as transformers_utils
import transformers.utils.import_utils as transformers_import_utils
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

if not hasattr(transformers_utils, "is_flash_attn_greater_or_equal_2_10"):
    transformers_utils.is_flash_attn_greater_or_equal_2_10 = lambda: False
if not hasattr(transformers_import_utils, "is_torch_fx_available"):
    transformers_import_utils.is_torch_fx_available = lambda: True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--topk", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.rope_scaling = None
    config._attn_implementation = "eager"
    model_class = get_class_from_dynamic_module("modeling_minicpm.MiniCPMForCausalLM", args.model)
    model_class._tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    model = model_class.from_pretrained(
        args.model,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.lm_head.weight = model.model.embed_tokens.weight
    model.to(args.device)
    model.eval()

    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(args.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(args.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        logits = outputs.logits.float()
        generated = input_ids
        for _ in range(args.max_new_tokens):
            generated_attention_mask = torch.ones_like(generated)
            next_outputs = model(
                input_ids=generated,
                attention_mask=generated_attention_mask,
                use_cache=False,
                return_dict=True,
            )
            next_token = next_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

    last_logits = logits[:, -1, :]
    top_values, top_indices = torch.topk(last_logits, k=args.topk, dim=-1)

    np.savez(
        output_dir / "reference.npz",
        input_ids=input_ids.cpu().numpy(),
        attention_mask=attention_mask.cpu().numpy() if attention_mask is not None else np.array([]),
        logits=logits.cpu().numpy(),
        top_indices=top_indices.cpu().numpy(),
        top_values=top_values.cpu().numpy(),
        generated=generated.cpu().numpy(),
    )
    metadata = {
        "model": args.model,
        "prompt": args.prompt,
        "dtype": args.dtype,
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "generated_text": tokenizer.decode(generated[0], skip_special_tokens=False),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
