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
import shutil
from pathlib import Path

import paddle

from paddleformers.transformers import MiniCPMConfig, MiniCPMForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", default="float32", choices=["float32"])
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer_dir = Path(args.tokenizer_dir) if args.tokenizer_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_safetensors = output_dir / "model.safetensors"
    if stale_safetensors.exists():
        stale_safetensors.unlink()

    paddle.seed(2026)
    config = MiniCPMConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        max_window_layers=2,
        use_sliding_window=False,
        scale_emb=12,
        dim_model_base=16,
        scale_depth=1.4,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        architectures=["MiniCPMForCausalLM"],
    )
    model = MiniCPMForCausalLM(config)
    config.save_pretrained(str(output_dir))
    paddle.save(model.state_dict(), str(output_dir / "model_state.pdparams"))

    tokenizer_files = [
        "tokenizer.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ]
    if tokenizer_dir is not None:
        for name in tokenizer_files:
            src = tokenizer_dir / name
            if src.exists():
                shutil.copy2(src, output_dir / name)

    if not (output_dir / "tokenizer.json").exists():
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
        vocab.update({chr(code): index for index, code in enumerate(range(32, 127), start=4)})
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.save(str(output_dir / "tokenizer.json"))

        tokenizer_config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": 512,
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
        }
        special_tokens_map = {
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
        }
        (output_dir / "tokenizer_config.json").write_text(
            json.dumps(tokenizer_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (output_dir / "special_tokens_map.json").write_text(
            json.dumps(special_tokens_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    print(f"Saved tiny random MiniCPM checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
