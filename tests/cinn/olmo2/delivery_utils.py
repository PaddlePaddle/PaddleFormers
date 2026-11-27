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

"""Frozen depth-only OLMo2 experiment contract; no device initialization."""
import hashlib
import json
from pathlib import Path

import numpy as np

SEED = 20261116
PARAMETERS = 545277952
INITIAL_HASH = "04c05250a425e200995ea25cac1ecaed8feb6abc85e03d562fff041a3dc59f82"
INPUT_HASH = "67cc80e06ef508535801734248f04c87d7144f1536bb8f504d0153b646bf23bf"
CONTRACT = dict(
    hidden_size=2048,
    intermediate_size=8192,
    num_hidden_layers=2,
    num_attention_heads=16,
    num_key_value_heads=16,
    vocab_size=100352,
    max_position_embeddings=4096,
    tie_word_embeddings=False,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validate_config(raw):
    for key, expected in CONTRACT.items():
        if raw.get(key) != expected:
            raise ValueError(f"Depth-only config mismatch: {key}={raw.get(key)!r}, expected {expected!r}")


def make_inputs(tokenizer, data):
    rows = [json.loads(line) for line in Path(data).read_text().splitlines()][:20]
    if len(rows) != 20:
        raise ValueError("Expected 20 training examples")
    tokens, labels = [], []
    for row in rows:
        ids = tokenizer.encode(row["question"] + "\n" + row["answer"], add_special_tokens=False)[:129]
        count = len(ids)
        ids += [tokenizer.eos_token_id] * (129 - count)
        lab = ids[1:]
        lab[max(count - 1, 0) :] = [-100] * (128 - max(count - 1, 0))
        tokens.append(ids[:-1])
        labels.append(lab)
    digest = hashlib.sha256(np.array([tokens, labels], dtype=np.int64).tobytes()).hexdigest()
    if digest != INPUT_HASH:
        raise ValueError(f"Tokenized inputs differ from the frozen benchmark: {digest}")
    return tokens, labels, digest
