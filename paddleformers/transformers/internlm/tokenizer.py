# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for InternLM."""

import json
from pathlib import Path
from typing import Optional

from ..intern_lm2_5.tokenizer import (
    VOCAB_FILES_NAMES,
    InternLMSentencePieceTokenizerMixin,
)
from ..tokenizer_utils import PreTrainedTokenizer


class InternLMTokenizer(InternLMSentencePieceTokenizerMixin, PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = {}
    model_input_names = ["input_ids", "attention_mask"]

    def _save_pretrained(self, save_directory, file_names, legacy_format=None, filename_prefix=None):
        saved_files = super()._save_pretrained(
            save_directory, file_names, legacy_format=legacy_format, filename_prefix=filename_prefix
        )
        # Transformers 5 omits these flags, but this Python tokenizer still uses them.
        config_path = Path(file_names[0])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.update(add_bos_token=self.add_bos_token, add_eos_token=self.add_eos_token)
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return saved_files

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.sp_model.bos_id()

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.sp_model.eos_id()
