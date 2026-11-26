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

import tempfile

import numpy as np
import pytest
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import CLIPImageProcessor, LlavaProcessor, PreTrainedTokenizerFast

from paddleformers.datasets.template.mm_plugin import get_mm_plugin
from paddleformers.transformers import AutoProcessor
from paddleformers.transformers.fast_vlm import (
    FastVlmConfig,
    FastVlmImageProcessor,
    FastVlmProcessor,
)


def components():
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3, "<image>": 7}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", additional_special_tokens=["<image>"]
    )
    image_processor = FastVlmImageProcessor(image_size=128)
    return tokenizer, image_processor


def test_native_processor_matches_hf_llava_and_masks_labels():
    tokenizer, image_processor = components()
    processor = FastVlmProcessor(image_processor, tokenizer)
    reference = LlavaProcessor(
        CLIPImageProcessor(
            size={"shortest_edge": 128},
            crop_size={"height": 128, "width": 128},
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0],
        ),
        tokenizer,
        patch_size=64,
        vision_feature_select_strategy="full",
    )
    image = Image.fromarray(np.arange(96 * 160 * 3, dtype=np.uint8).reshape(96, 160, 3))
    prompts = ["hello <image> world", "<image> hello"]
    actual = processor(images=[image, image], text=prompts, padding=True, return_tensors="pd")
    expected = reference(images=[image, image], text=prompts, padding=True, return_tensors="np")
    for key in ("input_ids", "attention_mask", "pixel_values"):
        np.testing.assert_allclose(actual[key].numpy(), expected[key], atol=0, rtol=0)
    plugin = get_mm_plugin("fast_vlm", image_token="<image>")
    tokens = actual["input_ids"][0].tolist()
    labels = plugin.process_tokens(tokens, processor)
    assert labels.count(-100) == 4
    messages = [{"role": "user", "content": "hello <image>"}]
    expanded = plugin.process_messages(
        messages, [image], [], [], {"pixel_values": actual["pixel_values"][:1]}, processor
    )
    assert expanded[0]["content"].count("<image>") == 4
    assert messages[0]["content"].count("<image>") == 1


def test_processor_validates_missing_images_and_truncation():
    tokenizer, image_processor = components()
    processor = FastVlmProcessor(image_processor, tokenizer)
    image = Image.new("RGB", (128, 128))
    with pytest.raises(ValueError, match="require images"):
        processor(text="hello <image>")
    with pytest.raises(ValueError, match="each image placeholder"):
        processor(images=[image, image], text="<image>")
    with pytest.raises(ValueError, match="truncation"):
        processor(images=image, text="hello <image>", truncation=True, max_length=2)


def test_auto_processor_loads_native_hf_processor_name():
    tokenizer, image_processor = components()
    reference = LlavaProcessor(
        CLIPImageProcessor(
            size={"shortest_edge": 128},
            crop_size={"height": 128, "width": 128},
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0],
        ),
        tokenizer,
        patch_size=64,
        vision_feature_select_strategy="full",
    )
    with tempfile.TemporaryDirectory() as directory:
        reference.save_pretrained(directory)
        FastVlmConfig().save_pretrained(directory)
        actual = AutoProcessor.from_pretrained(directory)
        assert isinstance(actual, FastVlmProcessor)
        batch = actual(images=Image.new("RGB", (128, 128)), text="hello <image>", return_tensors="pd")
        assert batch["input_ids"].tolist()[0].count(7) == 4
