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
from transformers import PreTrainedTokenizerFast

from paddleformers.datasets.template.mm_plugin import get_mm_plugin
from paddleformers.transformers import AutoProcessor
from paddleformers.transformers.deepseek_ocr2 import (
    DeepseekOcr2Config,
    DeepseekOcr2ImageProcessor,
    DeepseekOcr2Processor,
)


def processor():
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "hello": 2, "<image>": 7}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", additional_special_tokens=["<image>"]
    )
    return DeepseekOcr2Processor(
        DeepseekOcr2ImageProcessor(size={"height": 32, "width": 32}, tile_size=24), tokenizer, patch_size=4
    )


def test_preprocess_global_local_and_native_sft():
    proc = processor()
    image = Image.fromarray(np.arange(48 * 24 * 3, dtype=np.uint8).reshape(24, 48, 3))
    output = proc(images=image, text="hello <image>", return_tensors="pd")
    assert output["pixel_values"].shape == [1, 3, 32, 32]
    assert output["pixel_values_local"].shape == [2, 3, 24, 24]
    assert output["num_local_patches"].tolist() == [2]
    assert output["input_ids"].tolist()[0].count(7) == 433
    plugin = get_mm_plugin("deepseek_ocr2_native", image_token="<image>")
    mm = plugin.get_mm_inputs([image], [], [], proc)
    messages = [{"role": "user", "content": "hello <image>"}]
    expanded = plugin.process_messages(messages, [image], [], [], mm, proc)
    assert expanded[0]["content"].count("<image>") == 433
    assert messages[0]["content"].count("<image>") == 1
    assert plugin.process_tokens(output["input_ids"][0].tolist(), proc).count(-100) == 433
    np.testing.assert_array_equal(mm["pixel_values"].numpy(), output["pixel_values"].numpy())


def test_processor_validates_placeholders_and_truncation():
    proc = processor()
    image = Image.new("RGB", (16, 16))
    with pytest.raises(ValueError, match="requires images"):
        proc(text="hello <image>")
    with pytest.raises(ValueError, match="each image placeholder"):
        proc(images=[image, image], text="<image>")
    with pytest.raises(ValueError, match="truncation"):
        proc(images=image, text="hello <image>", truncation=True, max_length=2)
    no_crop = proc(images=image, text="<image>", crop_to_patches=False, return_tensors="pd")
    assert no_crop["input_ids"].tolist()[0].count(7) == 145
    assert "pixel_values_local" not in no_crop


def test_auto_processor_roundtrip():
    proc = processor()
    with tempfile.TemporaryDirectory() as directory:
        proc.save_pretrained(directory)
        DeepseekOcr2Config().save_pretrained(directory)
        actual = AutoProcessor.from_pretrained(directory)
    assert isinstance(actual, DeepseekOcr2Processor)
    assert actual.patch_size == 4
    image = Image.new("RGB", (16, 16))
    expected = proc(images=image, text="<image>", return_tensors="pd")
    output = actual(images=image, text="<image>", return_tensors="pd")
    for key in expected:
        np.testing.assert_array_equal(output[key].numpy(), expected[key].numpy())
