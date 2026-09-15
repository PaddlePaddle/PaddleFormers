# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

from paddleformers.datasets.template.template import TEMPLATES


def test_registered_llama4_plugin_accepts_masked_tokens():
    plugin = TEMPLATES["llama4"].mm_plugin
    assert "<|image|>" in plugin.masked_tokens
    assert "<|image_end|>" in plugin.masked_tokens
