# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

from paddleformers.datasets.template.template import TEMPLATES


def test_registered_llama4_plugin_accepts_masked_tokens():
    plugin = TEMPLATES["llama4_vl"].mm_plugin
    assert "<|image|>" in plugin.masked_tokens
    assert "<|image_end|>" in plugin.masked_tokens


def test_text_template_does_not_require_image_processor():
    messages = [{"role": "user", "content": "hello"}]
    assert TEMPLATES["llama4"].mm_plugin.process_messages(messages, [], [], [], {}, None) == messages


def test_image_plugin_preserves_original_resolution():
    from types import SimpleNamespace

    from PIL import Image

    seen = []

    def image_processor(images, return_tensors):
        seen.extend(image.size for image in images)
        return {"aspect_ratios": [(4, 3)]}

    plugin = TEMPLATES["llama4_vl"].mm_plugin
    result = plugin._get_mm_inputs(
        [Image.new("RGB", (1200, 900))], [], [], SimpleNamespace(image_processor=image_processor)
    )
    assert seen == [(1200, 900)]
    assert result["aspect_ratios"] == [(4, 3)]


def test_guard4_lora_registered():
    from types import SimpleNamespace

    from paddleformers.cli.utils.llm_utils import get_lora_target_modules

    targets = get_lora_target_modules(SimpleNamespace(config=SimpleNamespace(model_type="llama4")))
    assert ".*q_proj.*" in targets
    assert ".*down_proj.*" in targets
