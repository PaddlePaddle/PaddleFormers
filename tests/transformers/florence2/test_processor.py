# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import os
import tempfile
import unittest
from types import SimpleNamespace

from paddleformers.datasets.template.mm_plugin import get_mm_plugin
from paddleformers.datasets.template.template import (
    TEMPLATES,
    get_template_and_fix_tokenizer,
)
from paddleformers.transformers import Florence2Processor
from tests.testing_utils import require_package, slow


class _CharTokenizer:
    """Maps each character to its code point; enough to check where special tokens go."""

    bos_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        ids = [ord(char) for char in text]
        return [self.bos_token_id] + ids + [self.eos_token_id] if add_special_tokens else ids

    def convert_tokens_to_ids(self, token):
        return {"<s>": self.bos_token_id, "</s>": self.eos_token_id}[token]


class _VocabTokenizer:
    """Tokenizer stub with an explicit vocabulary; unknown tokens map to ``unk_token_id``."""

    unk_token_id = 3

    def __init__(self, vocab):
        self.vocab = vocab

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)


def _sft_row(template, tokenizer, messages):
    """Build (input_ids, labels) the way SFTDataset does: labels shifted left once."""
    prompt_ids, response_ids = template.encode_oneturn(tokenizer, messages)
    response_ids = response_ids + [tokenizer.convert_tokens_to_ids(template.suffix[-1])]
    tokens = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    return tokens, labels[1:] + [-100]


class Florence2ProcessorTest(unittest.TestCase):
    def setUp(self):
        self.processor = object.__new__(Florence2Processor)
        self.processor.task_prompts_without_inputs = {"<CAPTION>": "What does the image describe?"}
        self.processor.task_prompts_with_input = {"<REGION_TO_DESCRIPTION>": "What does the region {input} describe?"}

    def test_construct_prompts(self):
        prompts = self.processor._construct_prompts(["<CAPTION>", "<REGION_TO_DESCRIPTION><loc_1><loc_2>"])
        self.assertEqual(prompts[0], "What does the image describe?")
        self.assertIn("<loc_1><loc_2>", prompts[1])

    def test_post_process_detection(self):
        result = self.processor.post_process_generation("cat<loc_0><loc_1><loc_998><loc_999>", "<OD>", (100, 200))
        self.assertEqual(result["<OD>"]["labels"], ["cat"])
        self.assertEqual(len(result["<OD>"]["bboxes"][0]), 4)

    def test_post_process_open_vocabulary_detection_bbox(self):
        result = self.processor.post_process_generation(
            "cat<loc_0><loc_1><loc_998><loc_999>",
            "<OPEN_VOCABULARY_DETECTION>",
            (100, 200),
        )
        self.assertEqual(result["<OPEN_VOCABULARY_DETECTION>"]["labels"], ["cat"])
        self.assertEqual(len(result["<OPEN_VOCABULARY_DETECTION>"]["bboxes"][0]), 4)

    def test_post_process_open_vocabulary_detection_polygon(self):
        result = self.processor.post_process_generation(
            "cat<poly><loc_0><loc_1><loc_500><loc_501><loc_998><loc_999></poly>",
            "<OPEN_VOCABULARY_DETECTION>",
            (100, 200),
        )
        self.assertEqual(result["<OPEN_VOCABULARY_DETECTION>"]["labels"], ["cat"])
        self.assertIn("polygons", result["<OPEN_VOCABULARY_DETECTION>"])
        self.assertNotIn("bboxes", result["<OPEN_VOCABULARY_DETECTION>"])
        self.assertEqual(len(result["<OPEN_VOCABULARY_DETECTION>"]["polygons"][0][0]), 6)

    def test_post_process_segmentation(self):
        result = self.processor.post_process_generation(
            "cat<poly><loc_0><loc_1><loc_500><loc_501><loc_998><loc_999></poly>",
            "<REFERRING_EXPRESSION_SEGMENTATION>",
            (100, 200),
        )
        self.assertEqual(result["<REFERRING_EXPRESSION_SEGMENTATION>"]["labels"], ["cat"])
        self.assertEqual(len(result["<REFERRING_EXPRESSION_SEGMENTATION>"]["polygons"][0][0]), 6)

    def test_post_process_region_proposal(self):
        result = self.processor.post_process_generation(
            "<loc_0><loc_1><loc_998><loc_999>",
            "<REGION_PROPOSAL>",
            (100, 200),
        )
        self.assertEqual(len(result["<REGION_PROPOSAL>"]["bboxes"]), 1)

    def test_sft_message_format(self):
        self.processor.image_processor = object()
        plugin = get_mm_plugin("florence2", image_token="<image>")
        messages = [
            {"role": "user", "content": "<image><CAPTION>"},
            {"role": "assistant", "content": "A solid color image."},
        ]
        processed = plugin.process_messages(messages, ["image.jpg"], [], [], {}, self.processor)
        self.assertEqual(processed[0]["content"], "What does the image describe?")
        self.assertEqual(processed[1]["content"], "A solid color image.")
        with self.assertRaisesRegex(ValueError, "exactly one image"):
            plugin.process_messages([], [], [], [], {}, self.processor)
        with self.assertRaisesRegex(ValueError, "single-turn"):
            follow_up = [{"role": "user", "content": "<CAPTION>"}, {"role": "assistant", "content": "Gray."}]
            plugin.process_messages(messages + follow_up, ["image.jpg"], [], [], {}, self.processor)
        with self.assertRaisesRegex(ValueError, "Florence2Processor"):
            other_processor = SimpleNamespace(image_processor=object())
            plugin.process_messages(messages, ["image.jpg"], [], [], {}, other_processor)

    def test_validate_tokenizer_requires_task_tokens(self):
        plugin = get_mm_plugin("florence2", image_token="<image>")
        task_vocab = {token: 50265 + index for index, token in enumerate(Florence2Processor.task_tokens)}
        self.processor.tokenizer = _VocabTokenizer(task_vocab)

        plugin.validate_tokenizer(_VocabTokenizer(dict(task_vocab)), self.processor)
        # Plain BART tokenizer of the original checkpoints: task tokens are unknown and would be split.
        with self.assertRaisesRegex(ValueError, "missing 1024 Florence-2 task tokens"):
            plugin.validate_tokenizer(_VocabTokenizer({}), self.processor)
        with self.assertRaisesRegex(ValueError, "Florence2Processor"):
            plugin.validate_tokenizer(_VocabTokenizer({}), SimpleNamespace(tokenizer=_VocabTokenizer({})))

        dataset_config = {
            "tokenizer": _VocabTokenizer({}),
            "processor": self.processor,
            "template": "florence2",
            "tool_format": None,
            "default_system": None,
        }
        with self.assertRaisesRegex(ValueError, "save_pretrained"):
            get_template_and_fix_tokenizer(dataset_config)
        dataset_config["tokenizer"] = _VocabTokenizer(dict(task_vocab))
        self.assertIs(get_template_and_fix_tokenizer(dataset_config), TEMPLATES["florence2"])

    def test_sft_template_matches_processor_special_tokens(self):
        import paddle

        from paddleformers.transformers import (
            Florence2Config,
            Florence2ForConditionalGeneration,
        )

        tokenizer = _CharTokenizer()
        template = TEMPLATES["florence2"]
        messages = [{"role": "user", "content": "ab"}, {"role": "assistant", "content": "xy"}]
        prompt_ids, response_ids = template.encode_oneturn(tokenizer, messages)
        # Same layout as Florence2Processor (encoder) and HF labels (decoder): <s> text </s>.
        self.assertEqual(prompt_ids, tokenizer.encode("ab", add_special_tokens=True))
        self.assertEqual(response_ids, [0, ord("x"), ord("y")])
        self.assertEqual(template.suffix, ["</s>"])

        tokens, labels = _sft_row(template, tokenizer, messages)
        model = object.__new__(Florence2ForConditionalGeneration)
        model.config = Florence2Config(pad_token_id=1, bos_token_id=0, eos_token_id=2)
        input_ids, labels = paddle.to_tensor([tokens]), paddle.to_tensor([labels])
        self.assertTrue(model._is_sft_inputs(input_ids, labels))
        source_ids, decoder_labels, _ = model._split_sft_inputs(input_ids, labels, None)
        self.assertEqual(source_ids.tolist(), [tokenizer.encode("ab", add_special_tokens=True)])
        self.assertEqual(decoder_labels.tolist(), [tokenizer.encode("xy", add_special_tokens=True)])

    @slow
    @require_package("PIL")
    def test_sft_template_matches_pretrained_processor(self):
        from PIL import Image

        from paddleformers.transformers import AutoProcessor

        model_name_or_path = os.environ.get("FLORENCE2_PRETRAINED_MODEL", "microsoft/Florence-2-base")
        processor = AutoProcessor.from_pretrained(model_name_or_path, download_hub="huggingface")
        tokenizer = processor.tokenizer
        template = TEMPLATES["florence2"]
        image = Image.new("RGB", (32, 32), color=(64, 128, 192))
        messages = [{"role": "user", "content": "<image><CAPTION>"}, {"role": "assistant", "content": "A cat."}]
        messages = template.mm_plugin.process_messages(messages, [image], [], [], {}, processor)
        prompt_ids, response_ids = template.encode_oneturn(tokenizer, messages)
        suffix_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(template.suffix[-1]))

        self.assertEqual(
            prompt_ids, processor(text="<CAPTION>", images=image, return_tensors="np")["input_ids"][0].tolist()
        )
        self.assertEqual(response_ids + suffix_ids, tokenizer("A cat.")["input_ids"])

    @slow
    def test_saved_processor_tokenizer_passes_validation(self):
        from paddleformers.transformers import AutoProcessor, AutoTokenizer

        model_name_or_path = os.environ.get("FLORENCE2_PRETRAINED_MODEL", "microsoft/Florence-2-base")
        processor = AutoProcessor.from_pretrained(model_name_or_path, download_hub="huggingface")
        plugin = TEMPLATES["florence2"].mm_plugin
        with tempfile.TemporaryDirectory() as tmpdirname:
            processor.tokenizer.save_pretrained(tmpdirname)
            tokenizer = AutoTokenizer.from_pretrained(tmpdirname)

        plugin.validate_tokenizer(tokenizer, processor)
        self.assertEqual(len(tokenizer.encode("<loc_12>", add_special_tokens=False)), 1)


if __name__ == "__main__":
    unittest.main()
