import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from paddleformers.transformers import AutoProcessor, PaliGemmaProcessor


CHECKPOINT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..", "paligemma2-3b-pt-448"))


class DummyImageProcessor:
    image_seq_length = 1024
    model_input_names = ["pixel_values"]

    def __call__(self, images, **kwargs):
        return {"pixel_values": np.zeros((1, 3, 64, 64), dtype=np.float32)}


class DummyTokenizer:
    unk_token_id = 0
    bos_token = "<bos>"
    eos_token = "<eos>"
    init_kwargs = {}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self):
        self.add_bos_token = True
        self.add_eos_token = True

    def convert_tokens_to_ids(self, token):
        return 1 if token == "<image>" else self.unk_token_id

    def add_special_tokens(self, special_tokens):
        pass

    def add_tokens(self, tokens):
        pass

    def __call__(self, text, text_pair=None, return_token_type_ids=False, **kwargs):
        sequences = []
        for prompt in text:
            image_tokens = prompt.count("<image>")
            sequences.append([1] * image_tokens + [2] * (len(prompt.replace("<image>", "")) + 1))
        max_length = max(map(len, sequences))
        input_ids = np.zeros((len(sequences), max_length), dtype=np.int64)
        attention_mask = np.zeros_like(input_ids)
        for index, sequence in enumerate(sequences):
            input_ids[index, : len(sequence)] = sequence
            attention_mask[index, : len(sequence)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": np.zeros_like(input_ids),
        }


class PaliGemmaProcessorTest(unittest.TestCase):
    def setUp(self):
        with patch.object(PaliGemmaProcessor, "check_argument_for_proper_class"):
            self.processor = PaliGemmaProcessor(DummyImageProcessor(), DummyTokenizer())
        self.image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))

    def test_processor_outputs_expected_multimodal_inputs(self):
        inputs = self.processor(images=self.image, text="describe", return_tensors="np")

        self.assertEqual(set(inputs.keys()), {"input_ids", "attention_mask", "token_type_ids", "pixel_values"})
        self.assertEqual(int((inputs["input_ids"] == self.processor.image_token_id).sum()), 1024)
        self.assertEqual(inputs["pixel_values"].shape, (1, 3, 64, 64))
        self.assertEqual(inputs["input_ids"].shape, inputs["attention_mask"].shape)
        self.assertTrue(np.all(inputs["token_type_ids"] == 0))

    def test_processor_rejects_nested_multi_image_input(self):
        with self.assertRaisesRegex(ValueError, "exactly one image per prompt"):
            self.processor(images=[[self.image, self.image]], text="<image> describe", return_tensors="np")

    def test_auto_processor_resolves_paligemma_processor_from_preprocessor_config(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "preprocessor_config.json"), "w", encoding="utf-8") as config_file:
                json.dump({"processor_class": "PaliGemmaProcessor"}, config_file)

            with patch.object(PaliGemmaProcessor, "from_pretrained", return_value=self.processor) as from_pretrained:
                processor = AutoProcessor.from_pretrained(tempdir, local_files_only=True)

        self.assertIs(processor, self.processor)
        from_pretrained.assert_called_once_with(
            tempdir, trust_remote_code=None, download_hub="huggingface", _from_auto=True, local_files_only=True
        )

    @unittest.skipUnless(os.path.isdir(CHECKPOINT), "requires local paligemma2-3b-pt-448 checkpoint")
    def test_processor_real_checkpoint(self):
        processor = PaliGemmaProcessor.from_pretrained(CHECKPOINT)
        inputs = processor(images=self.image, text="describe", return_tensors="np")

        self.assertEqual(set(inputs.keys()), {"input_ids", "attention_mask", "token_type_ids", "pixel_values"})
        self.assertEqual(int((inputs["input_ids"] == processor.image_token_id).sum()), 1024)
        self.assertEqual(inputs["pixel_values"].shape, (1, 3, 448, 448))
        self.assertEqual(inputs["input_ids"].shape, inputs["attention_mask"].shape)
        self.assertTrue(np.all(inputs["token_type_ids"] == 0))


if __name__ == "__main__":
    unittest.main()
