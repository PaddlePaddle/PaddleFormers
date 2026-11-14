# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Local image/token preprocessing checks; no pretrained tokenizer downloads."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import paddle
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from paddleformers.transformers import FalconPerceptionConfig
from paddleformers.transformers.falcon_perception.processing import (
    ImageProcessor,
    get_pos_thw,
    load_image,
    process_batch,
    resize_image_if_necessary,
    smart_resize,
    tokenize_inputs,
)


def make_tokenizer():
    vocab = {f"token_{i}": i for i in range(300)}
    specials = {"<|pad|>": 0, "<|bos|>": 1, "<|unk|>": 2, "<|image|>": 227, "<|end_of_query|>": 263}
    for text, index in {**specials, "cat": 20, "dog": 21}.items():
        del vocab[f"token_{index}"]
        vocab[text] = index
    backend = Tokenizer(WordLevel(vocab, unk_token="<|unk|>"))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.post_processor = TemplateProcessing(single="<|bos|> $A", special_tokens=[("<|bos|>", 1)])
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<|bos|>",
        pad_token="<|pad|>",
        unk_token="<|unk|>",
        additional_special_tokens=["<|image|>"],
    )


class FalconPerceptionProcessingTest(unittest.TestCase):
    def setUp(self):
        previous = paddle.get_device()
        self.addCleanup(paddle.set_device, previous)
        paddle.set_device("cpu")
        self.config = FalconPerceptionConfig()
        self.tokenizer = make_tokenizer()
        self.processor = ImageProcessor(patch_size=16, merge_size=1, do_resize=False)

    def test_load_local_images(self):
        array = np.arange(8 * 12 * 3, dtype="uint8").reshape([8, 12, 3])
        image = Image.fromarray(array)
        self.assertIs(load_image(image), image)
        np.testing.assert_array_equal(np.asarray(load_image(array)), array)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            image.save(path)
            np.testing.assert_array_equal(np.asarray(load_image(str(path))), array)
        self.assertIsNone(load_image(None))
        with self.assertRaises(TypeError):
            load_image(123)

    def test_resize_keeps_aspect_ratio_and_bounds(self):
        for size, expected in (((40, 80), (64, 128)), ((160, 80), (128, 64)), ((96, 64), (96, 64))):
            with self.subTest(size=size):
                image = Image.new("RGB", size)
                self.assertEqual(resize_image_if_necessary(image, 64, 128).size, expected)

    def test_smart_resize_patch_alignment_and_invalid_shapes(self):
        resized = smart_resize(
            np.zeros([63, 97, 3], dtype="uint8"),
            factor=16,
            resample=Image.Resampling.BICUBIC,
            input_data_format="channels_last",
            min_pixels=32 * 32,
            max_pixels=128 * 128,
        )
        self.assertEqual(resized.shape, (64, 96, 3))
        for shape in ((8, 32, 3), (16, 4096, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                smart_resize(
                    np.zeros(shape, dtype="uint8"),
                    factor=16,
                    resample=Image.Resampling.BICUBIC,
                    input_data_format="channels_last",
                )

    def test_image_normalization_and_rgb_conversion(self):
        array = np.zeros([8, 12, 3], dtype="uint8")
        array[:, :, 1] = 255
        array[:, :, 2] = 128
        actual = self.processor.preprocess([Image.fromarray(array)])[0]
        self.assertEqual(actual.shape, (1, 8, 12, 3))
        np.testing.assert_allclose(actual[0], array.astype("float32") / 127.5 - 1, atol=1e-6)
        grayscale = self.processor.preprocess([Image.new("L", (12, 8), color=255)])[0]
        self.assertEqual(grayscale.shape, (1, 8, 12, 3))
        np.testing.assert_allclose(grayscale, 1.0, atol=1e-6)

    def test_empty_and_invalid_image_inputs(self):
        self.assertEqual(self.processor.preprocess(None), [])
        self.assertEqual(self.processor.preprocess([None]), [])
        self.assertIsNone(self.processor.batch_images_with_mask([], 16, 16))
        with self.assertRaises(ValueError):
            self.processor.preprocess([123])
        with self.assertRaisesRegex(ValueError, "3-channel"):
            self.processor.batch_images_with_mask([np.ones([1, 4, 4, 1], "float32")], 8, 8)

    def test_image_batch_padding_and_masks(self):
        arrays = [np.ones([1, 4, 6, 3], "float32"), np.full([1, 6, 4, 3], 2, "float32")]
        output = self.processor.batch_images_with_mask(arrays, 8, 8)
        pixels, masks = output["pixel_values"].numpy(), output["padding_mask"].numpy()
        self.assertEqual(pixels.shape, (2, 1, 8, 8, 3))
        self.assertEqual(masks.shape, (2, 1, 8, 8))
        for index, array in enumerate(arrays):
            height, width = array.shape[1:3]
            expected = np.zeros([1, 8, 8, 3], dtype="float32")
            expected[:, :height, :width] = array
            np.testing.assert_array_equal(pixels[index], expected)
            expected_mask = np.zeros([1, 8, 8], dtype="int64")
            expected_mask[:, :height, :width] = 1
            np.testing.assert_array_equal(masks[index], expected_mask)

    def test_image_token_expansion_and_bos(self):
        image = np.zeros([1, 32, 48, 3], dtype="float32")
        ids, selected = tokenize_inputs("cat <|image|> dog", [image], self.tokenizer, self.config, 16, 1, 128)
        expected = [1, 20, 244, 245, 246, 247, 248, *([227] * 6), 230, 21]
        np.testing.assert_array_equal(ids.numpy(), expected)
        self.assertEqual(len(selected), 1)
        self.assertIs(selected[0], image)
        self.assertEqual(int((ids == 1).sum()), 1)

    def test_image_blocks_are_never_partially_truncated(self):
        image = np.zeros([1, 32, 48, 3], dtype="float32")
        ids, selected = tokenize_inputs("cat <|image|> dog", [image], self.tokenizer, self.config, 16, 1, 8)
        np.testing.assert_array_equal(ids.numpy(), [1, 20, 21])
        self.assertEqual(selected, [])

    def test_multiple_images_keep_patch_order(self):
        images = [np.zeros([1, 32, 32, 3], "float32"), np.zeros([1, 32, 48, 3], "float32")]
        ids, selected = tokenize_inputs("<|image|> cat <|image|> dog", images, self.tokenizer, self.config, 16, 1, 128)
        expected = [1, 244, 245, 246, 247, 248, *([227] * 4), 230, 20, 244, 245, 246, 247, 248, *([227] * 6), 230, 21]
        np.testing.assert_array_equal(ids.numpy(), expected)
        self.assertEqual(len(selected), 2)
        for actual, original in zip(selected, images):
            self.assertIs(actual, original)

    def test_text_without_image(self):
        ids, selected = tokenize_inputs("cat dog", [], self.tokenizer, self.config, 16, 1, 128)
        np.testing.assert_array_equal(ids.numpy(), [1, 20, 21])
        self.assertEqual(selected, [])

    def test_spatial_and_temporal_positions(self):
        tokens = paddle.to_tensor([[0, 1, 244, 245, 246, 247, 248, 227, 227, 227, 227, 230, 20]])
        masks = paddle.ones([1, 1, 32, 32], dtype="bool")
        pos_t, pos_hw = get_pos_thw(tokens, masks, self.config, 16, pad_token_id=0)
        np.testing.assert_array_equal(pos_t.numpy(), [[0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3]])
        expected = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
        np.testing.assert_allclose(pos_hw.numpy()[0, 7:11], expected, atol=1e-6)
        self.assertTrue(np.isnan(pos_hw.numpy()[tokens.numpy() != 227]).all())
        with self.assertRaisesRegex(AssertionError, "Mismatch"):
            get_pos_thw(tokens, paddle.ones([1, 1, 32, 48], dtype="bool"), self.config, 16, pad_token_id=0)

    def test_process_batch_left_padding_and_image_alignment(self):
        images = [Image.new("RGB", (64, 64), color="white"), Image.new("RGB", (80, 64), color="black")]
        prompts = ["<|image|> cat", "<|image|> cat dog"]
        output = process_batch(self.tokenizer, self.config, list(zip(images, prompts)), 256, 64, 96)
        self.assertEqual(set(output), {"tokens", "pixel_values", "pixel_mask", "pos_t", "pos_hw", "pad_token_id"})
        tokens = output["tokens"].numpy()
        self.assertEqual(output["pixel_values"].shape, [2, 1, 96, 96, 3])
        self.assertEqual(output["pixel_mask"].shape, [2, 1, 96, 96])
        self.assertEqual(output["tokens"].dtype, paddle.int64)
        self.assertEqual(output["pos_t"].shape, output["tokens"].shape)
        self.assertEqual(output["pos_hw"].shape, [*output["tokens"].shape, 2])
        for index, count in enumerate((16, 20)):
            self.assertEqual(int((tokens[index] == 227).sum()), count)
            first_token = int(np.flatnonzero(tokens[index] != 0)[0])
            self.assertEqual(tokens[index, first_token], 1)
            self.assertFalse(np.any(tokens[index, first_token:] == 0))
            self.assertTrue(np.isfinite(output["pos_hw"].numpy()[index, tokens[index] == 227]).all())
            self.assertTrue(np.isnan(output["pos_hw"].numpy()[index, tokens[index] != 227]).all())
        self.assertGreater(int((tokens[0] == 0).sum()), 0)
        self.assertEqual(int((tokens[1] == 0).sum()), 0)
        np.testing.assert_array_equal(output["pixel_mask"].numpy().sum(axis=(1, 2, 3)), [4096, 5120])


if __name__ == "__main__":
    unittest.main()
