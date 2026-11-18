# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Exercise the segmentation generation pipeline with controlled token sampling."""
import unittest
from unittest.mock import patch

import numpy as np
import paddle
from PIL import Image
from pycocotools import mask as mask_utils

from paddleformers.transformers import FalconPerceptionForSegmentation
from tests.testing_utils import gpu_device_initializer
from tests.transformers.falcon_perception.test_modeling import (
    FalconPerceptionModelTester,
)
from tests.transformers.falcon_perception.test_processing import make_tokenizer


class FalconPerceptionGenerationTest(unittest.TestCase):
    @gpu_device_initializer(log_prefix="FalconPerceptionGenerationTest", gpu_id=0)
    def setUp(self):
        previous_dtype = paddle.get_default_dtype()
        self.addCleanup(paddle.set_default_dtype, previous_dtype)
        paddle.set_default_dtype("float32")
        paddle.seed(42)
        config = FalconPerceptionModelTester(self).get_config()
        config.spatial_patch_size = 16
        config.max_seq_len = 256
        config.do_segmentation = True
        config.segm_out_dim = 16
        config.num_segm_layers = 2
        self.model = FalconPerceptionForSegmentation(config).eval()
        # The released checkpoint supplies these persistent frequencies.
        self.model.freqs_cis_golden.set_value(paddle.randn(self.model.freqs_cis_golden.shape) * 0.2)
        self.model.itok_upsampler.rope.freqs.set_value(paddle.randn(self.model.itok_upsampler.rope.freqs.shape) * 0.2)
        self.model._tokenizer = make_tokenizer()
        self.image = Image.fromarray(np.random.default_rng(42).integers(0, 256, [64, 64, 3], dtype="uint8"))

    def generate_with_tokens(self, tokens):
        samples = [paddle.to_tensor([[token]], dtype="int64") for token in tokens]
        # Only sampling is controlled. Forward/cache, geometry heads, AnyUp,
        # segmentation projection, resizing, and RLE all execute normally.
        with patch(
            "paddleformers.transformers.falcon_perception.modeling.sample_next_token", side_effect=samples
        ) as sampler:
            result = self.model.generate(self.image, "cat", min_dimension=64, max_dimension=64, max_new_tokens=8)
        self.assertEqual(sampler.call_count, len(tokens))
        self.assertEqual(len(result), 1)
        return result[0]

    def test_generates_instance_and_resets_between_calls(self):
        detections = self.generate_with_tokens([240, 241, 262, 263])
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(set(detection), {"xy", "hw", "mask_rle"})
        for key in ("x", "y"):
            self.assertGreaterEqual(detection["xy"][key], 0)
            self.assertLessEqual(detection["xy"][key], 1)
        for key in ("h", "w"):
            self.assertGreater(detection["hw"][key], 0)
            self.assertLessEqual(detection["hw"][key], 1)
        rle = detection["mask_rle"]
        self.assertEqual(rle["size"], [64, 64])
        self.assertIsInstance(rle["counts"], str)
        mask = mask_utils.decode(dict(size=rle["size"], counts=rle["counts"].encode()))
        self.assertEqual(mask.shape, (64, 64))
        self.assertTrue(np.isin(mask, [0, 1]).all())
        self.assertEqual(self.generate_with_tokens([263]), [])

    def test_both_stop_tokens_produce_empty_result(self):
        for token in (self.model.config.eos_id, 263):
            with self.subTest(stop_token=token):
                self.assertEqual(self.generate_with_tokens([token]), [])


if __name__ == "__main__":
    unittest.main()
