# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Small-model inference checks using the ModelTester pattern of other models."""
import unittest

import numpy as np
import paddle
from PIL import Image
from pycocotools import mask as mask_utils

from paddleformers.transformers import (
    FalconPerceptionConfig,
    FalconPerceptionForSegmentation,
)
from paddleformers.transformers.falcon_perception.modeling import KVCache
from paddleformers.transformers.falcon_perception.processing import get_pos_thw
from tests.testing_utils import gpu_device_initializer


class FalconPerceptionModelTester:
    def __init__(self, parent, batch_size=2, seq_length=6):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length

    def get_config(self):
        return FalconPerceptionConfig(
            dim=32,
            n_layers=2,
            n_heads=4,
            head_dim=8,
            n_kv_heads=2,
            ffn_dim=48,
            vocab_size=300,
            max_seq_len=128,
            spatial_patch_size=2,
            coord_enc_dim=8,
            coord_dec_dim=24,
            coord_out_dim=16,
            size_enc_dim=8,
            size_dec_dim=24,
            size_out_dim=16,
            do_segmentation=False,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        # Avoid all task/control tokens in the text-only fixture.
        tokens = paddle.arange(20, 20 + self.batch_size * self.seq_length, dtype="int64")
        return config, tokens.reshape([self.batch_size, self.seq_length])

    def create_model(self):
        model = FalconPerceptionForSegmentation(self.get_config()).eval()
        # Exercise spatial RoPE rather than leaving its initialized frequencies zero.
        model.freqs_cis_golden.set_value(paddle.randn(model.freqs_cis_golden.shape) * 0.2)
        return model

    def new_cache(self, model, batch_size):
        config = model.config
        return KVCache(batch_size, config.max_seq_len, config.n_heads, config.head_dim, config.n_layers)

    def positions(self, tokens):
        return (
            paddle.arange(tokens.shape[1], dtype="int64")[None].expand(tokens.shape),
            paddle.full([*tokens.shape, 2], float("nan")),
        )

    def forward(self, model, tokens, *, pos_t=None, pos_hw=None, **kwargs):
        if pos_t is None:
            pos_t, pos_hw = self.positions(tokens)
        return model(
            tokens=tokens,
            attention_mask=model.get_attention_mask(tokens),
            kv_cache=self.new_cache(model, tokens.shape[0]),
            rope_pos_t=pos_t,
            rope_pos_hw=pos_hw,
            **kwargs,
        )

    def image_inputs(self, padded=False):
        tokens = paddle.to_tensor([[244, 245, 246, 247, 248, 227, 227, 227, 227, 230, 20, 21, 22]])
        width = 6 if padded else 4
        pixels = paddle.randn([1, 1, 4, width, 3])
        mask = paddle.ones([1, 1, 4, width], dtype="bool")
        if padded:
            mask[:, :, :, 4:] = False
        pos_t, pos_hw = get_pos_thw(tokens, mask, self.get_config(), spatial_patch_size=2, pad_token_id=0)
        return tokens, dict(pos_t=pos_t, pos_hw=pos_hw, pixel_values=pixels, pixel_mask=mask)

    def check_cached_matches_full(self, model, tokens, prefix_length, *, pos_t=None, pos_hw=None, **kwargs):
        if pos_t is None:
            pos_t, pos_hw = self.positions(tokens)
        with paddle.no_grad():
            full_logits, full_hidden = self.forward(model, tokens, pos_t=pos_t, pos_hw=pos_hw, **kwargs)
            cache = self.new_cache(model, tokens.shape[0])
            attention_mask = model.get_attention_mask(tokens)
            model(
                tokens=tokens[:, :prefix_length],
                attention_mask=attention_mask,
                kv_cache=cache,
                rope_pos_t=pos_t,
                rope_pos_hw=pos_hw,
                **kwargs,
            )
            for index in range(prefix_length, tokens.shape[1]):
                logits, hidden = model(tokens[:, index : index + 1], attention_mask=attention_mask, kv_cache=cache)
                np.testing.assert_allclose(
                    logits.numpy(), full_logits[:, index : index + 1].numpy(), atol=2e-5, rtol=2e-5
                )
                np.testing.assert_allclose(
                    hidden.numpy(), full_hidden[:, index : index + 1].numpy(), atol=2e-5, rtol=2e-5
                )
                self.parent.assertEqual(cache.get_pos(), index + 1)


class FalconPerceptionModelTest(unittest.TestCase):
    @gpu_device_initializer(log_prefix="FalconPerceptionModelTest", gpu_id=0)
    def setUp(self):
        paddle.seed(123)
        self.model_tester = FalconPerceptionModelTester(self)
        self.no_grad = paddle.no_grad()
        self.no_grad.__enter__()
        self.addCleanup(self.no_grad.__exit__, None, None, None)

    def test_model(self):
        config, tokens = self.model_tester.prepare_config_and_inputs()
        model = self.model_tester.create_model()
        outputs = self.model_tester.forward(model, tokens)
        self.assertIsInstance(outputs, tuple)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].shape, [2, 6, config.vocab_size])
        self.assertEqual(outputs[1].shape, [2, 6, config.dim])
        self.assertEqual(outputs[0].dtype, paddle.float32)
        self.assertTrue(bool(paddle.isfinite(outputs[0]).all()))
        self.assertTrue(bool(paddle.isfinite(outputs[1]).all()))

    def test_model_past(self):
        _, tokens = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_cached_matches_full(self.model_tester.create_model(), tokens, prefix_length=3)

    def test_multimodal_model_past(self):
        tokens, inputs = self.model_tester.image_inputs()
        self.model_tester.check_cached_matches_full(
            self.model_tester.create_model(), tokens, prefix_length=11, **inputs
        )

    def test_causal_attention(self):
        _, tokens = self.model_tester.prepare_config_and_inputs()
        model = self.model_tester.create_model()
        baseline = self.model_tester.forward(model, tokens)[0]
        changed = tokens.clone()
        changed[:, 3:] += 40
        actual = self.model_tester.forward(model, changed)[0]
        np.testing.assert_allclose(actual[:, :3].numpy(), baseline[:, :3].numpy(), atol=2e-5, rtol=2e-5)
        self.assertGreater(float(paddle.max(paddle.abs(actual[:, 3:] - baseline[:, 3:]))), 1e-4)

    def test_multimodal_forward_uses_only_valid_pixels(self):
        model = self.model_tester.create_model()
        tokens, inputs = self.model_tester.image_inputs(padded=True)
        baseline = self.model_tester.forward(model, tokens, **inputs)[0]
        padded_pixels = inputs["pixel_values"].clone()
        padded_pixels[:, :, :, 4:] = 999
        actual = self.model_tester.forward(model, tokens, **dict(inputs, pixel_values=padded_pixels))[0]
        np.testing.assert_allclose(actual.numpy(), baseline.numpy(), atol=2e-5, rtol=2e-5)
        changed_pixels = inputs["pixel_values"].clone()
        changed_pixels[:, :, :, :4] += 1
        changed = self.model_tester.forward(model, tokens, **dict(inputs, pixel_values=changed_pixels))[0]
        self.assertGreater(float(paddle.max(paddle.abs(changed[:, -1] - baseline[:, -1]))), 1e-4)

    def test_coordinate_and_size_inputs(self):
        model = self.model_tester.create_model()
        tokens = paddle.to_tensor([[20, 21, 240, 241, 22, 23]])
        baseline = self.model_tester.forward(model, tokens)[0]
        coords = paddle.to_tensor([[0.25, 0.75]])
        sizes = paddle.to_tensor([[0.4, 0.2]])
        logits, hidden = self.model_tester.forward(model, tokens, coord_xy=coords, size_hw=sizes)
        np.testing.assert_allclose(logits[:, :2].numpy(), baseline[:, :2].numpy(), atol=2e-5, rtol=2e-5)
        self.assertGreater(float(paddle.max(paddle.abs(logits[:, 2:] - baseline[:, 2:]))), 1e-4)
        for decoded in (model.decode_coords(hidden, tokens), model.decode_sizes(hidden, tokens)):
            self.assertEqual(decoded.shape, [1, 2, 8])
            self.assertTrue(bool(paddle.isfinite(decoded).all()))
        self.assertEqual(model.decode_coords(hidden, paddle.full_like(tokens, 20)).shape, [0, 2, 8])

    def test_size_decoding_bin_endpoints(self):
        model = self.model_tester.create_model()
        logits = paddle.full([1, 2, 8], -10.0)
        logits[0, 0, 0] = 10
        logits[0, 1, 7] = 10
        np.testing.assert_allclose(model.process_sizes(logits).numpy(), [[1 / 8, 1.0]], atol=1e-7)

    def test_inference_can_repeat_after_cache_reset(self):
        model = self.model_tester.create_model()
        _, tokens = self.model_tester.prepare_config_and_inputs()
        pos_t, pos_hw = self.model_tester.positions(tokens)
        cache = self.model_tester.new_cache(model, tokens.shape[0])
        inputs = dict(
            attention_mask=model.get_attention_mask(tokens), kv_cache=cache, rope_pos_t=pos_t, rope_pos_hw=pos_hw
        )
        first = model(tokens, **inputs)
        cache.reset()
        second = model(tokens, **inputs)
        for actual, expected in zip(second, first):
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())

    def test_postprocess_crop_resize_and_nms(self):
        model = self.model_tester.create_model()
        pixels = paddle.zeros([6, 8], dtype="bool")
        pixels[1:5, 2:6] = True
        logits = paddle.full([6, 8], -20.0)
        logits[1:5, 2:6] = 20
        xy, hw = {"x": 0.5, "y": 0.5}, {"h": 0.8, "w": 0.8}
        detections = model._postprocess_aux([xy, hw, logits, xy, hw, logits], pixels, (8, 10), 0.5)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["xy"], xy)
        self.assertEqual(detections[0]["hw"], hw)
        rle = detections[0]["mask_rle"]
        decoded = mask_utils.decode(dict(size=rle["size"], counts=rle["counts"].encode()))
        np.testing.assert_array_equal(decoded, np.ones([8, 10], dtype="uint8"))
        self.assertEqual(model._postprocess_aux([], pixels, (8, 10), 0.5), [])
        self.assertEqual(model._postprocess_aux([xy, hw], pixels, (8, 10), 0.5), [])

    def test_generate_rejects_unsupported_inputs(self):
        model = self.model_tester.create_model()
        image = Image.new("RGB", (16, 16))
        with self.assertRaisesRegex(ValueError, "compile=False"):
            model.generate(image, "cat", compile=True)
        with self.assertRaisesRegex(ValueError, "one image/query"):
            model.generate([image, image], ["cat", "cat"])
        with self.assertRaisesRegex(ValueError, "one image/query"):
            model.generate(image, ["cat", "dog"])


if __name__ == "__main__":
    unittest.main()
