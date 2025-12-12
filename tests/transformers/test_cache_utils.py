# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import paddle

from paddleformers.transformers import AutoModelForCausalLM, AutoTokenizer
from paddleformers.transformers.cache_utils import (
    DynamicCache,
    DynamicLayer,
    DynamicSlidingWindowLayer,
)
from paddleformers.transformers.configuration_utils import PretrainedConfig


class CacheUtilsTest(unittest.TestCase):
    """
    Test cache_utils.py implementations.
    """

    def setUp(self):
        """
        Set up common configs and tensors
        All tensor shapes are [B, S, N, H] (Batch, SeqLen, NumHeads, HeadDim)
        """
        self.config_full = PretrainedConfig()
        self.config_full.num_hidden_layers = 2
        self.config_full.layer_types = ["full_attention", "full_attention"]

        self.config_hybrid = PretrainedConfig()
        self.config_hybrid.num_hidden_layers = 2
        self.config_hybrid.sliding_window = 4
        self.config_hybrid.layer_types = ["full_attention", "sliding_attention"]

        # [B=2, N=1, S=3, H=1]
        self.prefill_batch2 = paddle.to_tensor(
            [[[[1.0], [2.0], [3.0]]], [[[10.0], [20.0], [30.0]]]],
            dtype="float32",
        )

        # [B=2, N=1, S=1, H=1]
        self.update_batch2 = paddle.to_tensor(
            [
                [[[4.0]]],  # Batch 0
                [[[40.0]]],  # Batch 1
            ],
            dtype="float32",
        )

    def test_dynamic_cache_lazy_init_default(self):
        """Test DynamicCache default (lazy) behavior without config"""

        cache_lazy = DynamicCache()
        self.assertEqual(len(cache_lazy.layers), 0)
        self.assertEqual(cache_lazy.layer_class_to_replicate, DynamicLayer)

        prefill = paddle.to_tensor([1.0]).reshape([1, 1, 1, 1])
        cache_lazy.update(prefill, prefill, 0)
        self.assertEqual(len(cache_lazy.layers), 1)
        self.assertIsInstance(cache_lazy.layers[0], DynamicLayer)

        cache_lazy.update(prefill, prefill, 1)
        self.assertEqual(len(cache_lazy.layers), 2)
        self.assertIsInstance(cache_lazy.layers[1], DynamicLayer)

    def test_dynamic_cache_init_with_config(self):
        """Test DynamicCache correctly initializes layer types from config"""
        config = PretrainedConfig()
        config.num_hidden_layers = 4
        config.sliding_window = 128
        config.layer_types = [
            "full_attention",
            "sliding_attention",
            "full_attention",
            "chunked_attention",
        ]

        cache = DynamicCache(config=config)

        self.assertEqual(len(cache.layers), 4)
        self.assertIsNone(cache.layer_class_to_replicate)

        self.assertIsInstance(cache.layers[0], DynamicLayer)
        self.assertIsInstance(cache.layers[1], DynamicSlidingWindowLayer)
        self.assertIsInstance(cache.layers[2], DynamicLayer)
        self.assertIsInstance(cache.layers[3], DynamicSlidingWindowLayer)

        self.assertEqual(cache.layers[1].sliding_window, 128)
        self.assertEqual(cache.layers[3].sliding_window, 128)

    def test_dynamic_cache_update_logic(self):
        """Test DynamicCache multi-layer update logic."""
        prefill = paddle.to_tensor([1.0, 2.0], dtype="float32").reshape([1, 1, -1, 1])
        update3 = paddle.to_tensor(3.0, dtype="float32").reshape([1, 1, 1, 1])
        update4 = paddle.to_tensor(4.0, dtype="float32").reshape([1, 1, 1, 1])

        # Scenario 1: Single layer
        cache = DynamicCache()
        cache.update(prefill, prefill, 0)
        cache.update(update3, update3, 0)
        self.assertEqual(cache.layers[0].keys[0, 0, :, 0].tolist(), [1.0, 2.0, 3.0])

        cache.update(update4, update4, 0)
        self.assertEqual(cache.layers[0].keys[0, 0, :, 0].tolist(), [1.0, 2.0, 3.0, 4.0])

        # Scenario 2: Multi-layer
        prefill1 = paddle.to_tensor([10.0, 20.0], dtype="float32").reshape([1, 1, -1, 1])
        update3_1 = paddle.to_tensor(30.0, dtype="float32").reshape([1, 1, 1, 1])
        update4_1 = paddle.to_tensor(40.0, dtype="float32").reshape([1, 1, 1, 1])

        cache = DynamicCache()
        cache.update(prefill, prefill, 0)
        cache.update(prefill1, prefill1, 1)

        cache.update(update3, update3, 0)
        cache.update(update3_1, update3_1, 1)
        cache.update(update4, update4, 0)
        cache.update(update4_1, update4_1, 1)

        self.assertEqual(cache.layers[0].keys[0, 0, :, 0].tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(cache.layers[1].keys[0, 0, :, 0].tolist(), [10.0, 20.0, 30.0, 40.0])

    def test_dynamic_cache_batch_select_indices(self):
        """Test batch_select_indices can correctly slice from the batch dim."""
        cache = DynamicCache(config=self.config_full)
        cache.update(self.prefill_batch2, self.prefill_batch2, 0)
        cache.update(self.prefill_batch2, self.prefill_batch2, 1)
        self.assertEqual(cache.layers[0].keys.shape[0], 2)

        indices = paddle.to_tensor([1], dtype="int64")
        cache.batch_select_indices(indices)

        self.assertEqual(cache.layers[0].keys.shape[0], 1)
        self.assertEqual(cache.layers[1].keys.shape[0], 1)
        self.assertEqual(
            cache.layers[0].keys[0, 0, :, 0].tolist(),
            [10.0, 20.0, 30.0],
        )

    def test_dynamic_sliding_window_layer_logic(self):
        """Specific test for paddleformers DynamicSlidingWindowLayer."""
        config = PretrainedConfig()
        config.num_hidden_layers = 1
        config.sliding_window = 4  # sliding_window = 4
        config.layer_types = ["sliding_attention"]

        cache = DynamicCache(config=config)
        self.assertIsInstance(cache.layers[0], DynamicSlidingWindowLayer)
        # Goal: store window - 1 = 3 tokens

        # 1. Prefill 3 tokens (less than window)
        prefill = paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32").reshape([1, 1, -1, 1])
        keys, values = cache.update(prefill, prefill, 0)

        self.assertEqual(keys[0, 0, :, 0].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(cache.layers[0].keys[0, 0, :, 0].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(cache.layers[0].cumulative_length, 3)

        # 2. Add 1 token (total 4)
        update4 = paddle.to_tensor(4.0, dtype="float32").reshape([1, 1, 1, 1])
        keys, values = cache.update(update4, update4, 0)

        self.assertEqual(keys[0, 0, :, 0].tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(cache.layers[0].keys[0, 0, :, 0].tolist(), [2.0, 3.0, 4.0])
        self.assertEqual(cache.layers[0].cumulative_length, 4)

        # 3. Add 1 more token (total 5)
        update5 = paddle.to_tensor(5.0, dtype="float32").reshape([1, 1, 1, 1])
        keys, values = cache.update(update5, update5, 0)

        self.assertEqual(keys[0, 0, :, 0].tolist(), [2.0, 3.0, 4.0, 5.0])
        self.assertEqual(cache.layers[0].keys[0, 0, :, 0].tolist(), [3.0, 4.0, 5.0])
        self.assertEqual(cache.layers[0].cumulative_length, 5)

        # 4. Test long prompt (prefill > window)
        cache_long = DynamicCache(config=config)
        long_prefill = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype="float32").reshape([1, 1, -1, 1])
        keys, values = cache_long.update(long_prefill, long_prefill, 0)

        self.assertEqual(keys[0, 0, :, 0].tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(cache_long.layers[0].keys[0, 0, :, 0].tolist(), [4.0, 5.0, 6.0])
        self.assertEqual(cache_long.layers[0].cumulative_length, 6)

    def test_cache_reorder(self):
        """Test reorder_cache (for beam search)"""
        cache = DynamicCache(config=self.config_full)
        cache.update(self.prefill_batch2, self.prefill_batch2, 0)

        beam_idx = paddle.to_tensor([1, 0], dtype="int64")
        cache.reorder_cache(beam_idx)

        self.assertEqual(cache.layers[0].keys.shape[0], 2)
        self.assertEqual(cache.layers[0].keys[0, 0, :, 0].tolist(), [10.0, 20.0, 30.0])
        self.assertEqual(cache.layers[0].keys[1, 0, :, 0].tolist(), [1.0, 2.0, 3.0])

    def test_cache_crop(self):
        """Test crop"""
        cache = DynamicCache(config=self.config_full)

        cache.update(self.prefill_batch2, self.prefill_batch2, 0)
        cache.update(self.prefill_batch2, self.prefill_batch2, 1)

        self.assertEqual(cache.get_seq_length(0), 3)
        self.assertEqual(cache.get_seq_length(1), 3)

        cache.crop(2)
        self.assertEqual(cache.get_seq_length(0), 2)
        self.assertEqual(cache.get_seq_length(1), 2)
        self.assertEqual(cache.layers[0].keys[0, 0, :, 0].tolist(), [1.0, 2.0])
        self.assertEqual(cache.layers[1].keys[1, 0, :, 0].tolist(), [10.0, 20.0])

        cache.crop(-1)
        self.assertEqual(cache.get_seq_length(0), 1)
        self.assertEqual(cache.get_seq_length(1), 1)
        self.assertEqual(cache.layers[0].keys[0, 0, :, 0].tolist(), [1.0])

    def test_get_mask_sizes(self):
        """Test get_mask_sizes"""
        cache = DynamicCache(config=self.config_hybrid)

        # 1. Prefill S=3
        cache.update(self.prefill_batch2, self.prefill_batch2, 0)
        cache.update(self.prefill_batch2, self.prefill_batch2, 1)

        self.assertEqual(cache.get_seq_length(0), 3)
        self.assertEqual(cache.layers[1].cumulative_length, 3)

        # 2. Prepare update S=1 (query_length=1)
        cache_position = paddle.to_tensor([3], dtype="int64")

        # Layer 0 (Full)
        kv_len, kv_off = cache.get_mask_sizes(cache_position, 0)
        self.assertEqual(kv_len, 4)
        self.assertEqual(kv_off, 0)

        # Layer 1 (Sliding, window=4), S_old=3 (not full)
        kv_len, kv_off = cache.get_mask_sizes(cache_position, 1)
        self.assertEqual(kv_len, 4)
        self.assertEqual(kv_off, 0)

        # 3. Update again, S_old = 4
        cache.update(self.update_batch2, self.update_batch2, 0)
        cache.update(self.update_batch2, self.update_batch2, 1)

        cache_position = paddle.to_tensor([4], dtype="int64")

        # Layer 0 (Full), S_old=4
        kv_len, kv_off = cache.get_mask_sizes(cache_position, 0)
        self.assertEqual(kv_len, 5)
        self.assertEqual(kv_off, 0)

        # Layer 1 (Sliding, window=4), S_old=4 (is full)
        kv_len, kv_off = cache.get_mask_sizes(cache_position, 1)
        self.assertEqual(kv_len, 4)
        self.assertEqual(kv_off, 1)

    def test_cache_properties_and_utils(self):
        """Test reset, __len__, is_sliding"""
        cache = DynamicCache(config=self.config_hybrid)
        cache.update(self.prefill_batch2, self.prefill_batch2, 0)
        cache.update(self.prefill_batch2, self.prefill_batch2, 1)

        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.is_sliding, [False, True])
        self.assertTrue(cache.is_initialized)
        self.assertFalse(cache.is_compileable)

        self.assertTrue(paddle.sum(cache.layers[0].keys) != 0)

        cache.reset()
        self.assertEqual(paddle.sum(cache.layers[0].keys).item(), 0)
        self.assertEqual(paddle.sum(cache.layers[1].keys).item(), 0)
        self.assertEqual(cache.layers[1].cumulative_length, 0)
        self.assertEqual(cache.get_seq_length(0), 3)

    def test_batch_repeat_interleave(self):
        """Test batch_repeat_interleave"""
        cache = DynamicCache(config=self.config_full)
        cache.update(self.prefill_batch2, self.prefill_batch2, 0)  # B=2
        cache.batch_repeat_interleave(3)  # B=6

        self.assertEqual(cache.get_seq_length(0), 3)
        self.assertEqual(cache.layers[0].keys.shape[0], 6)

        self.assertEqual(cache.layers[0].keys[0, 0, :, 0].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(cache.layers[0].keys[1, 0, :, 0].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(cache.layers[0].keys[2, 0, :, 0].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(cache.layers[0].keys[3, 0, :, 0].tolist(), [10.0, 20.0, 30.0])
        self.assertEqual(cache.layers[0].keys[4, 0, :, 0].tolist(), [10.0, 20.0, 30.0])
        self.assertEqual(cache.layers[0].keys[5, 0, :, 0].tolist(), [10.0, 20.0, 30.0])

    def test_get_max_cache_shape(self):
        """Test get_max_cache_shape"""
        cache = DynamicCache(config=self.config_hybrid)  # window=4
        cache.update(self.prefill_batch2, self.prefill_batch2, 0)
        cache.update(self.prefill_batch2, self.prefill_batch2, 1)

        self.assertEqual(cache.get_max_cache_shape(0), -1)
        self.assertEqual(cache.get_max_cache_shape(1), 4)

    def test_sliding_window_crop_error(self):
        """Test crop error on a full sliding window"""
        cache = DynamicCache(config=self.config_hybrid)  # window=4

        cache.update(self.prefill_batch2, self.prefill_batch2, 1)
        cache.update(self.update_batch2, self.update_batch2, 1)

        self.assertEqual(cache.layers[1].cumulative_length, 4)
        self.assertTrue(cache.layers[1].cumulative_length >= cache.layers[1].sliding_window)

        with self.assertRaises(ValueError):
            cache.layers[1].crop(2)

    def test_dynamic_cache_ddp_init(self):
        """Test initializing DynamicCache from ddp_cache_data"""
        key_states = paddle.randn([2, 1, 3, 1])
        value_states = paddle.randn([2, 1, 3, 1])

        sliding_window_tensor = paddle.to_tensor([128], dtype="int64")

        ddp_data = [(key_states, value_states, None), (key_states, value_states, sliding_window_tensor)]

        cache = DynamicCache(ddp_cache_data=ddp_data)

        self.assertEqual(len(cache), 2)
        self.assertIsInstance(cache.layers[0], DynamicLayer)
        self.assertIsInstance(cache.layers[1], DynamicSlidingWindowLayer)
        self.assertEqual(cache.layers[1].sliding_window, 128)
        self.assertEqual(cache.get_seq_length(0), 3)
        self.assertEqual(cache.get_seq_length(1), 3)
        self.assertEqual(cache.layers[0].keys.tolist(), key_states.tolist())

    def test_dynamic_cache_iter(self):
        """Test __iter__ for DynamicCache"""
        cache = DynamicCache(config=self.config_hybrid)
        cache.update(self.prefill_batch2, self.prefill_batch2, 0)
        cache.update(self.prefill_batch2, self.prefill_batch2, 1)

        cache_list = list(cache)
        self.assertEqual(len(cache_list), 2)

        # Layer 0 (Full)
        k0, v0, s0 = cache_list[0]

        self.assertEqual(k0.tolist(), self.prefill_batch2.tolist())
        self.assertIsNone(s0)

        # Layer 1 (Sliding)
        k1, v1, s1 = cache_list[1]
        self.assertEqual(k1.shape[2], 3)
        self.assertEqual(s1.item(), 4)

    def test_early_initialization(self):
        """Test early_initialization"""
        cache = DynamicCache(config=self.config_hybrid)
        self.assertFalse(cache.is_initialized)

        cache.early_initialization(batch_size=2, num_heads=1, head_dim=1, dtype="float32", device=paddle.get_device())

        self.assertTrue(cache.is_initialized)
        self.assertEqual(cache.get_seq_length(0), 0)

        expected_shape = [2, 1, 0, 1]

        self.assertEqual(cache.layers[0].keys.shape, expected_shape)
        self.assertEqual(cache.layers[1].keys.shape, expected_shape)


class ModelIntegrationTest(unittest.TestCase):
    """
    Integration tests utilizing actual models to verify cache behavior
    without offloading enabled.
    """

    def setUp(self):
        self.model_id = "PaddleFormers/tiny-random-qwen3"
        self.device = paddle.get_device()

    def test_model_inference_standard(self):
        """
        Standard integration test: Load a model and run inference with use_cache=True.
        Verifies that DynamicCache is used by default and populated correctly.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, convert_from_hf=True, return_dict=True)
        model.to(self.device)
        model.eval()

        input_text = "Hello, world!"
        inputs = tokenizer(input_text, return_tensors="pd")
        # Ensure inputs are on correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 1. Forward pass (prefill)
        with paddle.no_grad():
            outputs = model(**inputs, use_cache=True)

        # 2. Verify cache existence
        self.assertIsNotNone(outputs.past_key_values)
        self.assertIsInstance(outputs.past_key_values, DynamicCache)
        self.assertEqual(len(outputs.past_key_values), model.config.num_hidden_layers)

        # 3. Verify cache device (Should match model device, NOT CPU/Offloaded)
        first_layer_keys = outputs.past_key_values.layers[0].keys
        # Check if they are on the same device as the model parameters
        model_place = model.parameters()[0].place
        self.assertEqual(
            first_layer_keys.place.gpu_device_id(), model_place.gpu_device_id() if model_place.is_gpu_place() else 0
        )

        # 4. Generate next token to verify cache update
        next_token = paddle.to_tensor([[123]]).to(self.device)
        with paddle.no_grad():
            outputs_next = model(input_ids=next_token, past_key_values=outputs.past_key_values, use_cache=True)

        new_seq_len = outputs_next.past_key_values.get_seq_length(0)
        # Expected: original input length + 1
        self.assertEqual(new_seq_len, inputs["input_ids"].shape[1] + 1)


@unittest.skipIf(not paddle.device.is_compiled_with_cuda(), "Offloading requires CUDA")
class CacheOffloadingTest(unittest.TestCase):
    """
    Test the offloading functionality of DynamicCache.
    These tests require a GPU environment because the offloading mechanism
    in cache_utils.py specifically uses `gpu_device_id()` and GPU streams.
    """

    def setUp(self):
        self.device = "gpu"
        # Create some random tensors on GPU
        self.key_states = paddle.randn([1, 1, 5, 4]).to(self.device)
        self.value_states = paddle.randn([1, 1, 5, 4]).to(self.device)

    def test_offloading_basic_logic(self):
        """
        Unit test for offloading
        """
        # Initialize DynamicCache with offloading=True
        cache = DynamicCache(offloading=True)

        # Update
        key_states = paddle.randn((1, 4, 10, 32))
        value_states = paddle.randn((1, 4, 10, 32))
        cache.update(key_states, value_states, layer_idx=0)
        self.assertTrue(cache.offloading)
        self.assertEqual(cache.get_seq_length(0), 10)

    def test_model_inference_with_offloading(self):
        """
        Integration test: Load a tiny random Qwen3 model, enable offloading in cache,
        and perform inference. Verify that generation works and cache is used.
        """
        model_id = "PaddleFormers/tiny-random-qwen3"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, convert_from_hf=True, return_dict=True)
        model.to(self.device)
        model.eval()

        input_text = "Hello, world!"
        inputs = tokenizer(input_text, return_tensors="pd")

        # Manually move tensors to device to avoid BatchEncoding backend check issues
        # (AutoTokenizer might return a standard HF BatchEncoding which checks for torch)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 1. Initialize DynamicCache with offloading enabled
        past_key_values = DynamicCache(config=model.config, offloading=True)

        # 2. Run a forward pass (prefill)
        with paddle.no_grad():
            outputs = model(**inputs, past_key_values=past_key_values, use_cache=True, return_dict=True)

        # 3. Check if cache is populated and offloaded
        self.assertIsNotNone(outputs.past_key_values)
        self.assertEqual(len(outputs.past_key_values), model.config.num_hidden_layers)

        # 4. Run a decoding step (generate one new token)
        next_token = paddle.to_tensor([[123]]).to(self.device)  # Dummy token
        with paddle.no_grad():
            outputs_next = model(input_ids=next_token, past_key_values=outputs.past_key_values, use_cache=True)

        # Check cache grew
        new_seq_len = outputs_next.past_key_values.get_seq_length(0)
        old_seq_len = inputs["input_ids"].shape[1]
        self.assertEqual(new_seq_len, old_seq_len + 1)


if __name__ == "__main__":
    unittest.main()
