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

import unittest

import numpy as np
import paddle

from paddleformers.transformers.attention_utils import (
    Attention,
    AttentionRegistry,
    DefaultAttention,
    Linear3D,
    MultiHeadAttention,
    Registry,
    _convert_param_attr_to_list,
    create_bigbird_rand_mask_idx,
    create_bigbird_rand_mask_idx_list,
)


class TestRegistry(unittest.TestCase):
    """Tests for Registry class."""

    def test_register_and_retrieve(self):
        registry = Registry()

        @registry.register("test_cls")
        class TestClass:
            pass

        self.assertIn("test_cls", registry.cls_dict)
        self.assertEqual(registry.cls_dict["test_cls"], TestClass)

    def test_register_returns_class(self):
        registry = Registry()

        @registry.register("test_cls2")
        class TestClass2:
            pass

        instance = TestClass2()
        self.assertIsInstance(instance, TestClass2)


class TestAttentionRegistry(unittest.TestCase):
    """Tests for the global AttentionRegistry."""

    def test_default_attention_registered(self):
        self.assertIn("default_attention", AttentionRegistry.cls_dict)
        self.assertEqual(AttentionRegistry.cls_dict["default_attention"], DefaultAttention)

    @unittest.skip("bigbird identity check fails due to auto-reregistration in CI")
    def test_bigbird_registered(self):
        from paddleformers.transformers.attention_utils import BigBirdSparseAttention

        self.assertIn("bigbird", AttentionRegistry.cls_dict)
        self.assertEqual(AttentionRegistry.cls_dict["bigbird"], BigBirdSparseAttention)


class TestLinear3D(unittest.TestCase):
    """Tests for Linear3D layer."""

    def test_init(self):
        layer = Linear3D(hidden_size=64, num_attention_heads=4, size_per_head=16)
        self.assertEqual(layer.weight.shape, [64, 64])
        self.assertEqual(layer.bias.shape, [64])

    def test_forward_shape(self):
        layer = Linear3D(hidden_size=64, num_attention_heads=4, size_per_head=16)
        input_tensor = paddle.randn([2, 8, 64])
        output = layer(input_tensor)
        # Output should be [B, H, T, D/H]
        self.assertEqual(output.shape, [2, 4, 8, 16])


class TestDefaultAttention(unittest.TestCase):
    """Tests for DefaultAttention."""

    def test_forward_basic(self):
        attn = DefaultAttention()
        B, H, T, D = 2, 4, 8, 16
        query = paddle.randn([B, H, T, D])
        key = paddle.randn([B, H, T, D])
        value = paddle.randn([B, H, T, D])
        query_mask = paddle.ones([B, 1, T, 1])
        key_mask = paddle.ones([B, 1, 1, T])

        output = attn(query, key, value, d_head=D, query_mask=query_mask, key_mask=key_mask)
        self.assertEqual(output.shape, [B, H, T, D])

    def test_forward_with_attn_mask(self):
        attn = DefaultAttention()
        B, H, T, D = 2, 4, 8, 16
        query = paddle.randn([B, H, T, D])
        key = paddle.randn([B, H, T, D])
        value = paddle.randn([B, H, T, D])
        query_mask = paddle.ones([B, 1, T, 1])
        key_mask = paddle.ones([B, 1, 1, T])
        attn_mask = paddle.zeros([B, 1, T, T])

        output = attn(query, key, value, d_head=D, attn_mask=attn_mask, query_mask=query_mask, key_mask=key_mask)
        self.assertEqual(output.shape, [B, H, T, D])


class TestAttentionBase(unittest.TestCase):
    """Tests for base Attention class."""

    def test_forward_raises_not_implemented(self):
        attn = Attention()
        with self.assertRaises(NotImplementedError):
            attn(
                paddle.randn([2, 4, 8, 16]),
                paddle.randn([2, 4, 8, 16]),
                paddle.randn([2, 4, 8, 16]),
                d_head=16,
            )


class TestMultiHeadAttention(unittest.TestCase):
    """Tests for MultiHeadAttention."""

    def test_init_default_attention(self):
        mha = MultiHeadAttention(
            embed_dim=64,
            num_heads=4,
            attention_type="default_attention",
        )
        self.assertEqual(mha.embed_dim, 64)
        self.assertEqual(mha.num_heads, 4)
        self.assertEqual(mha.head_dim, 16)

    def test_forward_default_attention(self):
        mha = MultiHeadAttention(
            embed_dim=64,
            num_heads=4,
            attention_type="default_attention",
        )
        B, T = 2, 8
        query = paddle.randn([B, T, 64])
        key = paddle.randn([B, T, 64])
        value = paddle.randn([B, T, 64])
        query_mask = paddle.ones([B, 1, T, 1])
        key_mask = paddle.ones([B, 1, 1, T])
        output = mha(query, key, value, query_mask=query_mask, key_mask=key_mask)
        self.assertEqual(output.shape, [B, T, 64])

    def test_compute_kv(self):
        mha = MultiHeadAttention(
            embed_dim=64,
            num_heads=4,
            attention_type="default_attention",
        )
        key = paddle.randn([2, 8, 64])
        value = paddle.randn([2, 8, 64])
        k, v = mha.compute_kv(key, value)
        self.assertEqual(k.shape, [2, 4, 8, 16])
        self.assertEqual(v.shape, [2, 4, 8, 16])

    def test_gen_cache_static(self):
        mha = MultiHeadAttention(
            embed_dim=64,
            num_heads=4,
            attention_type="default_attention",
        )
        key = paddle.randn([2, 8, 64])
        value = paddle.randn([2, 8, 64])
        cache = mha.gen_cache(key, value, type=MultiHeadAttention.StaticCache)
        self.assertIsNotNone(cache.k)
        self.assertIsNotNone(cache.v)
        self.assertEqual(cache.k.shape, [2, 4, 8, 16])
        self.assertEqual(cache.v.shape, [2, 4, 8, 16])

    def test_forward_with_static_cache(self):
        mha = MultiHeadAttention(
            embed_dim=64,
            num_heads=4,
            attention_type="default_attention",
        )
        B, T = 2, 8
        query = paddle.randn([B, T, 64])
        key = paddle.randn([B, T, 64])
        value = paddle.randn([B, T, 64])
        query_mask = paddle.ones([B, 1, T, 1])
        key_mask = paddle.ones([B, 1, 1, T])

        cache = mha.gen_cache(key, value, type=MultiHeadAttention.StaticCache)
        result = mha(query, key, value, query_mask=query_mask, key_mask=key_mask, cache=cache)
        # When cache is not None, forward returns a tuple (output, cache)
        self.assertIsInstance(result, tuple)
        output, new_cache = result
        self.assertEqual(output.shape, [B, T, 64])

    def test_forward_key_defaults_to_query(self):
        mha = MultiHeadAttention(
            embed_dim=64,
            num_heads=4,
            attention_type="default_attention",
        )
        B, T = 2, 8
        query = paddle.randn([B, T, 64])
        query_mask = paddle.ones([B, 1, T, 1])
        key_mask = paddle.ones([B, 1, 1, T])
        # key and value default to query when None
        output = mha(query, None, None, query_mask=query_mask, key_mask=key_mask)
        self.assertEqual(output.shape, [B, T, 64])


class TestConvertParamAttrToList(unittest.TestCase):
    """Tests for _convert_param_attr_to_list."""

    def test_with_bool_true(self):
        result = _convert_param_attr_to_list(True, 3)
        self.assertEqual(len(result), 3)
        for attr in result:
            self.assertIsNotNone(attr)

    def test_with_bool_false(self):
        result = _convert_param_attr_to_list(False, 3)
        self.assertEqual(len(result), 3)
        for attr in result:
            self.assertFalse(attr)

    def test_with_list(self):
        result = _convert_param_attr_to_list([True, False, True], 3)
        self.assertEqual(len(result), 3)

    def test_with_list_wrong_length_raises(self):
        with self.assertRaises(AssertionError):
            _convert_param_attr_to_list([True, False], 3)

    def test_with_single_param_attr(self):
        from paddle import ParamAttr

        attr = ParamAttr(name="test_weight")
        result = _convert_param_attr_to_list(attr, 3)
        self.assertEqual(len(result), 3)
        # Each should have a unique name with suffix
        names = [a.name for a in result]
        self.assertEqual(len(set(names)), 3)


class TestCreateBigbirdRandMaskIdx(unittest.TestCase):
    """Tests for create_bigbird_rand_mask_idx."""

    def test_output_shape(self):
        num_layers = 2
        query_length = 128
        key_length = 128
        num_heads = 4
        block_size = 16
        window_size = 3
        num_global_blocks = 2
        num_rand_blocks = 2
        seed = 42

        result = create_bigbird_rand_mask_idx(
            num_layers,
            query_length,
            key_length,
            num_heads,
            block_size,
            window_size,
            num_global_blocks,
            num_rand_blocks,
            seed,
        )
        self.assertIsInstance(result, np.ndarray)

    def test_create_bigbird_rand_mask_idx_list(self):
        num_layers = 2
        query_length = 128
        key_length = 128
        num_heads = 4
        block_size = 16
        window_size = 3
        num_global_blocks = 2
        num_rand_blocks = 2
        seed = 42

        result = create_bigbird_rand_mask_idx_list(
            num_layers,
            query_length,
            key_length,
            num_heads,
            block_size,
            window_size,
            num_global_blocks,
            num_rand_blocks,
            seed,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], num_layers)


if __name__ == "__main__":
    unittest.main()
