# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from paddleformers.fleet.training.initialize import initialize_fleet
from paddleformers.fleet.transformer.transformer_config import TransformerConfig

strategy = paddle.distributed.fleet.DistributedStrategy()
initialize_fleet(strategy=strategy)


class TestMoeLayerFreqAndFirstKDenseReplace(unittest.TestCase):
    """Tests for the moe_layer_freq / first_k_dense_replace logic in TransformerConfig.__post_init__."""

    def test_both_none_defaults_moe_layer_freq_to_1(self):
        """When both first_k_dense_replace and moe_layer_freq are None, moe_layer_freq defaults to 1."""
        config = TransformerConfig(
            first_k_dense_replace=None,
            moe_layer_freq=None,
            num_hidden_layers=12,
        )
        self.assertEqual(config.moe_layer_freq, 1)

    def test_first_k_dense_replace_with_list_moe_layer_freq_raises(self):
        """When first_k_dense_replace is set and moe_layer_freq is a list (not int), should raise ValueError."""
        with self.assertRaises(ValueError):
            TransformerConfig(
                first_k_dense_replace=2,
                moe_layer_freq=[1, 0, 1, 0],
                num_hidden_layers=4,
            )

    def test_first_k_dense_replace_with_int_moe_layer_freq(self):
        """When first_k_dense_replace is set and moe_layer_freq is an int,
        it should generate a pattern based on the frequency."""
        config = TransformerConfig(
            first_k_dense_replace=2,
            moe_layer_freq=2,
            num_hidden_layers=8,
        )
        # first 2 layers are dense (0), remaining layers follow pattern: 1 if (i % 2 == 0) else 0
        # Pattern for range(8): i=0->1, i=1->0, i=2->1, i=3->0, i=4->1, i=5->0, i=6->1, i=7->0
        expected = [0, 0, 0, 1, 0, 1, 0, 1]
        self.assertEqual(config.moe_layer_freq, expected)

    def test_first_k_dense_replace_with_moe_layer_freq_none(self):
        """When first_k_dense_replace is set and moe_layer_freq is None,
        the remaining layers should all be MoE (all 1s)."""
        config = TransformerConfig(
            first_k_dense_replace=3,
            moe_layer_freq=None,
            num_hidden_layers=8,
        )
        # both-None check won't trigger since first_k_dense_replace is set,
        # moe_layer_freq stays None (falsy).
        # else branch: moe_layer_pattern = [1] * (8 - 3) = [1, 1, 1, 1, 1]
        # final = [0, 0, 0] + [1, 1, 1, 1, 1]
        expected = [0, 0, 0, 1, 1, 1, 1, 1]
        self.assertEqual(config.moe_layer_freq, expected)

    def test_first_k_dense_replace_with_moe_layer_freq_zero(self):
        """When first_k_dense_replace is set and moe_layer_freq is 0 (falsy int),
        the pattern should fall into the else branch producing all 1s for non-dense layers."""
        config = TransformerConfig(
            first_k_dense_replace=4,
            moe_layer_freq=0,
            num_hidden_layers=10,
        )
        # moe_layer_freq=0 is falsy, so moe_layer_pattern = [1] * (10 - 4) = [1, 1, 1, 1, 1, 1]
        expected = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        self.assertEqual(config.moe_layer_freq, expected)

    def test_first_k_dense_replace_with_moe_layer_freq_3(self):
        """When first_k_dense_replace is set with moe_layer_freq=3,
        every 3rd layer (index % 3 == 0) should be MoE."""
        config = TransformerConfig(
            first_k_dense_replace=1,
            moe_layer_freq=3,
            num_hidden_layers=7,
        )
        # first 1 layer is dense
        # Pattern for range(7): i=0->1, i=1->0, i=2->0, i=3->1, i=4->0, i=5->0, i=6->1
        expected = [0, 0, 0, 1, 0, 0, 1]
        self.assertEqual(config.moe_layer_freq, expected)

    def test_first_k_dense_replace_equals_num_hidden_layers(self):
        """Edge case: first_k_dense_replace equals num_hidden_layers,
        all layers should be dense (all 0s)."""
        config = TransformerConfig(
            first_k_dense_replace=6,
            moe_layer_freq=0,
            num_hidden_layers=6,
        )
        # moe_layer_freq=0 is falsy, so moe_layer_pattern = [1] * (6 - 6) = []
        expected = [0, 0, 0, 0, 0, 0]
        self.assertEqual(config.moe_layer_freq, expected)

    def test_only_moe_layer_freq_int_no_first_k_dense(self):
        """When only moe_layer_freq is set (as int) and first_k_dense_replace is None,
        moe_layer_freq should remain as the integer value."""
        config = TransformerConfig(
            first_k_dense_replace=None,
            moe_layer_freq=2,
            num_hidden_layers=8,
        )
        self.assertEqual(config.moe_layer_freq, 2)

    def test_only_first_k_dense_replace_no_moe_layer_freq(self):
        """When first_k_dense_replace is set and moe_layer_freq is not specified (defaults to None),
        should generate the correct pattern."""
        config = TransformerConfig(
            first_k_dense_replace=2,
            num_hidden_layers=6,
        )
        # moe_layer_freq=None => both-None check won't trigger because first_k_dense_replace is set
        # After the None-None check, moe_layer_freq is still None
        # first_k_dense_replace is truthy => enter the block
        # moe_layer_freq is None (falsy) => moe_layer_pattern = [1] * (6-2) = [1,1,1,1]
        expected = [0, 0, 1, 1, 1, 1]
        self.assertEqual(config.moe_layer_freq, expected)

    def test_first_k_dense_replace_1_moe_layer_freq_1(self):
        """first_k_dense_replace=1, moe_layer_freq=1: first layer dense, rest all MoE."""
        config = TransformerConfig(
            first_k_dense_replace=1,
            moe_layer_freq=1,
            num_hidden_layers=5,
        )
        # moe_layer_freq=1 is truthy int, pattern: 1 if (i % 1 == 0) else 0 => all 1s
        expected = [0, 1, 1, 1, 1]
        self.assertEqual(config.moe_layer_freq, expected)


class TestRoutedScalingFactorConfig(unittest.TestCase):
    """Tests for the routed_scaling_factor and routed_scaling_factor_learnable fields
    in TransformerConfig."""

    def test_routed_scaling_factor_default_is_1(self):
        """routed_scaling_factor defaults to 1.0 when not specified."""
        config = TransformerConfig(num_hidden_layers=4)
        self.assertAlmostEqual(config.routed_scaling_factor, 1.0)

    def test_routed_scaling_factor_learnable_default_is_false(self):
        """routed_scaling_factor_learnable defaults to False when not specified."""
        config = TransformerConfig(num_hidden_layers=4)
        self.assertFalse(config.routed_scaling_factor_learnable)

    def test_routed_scaling_factor_float(self):
        """routed_scaling_factor accepts a float value (e.g., 2.5 for DeepSeek-V3)."""
        config = TransformerConfig(num_hidden_layers=4, routed_scaling_factor=2.5)
        self.assertAlmostEqual(config.routed_scaling_factor, 2.5)

    def test_routed_scaling_factor_learnable_true(self):
        """routed_scaling_factor_learnable can be set to True."""
        config = TransformerConfig(
            num_hidden_layers=4,
            routed_scaling_factor=2.5,
            routed_scaling_factor_learnable=True,
        )
        self.assertAlmostEqual(config.routed_scaling_factor, 2.5)
        self.assertTrue(config.routed_scaling_factor_learnable)


class TestMoETokenDispatcherConfig(unittest.TestCase):
    def test_hybridep_dispatcher_type_is_preserved(self):
        config = TransformerConfig(
            num_hidden_layers=4,
            n_routed_experts=8,
            moe_token_dispatcher_type="hybridep",
        )

        self.assertEqual(config.moe_token_dispatcher_type, "hybridep")
        self.assertTrue(config.moe_use_fusion_node)


class TestMagicInit(unittest.TestCase):
    """Tests for the magic_init functionality in TransformerConfig."""

    def test_magic_init_false_default_behavior(self):
        """When magic_init is False (default), normal init methods should be used."""
        config = TransformerConfig(
            num_hidden_layers=12,
            hidden_size=768,
            magic_init=False,
        )
        # When False, init_method should be set but not the magic init
        self.assertIsNotNone(config.init_method)
        self.assertIsNotNone(config.output_layer_init_method)

    def test_magic_init_true_sigma_calculation(self):
        """When magic_init is True, sigma should be sqrt(0.3333 / hidden_size)."""
        import math

        hidden_size = 768
        config = TransformerConfig(
            num_hidden_layers=12,
            hidden_size=hidden_size,
            magic_init=True,
        )
        expected_sigma = math.sqrt(0.3333 / hidden_size)
        self.assertAlmostEqual(config.init_method_std, expected_sigma, places=6)

    def test_magic_init_true_all_methods_same(self):
        """When magic_init is True, all init methods should be the same."""
        config = TransformerConfig(
            num_hidden_layers=12,
            hidden_size=768,
            magic_init=True,
        )
        # All init methods should be the same function
        self.assertIs(config.init_method, config.output_layer_init_method)
        self.assertIs(config.init_method, config.embedding_init_method)

    def test_magic_init_true_different_hidden_sizes(self):
        """Test sigma calculation with different hidden sizes."""
        import math

        for hidden_size in [512, 768, 1024, 2048, 4096]:
            config = TransformerConfig(
                num_hidden_layers=12,
                hidden_size=hidden_size,
                magic_init=True,
            )
            expected_sigma = math.sqrt(0.3333 / hidden_size)
            self.assertAlmostEqual(config.init_method_std, expected_sigma, places=6)

    def test_magic_init_true_init_method_matches_get_magic_init_method(self):
        """When magic_init is True, init method should match get_magic_init_method."""
        import math

        from paddleformers.fleet.utils import get_magic_init_method

        hidden_size = 768
        config = TransformerConfig(
            num_hidden_layers=12,
            hidden_size=hidden_size,
            magic_init=True,
        )

        # Create test weight
        weight = paddle.randn([100, 100])

        # Apply config's init method
        config.init_method(weight)

        # Calculate expected using get_magic_init_method
        expected_sigma = math.sqrt(0.3333 / hidden_size)
        magic_init = get_magic_init_method(expected_sigma)
        expected_weight = paddle.randn([100, 100])
        magic_init(expected_weight)

        # Compare results using same random seed
        paddle.seed(1234)
        weight1 = paddle.randn([100, 100])
        config.init_method(weight1)

        paddle.seed(1234)
        weight2 = paddle.randn([100, 100])
        magic_init(weight2)

        paddle.testing.assert_close(weight1, weight2, rtol=1e-6, atol=1e-6)

    def test_magic_init_false_uses_normal_init(self):
        """When magic_init is False, normal init methods should be used."""
        config = TransformerConfig(
            num_hidden_layers=12,
            hidden_size=768,
            magic_init=False,
        )
        # Should have init_method_std set to normal value
        self.assertIsNotNone(config.init_method_std)
        # Should be a reasonable value for normal init (not the magic init value)
        import math

        magic_sigma = math.sqrt(0.3333 / 768)
        self.assertNotAlmostEqual(config.init_method_std, magic_sigma, places=6)

    def test_magic_init_true_with_moe(self):
        """Test magic_init works correctly with MoE models."""
        import math

        config = TransformerConfig(
            num_hidden_layers=12,
            hidden_size=768,
            n_routed_experts=8,
            magic_init=True,
        )
        expected_sigma = math.sqrt(0.3333 / 768)
        self.assertAlmostEqual(config.init_method_std, expected_sigma, places=6)
        # All init methods should still be the same
        self.assertIs(config.init_method, config.output_layer_init_method)
        self.assertIs(config.init_method, config.embedding_init_method)

    def test_magic_init_true_raises_on_zero_hidden_size(self):
        """When magic_init is True and hidden_size is 0, should raise ValueError."""
        with self.assertRaises(
            ValueError,
            msg="hidden_size must be non-zero when magic_init is True.",
        ):
            TransformerConfig(
                num_hidden_layers=12,
                hidden_size=0,
                magic_init=True,
            )


class TestPadTokenId(unittest.TestCase):
    """Tests for the pad_token_id field on TransformerConfig."""

    def test_default_is_zero(self):
        config = TransformerConfig(num_hidden_layers=2)
        self.assertEqual(config.pad_token_id, 0)

    def test_override_value(self):
        config = TransformerConfig(num_hidden_layers=2, pad_token_id=151643)
        self.assertEqual(config.pad_token_id, 151643)


if __name__ == "__main__":
    unittest.main()
