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
from unittest.mock import MagicMock, patch


class TestQuickAccessMoEFactory(unittest.TestCase):
    """Tests for paddleformers.nn.moe_deepep.moe_factory.QuickAccessMoEFactory"""

    def test_import(self):
        """Test that QuickAccessMoEFactory can be imported."""
        from paddleformers.nn.moe_deepep.moe_factory import QuickAccessMoEFactory

        self.assertIsNotNone(QuickAccessMoEFactory)

    def test_create_from_model_name_missing_model_type(self):
        """Test that ValueError is raised when model_type is not set on config."""
        from paddleformers.nn.moe_deepep.moe_factory import QuickAccessMoEFactory
        from paddleformers.transformers.configuration_utils import PretrainedConfig

        config = PretrainedConfig(hidden_size=128, moe_intermediate_size=256)
        # Ensure model_type is None
        config.model_type = None

        with self.assertRaises(ValueError) as ctx:
            QuickAccessMoEFactory.create_from_model_name(
                pretrained_config=config,
                expert_class=None,
                gate_activation="softmax",
                expert_activation="silu",
                train_topk_method="greedy",
                inference_topk_method="greedy",
                transpose_gate_weight=False,
            )
        self.assertIn("Cannot determine model type", str(ctx.exception))

    def test_create_from_model_name_calls_modular_moe_layer(self):
        """Test that create_from_model_name correctly builds moe_config and calls ModularMoELayer."""
        from paddleformers.nn.moe_deepep.moe_factory import QuickAccessMoEFactory
        from paddleformers.transformers.configuration_utils import PretrainedConfig

        config = PretrainedConfig(
            hidden_size=64,
            moe_intermediate_size=128,
            model_type="qwen2_moe",
            num_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            hidden_act="gelu",
        )

        with patch("paddleformers.nn.moe_deepep.moe_factory.ModularMoELayer") as mock_layer_cls:
            mock_instance = MagicMock()
            mock_layer_cls.return_value = mock_instance

            result = QuickAccessMoEFactory.create_from_model_name(
                pretrained_config=config,
                expert_class=MagicMock(),
                gate_activation="sigmoid",
                expert_activation="relu",
                train_topk_method="greedy",
                inference_topk_method="greedy",
                transpose_gate_weight=True,
            )

            self.assertIs(result, mock_instance)
            mock_layer_cls.assert_called_once()

    def test_create_from_model_name_moe_config_contents(self):
        """Test that moe_config dict contains the correct key-value pairs."""
        from paddleformers.nn.moe_deepep.moe_factory import QuickAccessMoEFactory
        from paddleformers.transformers.configuration_utils import PretrainedConfig

        config = PretrainedConfig(
            hidden_size=32,
            moe_intermediate_size=64,
            model_type="deepseek_moe",
            num_experts=8,
            num_experts_per_tok=2,
        )

        with patch("paddleformers.nn.moe_deepep.moe_factory.ModularMoELayer") as mock_layer_cls:
            QuickAccessMoEFactory.create_from_model_name(
                pretrained_config=config,
                expert_class=None,
                gate_activation="softmax",
                expert_activation="silu",
                train_topk_method="group_limited_greedy",
                inference_topk_method="greedy",
                transpose_gate_weight=False,
            )

            call_kwargs = mock_layer_cls.call_args[1]
            moe_config = call_kwargs["moe_config"]
            self.assertEqual(moe_config["gate_activation"], "softmax")
            self.assertEqual(moe_config["expert_activation"], "silu")
            self.assertEqual(moe_config["train_topk_method"], "group_limited_greedy")
            self.assertEqual(moe_config["inference_topk_method"], "greedy")

    def test_create_from_model_name_num_experts_from_n_routed(self):
        """Test num_experts fallback chain: num_experts -> n_routed_experts -> moe_num_experts."""
        from paddleformers.nn.moe_deepep.moe_factory import QuickAccessMoEFactory
        from paddleformers.transformers.configuration_utils import PretrainedConfig

        # Use n_routed_experts as fallback
        config = PretrainedConfig(
            hidden_size=32,
            moe_intermediate_size=64,
            model_type="mixtral",
            n_routed_experts=6,
        )

        with patch("paddleformers.nn.moe_deepep.moe_factory.ModularMoELayer") as mock_layer_cls:
            QuickAccessMoEFactory.create_from_model_name(
                pretrained_config=config,
                expert_class=None,
                gate_activation="softmax",
                expert_activation="silu",
                train_topk_method="greedy",
                inference_topk_method="greedy",
                transpose_gate_weight=False,
            )

            call_kwargs = mock_layer_cls.call_args[1]
            self.assertEqual(call_kwargs["num_experts"], 6)

    def test_create_from_model_name_num_experts_from_moe_num_experts(self):
        """Test num_experts fallback: moe_num_experts when num_experts and n_routed_experts are not set."""
        from paddleformers.nn.moe_deepep.moe_factory import QuickAccessMoEFactory
        from paddleformers.transformers.configuration_utils import PretrainedConfig

        config = PretrainedConfig(
            hidden_size=32,
            moe_intermediate_size=64,
            model_type="test_moe",
            moe_num_experts=10,
        )

        with patch("paddleformers.nn.moe_deepep.moe_factory.ModularMoELayer") as mock_layer_cls:
            QuickAccessMoEFactory.create_from_model_name(
                pretrained_config=config,
                expert_class=None,
                gate_activation="softmax",
                expert_activation="silu",
                train_topk_method="greedy",
                inference_topk_method="greedy",
                transpose_gate_weight=False,
            )

            call_kwargs = mock_layer_cls.call_args[1]
            self.assertEqual(call_kwargs["num_experts"], 10)

    def test_create_from_model_name_num_experts_per_tok_from_moe_k(self):
        """Test num_experts_per_tok fallback to moe_k."""
        from paddleformers.nn.moe_deepep.moe_factory import QuickAccessMoEFactory
        from paddleformers.transformers.configuration_utils import PretrainedConfig

        config = PretrainedConfig(
            hidden_size=32,
            moe_intermediate_size=64,
            model_type="test_moe",
            num_experts=4,
            moe_k=3,
        )

        with patch("paddleformers.nn.moe_deepep.moe_factory.ModularMoELayer") as mock_layer_cls:
            QuickAccessMoEFactory.create_from_model_name(
                pretrained_config=config,
                expert_class=None,
                gate_activation="softmax",
                expert_activation="silu",
                train_topk_method="greedy",
                inference_topk_method="greedy",
                transpose_gate_weight=False,
            )

            call_kwargs = mock_layer_cls.call_args[1]
            self.assertEqual(call_kwargs["num_experts_per_tok"], 3)

    def test_create_from_model_name_num_shared_experts_from_moe_num_shared(self):
        """Test num_shared_experts fallback to moe_num_shared_experts."""
        from paddleformers.nn.moe_deepep.moe_factory import QuickAccessMoEFactory
        from paddleformers.transformers.configuration_utils import PretrainedConfig

        config = PretrainedConfig(
            hidden_size=32,
            moe_intermediate_size=64,
            model_type="test_moe",
            num_experts=4,
            moe_num_shared_experts=2,
        )

        with patch("paddleformers.nn.moe_deepep.moe_factory.ModularMoELayer") as mock_layer_cls:
            QuickAccessMoEFactory.create_from_model_name(
                pretrained_config=config,
                expert_class=None,
                gate_activation="softmax",
                expert_activation="silu",
                train_topk_method="greedy",
                inference_topk_method="greedy",
                transpose_gate_weight=False,
            )

            call_kwargs = mock_layer_cls.call_args[1]
            self.assertEqual(call_kwargs["num_shared_experts"], 2)

    def test_create_from_model_name_expert_activation_from_hidden_act(self):
        """Test expert_activation falls back to hidden_act config value."""
        from paddleformers.nn.moe_deepep.moe_factory import QuickAccessMoEFactory
        from paddleformers.transformers.configuration_utils import PretrainedConfig

        config = PretrainedConfig(
            hidden_size=32,
            moe_intermediate_size=64,
            model_type="test_moe",
            num_experts=4,
            hidden_act="tanh",
        )

        with patch("paddleformers.nn.moe_deepep.moe_factory.ModularMoELayer") as mock_layer_cls:
            QuickAccessMoEFactory.create_from_model_name(
                pretrained_config=config,
                expert_class=None,
                gate_activation="softmax",
                expert_activation="silu",
                train_topk_method="greedy",
                inference_topk_method="greedy",
                transpose_gate_weight=False,
            )

            call_kwargs = mock_layer_cls.call_args[1]
            # expert_activation param from config.get("hidden_act", config.get("expert_activation", "silu"))
            # hidden_act is "tanh", so expert_activation should be "tanh"
            self.assertEqual(call_kwargs["expert_activation"], "tanh")

    def test_create_from_model_name_transpose_gate_weight_passed(self):
        """Test transpose_gate_weight is passed through to ModularMoELayer."""
        from paddleformers.nn.moe_deepep.moe_factory import QuickAccessMoEFactory
        from paddleformers.transformers.configuration_utils import PretrainedConfig

        config = PretrainedConfig(
            hidden_size=32,
            moe_intermediate_size=64,
            model_type="test_moe",
            num_experts=4,
        )

        with patch("paddleformers.nn.moe_deepep.moe_factory.ModularMoELayer") as mock_layer_cls:
            QuickAccessMoEFactory.create_from_model_name(
                pretrained_config=config,
                expert_class=MagicMock(),
                gate_activation="softmax",
                expert_activation="silu",
                train_topk_method="greedy",
                inference_topk_method="greedy",
                transpose_gate_weight=True,
            )

            call_kwargs = mock_layer_cls.call_args[1]
            self.assertTrue(call_kwargs["transpose_gate_weight"])

    def test_create_from_model_name_model_type_passed(self):
        """Test model_type from config is passed to ModularMoELayer."""
        from paddleformers.nn.moe_deepep.moe_factory import QuickAccessMoEFactory
        from paddleformers.transformers.configuration_utils import PretrainedConfig

        config = PretrainedConfig(
            hidden_size=32,
            moe_intermediate_size=64,
            model_type="qwen3_moe",
            num_experts=4,
        )

        with patch("paddleformers.nn.moe_deepep.moe_factory.ModularMoELayer") as mock_layer_cls:
            QuickAccessMoEFactory.create_from_model_name(
                pretrained_config=config,
                expert_class=None,
                gate_activation="softmax",
                expert_activation="silu",
                train_topk_method="greedy",
                inference_topk_method="greedy",
                transpose_gate_weight=False,
            )

            call_kwargs = mock_layer_cls.call_args[1]
            self.assertEqual(call_kwargs["model_type"], "qwen3_moe")

    def test_create_from_model_name_pretrained_config_passed(self):
        """Test pretrained_config object is passed through."""
        from paddleformers.nn.moe_deepep.moe_factory import QuickAccessMoEFactory
        from paddleformers.transformers.configuration_utils import PretrainedConfig

        config = PretrainedConfig(
            hidden_size=32,
            moe_intermediate_size=64,
            model_type="test_moe",
            num_experts=4,
        )

        with patch("paddleformers.nn.moe_deepep.moe_factory.ModularMoELayer") as mock_layer_cls:
            QuickAccessMoEFactory.create_from_model_name(
                pretrained_config=config,
                expert_class=None,
                gate_activation="softmax",
                expert_activation="silu",
                train_topk_method="greedy",
                inference_topk_method="greedy",
                transpose_gate_weight=False,
            )

            call_kwargs = mock_layer_cls.call_args[1]
            self.assertIs(call_kwargs["pretrained_config"], config)

    def test_create_from_model_name_expert_class_passed(self):
        """Test expert_class is passed through to ModularMoELayer."""
        from paddleformers.nn.moe_deepep.moe_factory import QuickAccessMoEFactory
        from paddleformers.transformers.configuration_utils import PretrainedConfig

        mock_expert_cls = MagicMock()
        config = PretrainedConfig(
            hidden_size=32,
            moe_intermediate_size=64,
            model_type="test_moe",
            num_experts=4,
        )

        with patch("paddleformers.nn.moe_deepep.moe_factory.ModularMoELayer") as mock_layer_cls:
            QuickAccessMoEFactory.create_from_model_name(
                pretrained_config=config,
                expert_class=mock_expert_cls,
                gate_activation="softmax",
                expert_activation="silu",
                train_topk_method="greedy",
                inference_topk_method="greedy",
                transpose_gate_weight=False,
            )

            call_kwargs = mock_layer_cls.call_args[1]
            self.assertIs(call_kwargs["expert_class"], mock_expert_cls)

    def test_create_from_model_name_all_experts_chain(self):
        """Test that all three num_experts keys are tried in order: num_experts, n_routed_experts, moe_num_experts."""
        from paddleformers.nn.moe_deepep.moe_factory import QuickAccessMoEFactory
        from paddleformers.transformers.configuration_utils import PretrainedConfig

        # num_experts has highest priority
        config = PretrainedConfig(
            hidden_size=32,
            moe_intermediate_size=64,
            model_type="test_moe",
            num_experts=5,
            n_routed_experts=7,
            moe_num_experts=9,
        )

        with patch("paddleformers.nn.moe_deepep.moe_factory.ModularMoELayer") as mock_layer_cls:
            QuickAccessMoEFactory.create_from_model_name(
                pretrained_config=config,
                expert_class=None,
                gate_activation="softmax",
                expert_activation="silu",
                train_topk_method="greedy",
                inference_topk_method="greedy",
                transpose_gate_weight=False,
            )

            call_kwargs = mock_layer_cls.call_args[1]
            # num_experts=5 should take priority
            self.assertEqual(call_kwargs["num_experts"], 5)

    def test_create_from_model_name_is_static_method(self):
        """Test that create_from_model_name is a static method."""
        from paddleformers.nn.moe_deepep.moe_factory import QuickAccessMoEFactory

        self.assertTrue(callable(getattr(QuickAccessMoEFactory, "create_from_model_name")))
        import inspect

        self.assertIsInstance(
            inspect.getattr_static(QuickAccessMoEFactory, "create_from_model_name"),
            staticmethod,
        )


if __name__ == "__main__":
    unittest.main()
