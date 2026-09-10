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

import numpy as np
import paddle
import paddle.nn as nn


class TestMOELayerBase(unittest.TestCase):
    """Tests for paddleformers.nn.moe.abstract.MOELayerBase"""

    def test_import(self):
        """Test that MOELayerBase can be imported."""
        from paddleformers.nn.moe.abstract import MOELayerBase

        self.assertIsNotNone(MOELayerBase)

    def test_is_paddle_layer(self):
        """Test that MOELayerBase is a subclass of paddle.nn.Layer."""
        from paddleformers.nn.moe.abstract import MOELayerBase

        self.assertTrue(issubclass(MOELayerBase, nn.Layer))

    def test_instantiation(self):
        """Test that MOELayerBase can be instantiated."""
        from paddleformers.nn.moe.abstract import MOELayerBase

        layer = MOELayerBase()
        self.assertIsInstance(layer, nn.Layer)
        self.assertIsInstance(layer, MOELayerBase)

    def test_subclass_instantiation(self):
        """Test that a subclass of MOELayerBase can be instantiated and used as a Layer."""
        from paddleformers.nn.moe.abstract import MOELayerBase

        class DummyMoELayer(MOELayerBase):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        layer = DummyMoELayer()
        self.assertIsInstance(layer, MOELayerBase)
        x = paddle.randn([4, 10])
        out = layer(x)
        self.assertEqual(out.shape, [4, 10])

    def test_subclass_with_parameters(self):
        """Test that MOELayerBase subclass can have trainable parameters."""
        from paddleformers.nn.moe.abstract import MOELayerBase

        class ParamMoELayer(MOELayerBase):
            def __init__(self, hidden_size):
                super().__init__()
                self.weight = self.create_parameter(
                    shape=[hidden_size, hidden_size],
                )

        layer = ParamMoELayer(16)
        params = list(layer.parameters())
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0].shape, [16, 16])

    def test_train_eval_mode(self):
        """Test that MOELayerBase subclass can switch between train and eval modes."""
        from paddleformers.nn.moe.abstract import MOELayerBase

        class SimpleMoELayer(MOELayerBase):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 8)

        layer = SimpleMoELayer()
        layer.train()
        self.assertTrue(layer.training)
        layer.eval()
        self.assertFalse(layer.training)

    def test_named_parameters(self):
        """Test that named_parameters works on MOELayerBase subclass."""
        from paddleformers.nn.moe.abstract import MOELayerBase

        class NamedMoELayer(MOELayerBase):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 4)
                self.fc2 = nn.Linear(4, 4)

        layer = NamedMoELayer()
        names = [n for n, _ in layer.named_parameters()]
        self.assertIn("fc1.weight", names)
        self.assertIn("fc1.bias", names)
        self.assertIn("fc2.weight", names)
        self.assertIn("fc2.bias", names)

    def test_to_call(self):
        """Test that MOELayerBase instance can be called directly."""
        from paddleformers.nn.moe.abstract import MOELayerBase

        class CallableMoE(MOELayerBase):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 5)

            def forward(self, x):
                return x + 1.0

        layer = CallableMoE()
        x = paddle.zeros([2, 5])
        out = layer(x)
        expected = paddle.ones([2, 5])
        np.testing.assert_allclose(out.numpy(), expected.numpy())

    def test_state_dict_roundtrip(self):
        """Test that state_dict save/load works on MOELayerBase subclass."""
        from paddleformers.nn.moe.abstract import MOELayerBase

        class StatefulMoE(MOELayerBase):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 3)

        layer = StatefulMoE()
        state = layer.state_dict()
        self.assertIn("linear.weight", state)
        self.assertIn("linear.bias", state)

        layer2 = StatefulMoE()
        layer2.set_state_dict(state)
        self.assertIsNotNone(layer2)

    def test_add_sublayer(self):
        """Test that sublayers can be added to MOELayerBase subclass."""
        from paddleformers.nn.moe.abstract import MOELayerBase

        class LayeredMoE(MOELayerBase):
            def __init__(self):
                super().__init__()
                self.add_sublayer("expert1", nn.Linear(6, 6))
                self.add_sublayer("expert2", nn.Linear(6, 6))

        layer = LayeredMoE()
        self.assertIsNotNone(layer.expert1)
        self.assertIsNotNone(layer.expert2)


if __name__ == "__main__":
    unittest.main()
