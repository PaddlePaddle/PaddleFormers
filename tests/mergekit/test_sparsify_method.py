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

import numpy as np
import paddle

from paddleformers.mergekit import MergeConfig, SparsifyMethod


class TestSparsifyMethod(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tensor = np.array(
            [
                [-0.834894061088562, 0.7672924399375916, -0.981352686882019, 0.8236614465713501, 0.19363074004650116],
                [
                    0.7413361668586731,
                    -0.44731196761131287,
                    0.9544159173965454,
                    0.07453861087560654,
                    0.5572543144226074,
                ],
                [
                    0.9128026962280273,
                    -0.23344580829143524,
                    0.8464474678039551,
                    0.14241063594818115,
                    -0.8475964069366455,
                ],
                [0.598555326461792, 0.9459332823753357, -0.35118913650512695, 0.5437421798706055, 0.6906668543815613],
            ],
            dtype="float32",
        )

    def test_none(self):
        merge_config = MergeConfig(sparsify_type=None)
        sparsify_method = SparsifyMethod(merge_config=merge_config)
        sparsify_tensor = sparsify_method.sparsify(self.tensor.copy())
        self.assertEqual(sparsify_tensor.shape, (4, 5))
        self.assertTrue(np.array_equal(sparsify_tensor, self.tensor))

    def test_dare(self):
        np.random.seed(42)
        merge_config = MergeConfig(sparsify_type="dare", rescale=True, reserve_p=0.7)
        sparsify_method = SparsifyMethod(merge_config=merge_config)
        sparsify_tensor = sparsify_method.sparsify(self.tensor.copy())
        self.assertEqual(sparsify_tensor.shape, (4, 5))

    def test_magprune(self):
        np.random.seed(42)
        merge_config = MergeConfig(sparsify_type="magprune", rescale=True, reserve_p=0.7)
        sparsify_method = SparsifyMethod(merge_config=merge_config)
        sparsify_tensor = sparsify_method.sparsify(self.tensor.copy())
        self.assertEqual(sparsify_tensor.shape, (4, 5))

    def test_trim(self):
        np.random.seed(42)
        merge_config = MergeConfig(sparsify_type="trim", rescale=True, reserve_p=0.7)
        sparsify_method = SparsifyMethod(merge_config=merge_config)
        sparsify_tensor = sparsify_method.sparsify(self.tensor.copy())
        self.assertEqual(sparsify_tensor.shape, (4, 5))
        expected_result = np.array(
            [
                [-0.9439255595207214, 0.867495596408844, -1.1095106601715088, 0.9312260150909424, 0.0],
                [0.8381496071815491, 0.0, 1.0790561437606812, 0.0, 0.6300279498100281],
                [1.0320085287094116, 0.0, 0.956987738609314, 0.0, -0.958286702632904],
                [0.6767225861549377, 1.0694657564163208, 0.0, 0.6147512197494507, 0.7808632254600525],
            ],
            dtype="float32",
        )
        self.assertTrue(np.array_equal(sparsify_tensor, expected_result))

    @classmethod
    def to_paddle_tensor(cls, numpy_tensor):
        """Convert a numpy array to a paddle tensor."""
        return paddle.to_tensor(numpy_tensor, dtype="float32")

    def test_none_paddle(self):
        paddle_tensor = self.to_paddle_tensor(self.tensor)
        merge_config = MergeConfig(sparsify_type=None, tensor_type="pd")
        sparsify_method = SparsifyMethod(merge_config=merge_config)
        sparsify_tensor = sparsify_method.sparsify(paddle_tensor)
        self.assertEqual(sparsify_tensor.shape, paddle_tensor.shape)
        self.assertTrue(
            paddle.allclose(sparsify_tensor, paddle_tensor, atol=1e-6),
            "Paddle tensor sparsify (none) failed to match input tensor.",
        )

    def test_dare_paddle(self):
        paddle.seed(42)  # Fix random seed for reproducibility
        paddle_tensor = self.to_paddle_tensor(self.tensor)
        merge_config = MergeConfig(sparsify_type="dare", rescale=True, reserve_p=0.7, tensor_type="pd")
        sparsify_method = SparsifyMethod(merge_config=merge_config)
        sparsify_tensor = sparsify_method.sparsify(paddle_tensor)
        self.assertEqual(sparsify_tensor.shape, paddle_tensor.shape)

    def test_magprune_paddle(self):
        paddle.seed(42)  # Fix random seed for reproducibility
        paddle_tensor = self.to_paddle_tensor(self.tensor)
        merge_config = MergeConfig(sparsify_type="magprune", rescale=True, reserve_p=0.7, tensor_type="pd")
        sparsify_method = SparsifyMethod(merge_config=merge_config)
        sparsify_tensor = sparsify_method.sparsify(paddle_tensor)
        self.assertEqual(sparsify_tensor.shape, paddle_tensor.shape)

    def test_trim_paddle(self):
        paddle.seed(42)  # Fix random seed for reproducibility
        paddle_tensor = self.to_paddle_tensor(self.tensor)
        merge_config = MergeConfig(sparsify_type="trim", rescale=True, reserve_p=0.7, tensor_type="pd")
        sparsify_method = SparsifyMethod(merge_config=merge_config)
        sparsify_tensor = sparsify_method.sparsify(paddle_tensor)
        self.assertEqual(sparsify_tensor.shape, paddle_tensor.shape)

        expected_result = paddle.to_tensor(
            [
                [-0.9439255595207214, 0.867495596408844, -1.1095106601715088, 0.9312260150909424, 0.0],
                [0.8381496071815491, 0.0, 1.0790561437606812, 0.0, 0.6300279498100281],
                [1.0320085287094116, 0.0, 0.956987738609314, 0.0, -0.958286702632904],
                [0.6767225861549377, 1.0694657564163208, 0.0, 0.6147512197494507, 0.7808632254600525],
            ],
            dtype="float32",
        )
        self.assertTrue(
            paddle.allclose(sparsify_tensor, expected_result, atol=1e-6),
            "Paddle tensor sparsify (trim) result does not match expected result.",
        )
