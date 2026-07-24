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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest

import paddle

from paddleformers.fleet.transformer.enums import AttnMaskType


class TestAttnMaskType(unittest.TestCase):
    """Tests for AttnMaskType enum."""

    def test_causal_mask_type(self):
        """Test AttnMaskType.causal exists."""
        self.assertTrue(hasattr(AttnMaskType, "causal"))

    def test_padding_mask_type(self):
        """Test AttnMaskType.padding exists."""
        self.assertTrue(hasattr(AttnMaskType, "padding"))

    def test_arbitrary_mask_type(self):
        """Test AttnMaskType.arbitrary exists."""
        self.assertTrue(hasattr(AttnMaskType, "arbitrary"))


class TestIdentityOp(unittest.TestCase):
    """Tests for IdentityOp and IdentityFuncOp."""

    def test_identity_op_forward(self):
        """Test IdentityOp forward returns input unchanged."""
        from paddleformers.fleet.transformer.identity_op import IdentityOp

        x = paddle.randn([4, 8])
        op = IdentityOp()
        out = op(x)
        self.assertTrue(paddle.allclose(x, out).item())

    def test_identity_func_op_forward(self):
        """Test IdentityFuncOp forward returns input unchanged."""
        from paddleformers.fleet.transformer.identity_op import IdentityFuncOp

        x = paddle.randn([4, 8])
        op = IdentityFuncOp()
        # IdentityFuncOp is designed as IdentityFuncOp(...)(x) -> IdentityOp(x) -> x
        # op() returns the forward method, then calling that with x gives x
        op = IdentityFuncOp()
        result = op()(x)
        self.assertTrue(paddle.allclose(x, result).item())

    def test_identity_func_op_callable(self):
        """Test IdentityFuncOp is callable."""
        from paddleformers.fleet.transformer.identity_op import IdentityFuncOp

        op = IdentityFuncOp()
        # IdentityFuncOp() returns op's forward method (a bound method)
        func = op()
        self.assertTrue(callable(func))


class TestTransformerUtils(unittest.TestCase):
    """Tests for transformer utility functions."""

    def test_attention_mask_func(self):
        """Test attention_mask_func."""
        from paddleformers.fleet.transformer.utils import attention_mask_func

        scores = paddle.randn([1, 1, 4, 4])
        mask = paddle.triu(paddle.ones([1, 1, 4, 4]), diagonal=1).cast("bool")
        result = attention_mask_func(scores, mask)
        self.assertEqual(result.shape, scores.shape)

    def test_is_layer_window_attention_no_sliding_window(self):
        """Test is_layer_window_attention with no sliding window."""
        from paddleformers.fleet.transformer.utils import (
            is_layer_window_attention,
        )

        result = is_layer_window_attention(None, None, 1)
        self.assertFalse(result)


class TestProcessGroupCollection(unittest.TestCase):
    """Tests for ProcessGroupCollection."""

    def test_use_mpu_process_groups(self):
        """Test use_mpu_process_groups."""
        from paddleformers.fleet.process_groups_config import (
            ProcessGroupCollection,
        )

        pg = ProcessGroupCollection.use_mpu_process_groups()
        self.assertIsNotNone(pg)


if __name__ == "__main__":
    unittest.main()
