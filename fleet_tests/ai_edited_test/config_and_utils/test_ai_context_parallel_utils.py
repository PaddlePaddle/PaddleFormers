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


class TestContextParallelUtilsMarkParam(unittest.TestCase):
    """Tests for mark_context_parallel_parameter_disable_scale_grad."""

    def test_mark_layer(self):
        """Test marking a layer."""
        from paddleformers.fleet.context_parallel_utils import (
            mark_context_parallel_parameter_disable_scale_grad,
        )

        layer = paddle.nn.Linear(4, 4)
        mark_context_parallel_parameter_disable_scale_grad(layer)
        self.assertTrue(layer.weight.context_parallel_disable_scale_grad)

    def test_mark_layer_with_bias(self):
        """Test marking a layer with bias."""
        from paddleformers.fleet.context_parallel_utils import (
            mark_context_parallel_parameter_disable_scale_grad,
        )

        layer = paddle.nn.Linear(4, 4, bias_attr=True)
        mark_context_parallel_parameter_disable_scale_grad(layer)
        self.assertTrue(layer.weight.context_parallel_disable_scale_grad)
        self.assertTrue(layer.bias.context_parallel_disable_scale_grad)

    def test_mark_tensor(self):
        """Test marking a tensor."""
        from paddleformers.fleet.context_parallel_utils import (
            mark_context_parallel_parameter_disable_scale_grad,
        )

        t = paddle.randn([4, 4])
        mark_context_parallel_parameter_disable_scale_grad(t)
        self.assertTrue(t.context_parallel_disable_scale_grad)

    def test_mark_invalid_type_raises(self):
        """Test marking invalid type raises TypeError."""
        from paddleformers.fleet.context_parallel_utils import (
            mark_context_parallel_parameter_disable_scale_grad,
        )

        with self.assertRaises(TypeError):
            mark_context_parallel_parameter_disable_scale_grad("not_a_param")


class TestContextParallelParamDisableScale(unittest.TestCase):
    """Tests for context_parallel_parameter_disable_scale_grad."""

    def test_unmarked_param_returns_false(self):
        """Test unmarked param returns False."""
        from paddleformers.fleet.context_parallel_utils import (
            context_parallel_parameter_disable_scale_grad,
        )

        t = paddle.randn([4, 4])
        self.assertFalse(context_parallel_parameter_disable_scale_grad(t))

    def test_marked_param_returns_true(self):
        """Test marked param returns True."""
        from paddleformers.fleet.context_parallel_utils import (
            context_parallel_parameter_disable_scale_grad,
            mark_context_parallel_parameter_disable_scale_grad,
        )

        t = paddle.randn([4, 4])
        mark_context_parallel_parameter_disable_scale_grad(t)
        self.assertTrue(context_parallel_parameter_disable_scale_grad(t))


class TestPreprocessIndex(unittest.TestCase):
    """Tests for preprocess_index function."""

    def test_preprocess_index_basic(self):
        """Test basic preprocess_index."""
        from paddleformers.fleet.context_parallel_utils import preprocess_index

        indices = paddle.to_tensor([10, 20, 30, 40], dtype=paddle.int32)
        result = preprocess_index(
            indices, chunk_id=1, seq_blocksize=16, max_seqlen_q=16
        )
        self.assertEqual(result.shape, indices.shape)

    def test_preprocess_index_clips_min(self):
        """Test preprocess_index clips minimum to 0."""
        from paddleformers.fleet.context_parallel_utils import preprocess_index

        indices = paddle.to_tensor([10, 20, 30, 40], dtype=paddle.int32)
        result = preprocess_index(
            indices, chunk_id=2, seq_blocksize=16, max_seqlen_q=16
        )
        # After subtracting 32, values become [-22, -12, -2, 8], clipped to [0, 0, 0, 8]
        self.assertTrue(paddle.all(result >= 0))

    def test_preprocess_index_clips_max(self):
        """Test preprocess_index clips maximum to max_seqlen_q."""
        from paddleformers.fleet.context_parallel_utils import preprocess_index

        indices = paddle.to_tensor([50, 60, 70, 80], dtype=paddle.int32)
        result = preprocess_index(
            indices, chunk_id=0, seq_blocksize=16, max_seqlen_q=16
        )
        # After clipping, values should be <= max_seqlen_q
        self.assertTrue(paddle.all(result <= 16))


class TestPreprocessIndexDualChunks(unittest.TestCase):
    """Tests for preprocess_index_dual_chunks function."""

    def test_dual_chunks_basic(self):
        """Test basic preprocess_index_dual_chunks."""
        from paddleformers.fleet.context_parallel_utils import (
            preprocess_index_dual_chunks,
        )

        indices = paddle.to_tensor([10, 20, 30, 40], dtype=paddle.int32)
        result = preprocess_index_dual_chunks(
            indices,
            chunk_id_first=0,
            chunk_id_second=3,
            seq_blocksize=16,
            max_seqlen_q=16,
        )
        self.assertEqual(result.shape, indices.shape)

    def test_dual_chunks_output_nonneg(self):
        """Test dual chunks output is non-negative."""
        from paddleformers.fleet.context_parallel_utils import (
            preprocess_index_dual_chunks,
        )

        indices = paddle.to_tensor([10, 20, 30, 40], dtype=paddle.int32)
        result = preprocess_index_dual_chunks(
            indices,
            chunk_id_first=0,
            chunk_id_second=1,
            seq_blocksize=16,
            max_seqlen_q=16,
        )
        self.assertTrue(paddle.all(result >= 0))


if __name__ == "__main__":
    unittest.main()
