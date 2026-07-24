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
from unittest.mock import patch

import numpy as np
import paddle


class TestRotaryEmbeddingUseAccuracyCompatible(unittest.TestCase):
    """Tests for RotaryEmbedding use_accuracy_compatible inv_freq path."""

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_inv_freq_matches_default_branch(self, mock_ps):
        """The accuracy-compatible CPU inv_freq must match the default
        (GPU/non-compatible) computation numerically."""
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None

        rope_default = RotaryEmbedding(
            head_dim=64,
            rotary_percent=1.0,
            use_accuracy_compatible=False,
        )
        rope_compat = RotaryEmbedding(
            head_dim=64,
            rotary_percent=1.0,
            use_accuracy_compatible=True,
        )

        np.testing.assert_allclose(
            rope_compat.inv_freq.astype("float32").numpy(),
            rope_default.inv_freq.astype("float32").numpy(),
            rtol=1e-6,
            atol=1e-6,
        )

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_inv_freq_shape(self, mock_ps):
        """inv_freq must have head_dim / 2 elements."""
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(
            head_dim=64,
            rotary_percent=1.0,
            use_accuracy_compatible=True,
        )
        self.assertEqual(rope.inv_freq.shape, [32])

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_inv_freq_on_gpu_when_cuda(self, mock_ps):
        """When compiled with CUDA, the accuracy-compatible branch moves
        inv_freq back to GPU after the CPU computation."""
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(
            head_dim=64,
            rotary_percent=1.0,
            use_accuracy_compatible=True,
        )
        if paddle.is_compiled_with_cuda():
            self.assertTrue(rope.inv_freq.place.is_gpu_place())

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_inv_freq_on_cpu_when_not_cuda(self, mock_ps, _mock_cuda):
        """When not compiled with CUDA, the accuracy-compatible branch keeps
        inv_freq on CPU."""
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(
            head_dim=64,
            rotary_percent=1.0,
            use_accuracy_compatible=True,
        )
        self.assertTrue(rope.inv_freq.place.is_cpu_place())

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_partial_rotary_matches_default(self, mock_ps):
        """Accuracy-compatible path must also match default for
        rotary_percent < 1.0 (reduced dim)."""
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        rope_default = RotaryEmbedding(
            head_dim=64,
            rotary_percent=0.5,
            use_accuracy_compatible=False,
        )
        rope_compat = RotaryEmbedding(
            head_dim=64,
            rotary_percent=0.5,
            use_accuracy_compatible=True,
        )
        self.assertEqual(rope_compat.inv_freq.shape, [16])
        np.testing.assert_allclose(
            rope_compat.inv_freq.astype("float32").numpy(),
            rope_default.inv_freq.astype("float32").numpy(),
            rtol=1e-6,
            atol=1e-6,
        )

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_rope_scaling_with_accuracy_compatible(self, mock_ps):
        """rope_scaling on top of the accuracy-compatible inv_freq still
        produces a valid inv_freq matching the default path."""
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        rope_default = RotaryEmbedding(
            head_dim=64,
            rotary_percent=1.0,
            rope_scaling=True,
            rope_scaling_factor=8.0,
            use_accuracy_compatible=False,
        )
        rope_compat = RotaryEmbedding(
            head_dim=64,
            rotary_percent=1.0,
            rope_scaling=True,
            rope_scaling_factor=8.0,
            use_accuracy_compatible=True,
        )
        np.testing.assert_allclose(
            rope_compat.inv_freq.astype("float32").numpy(),
            rope_default.inv_freq.astype("float32").numpy(),
            rtol=1e-5,
            atol=1e-6,
        )


class TestYarnRotaryEmbeddingUseAccuracyCompatible(unittest.TestCase):
    """Tests for YarnRotaryEmbedding passing use_accuracy_compatible through."""

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_yarn_forwards_flag_to_base_inv_freq(self, mock_ps):
        """YarnRotaryEmbedding must forward use_accuracy_compatible to the
        base RotaryEmbedding so the base inv_freq matches the default path."""
        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            YarnRotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        yarn_default = YarnRotaryEmbedding(
            head_dim=64,
            use_accuracy_compatible=False,
        )
        yarn_compat = YarnRotaryEmbedding(
            head_dim=64,
            use_accuracy_compatible=True,
        )
        np.testing.assert_allclose(
            yarn_compat.inv_freq.astype("float32").numpy(),
            yarn_default.inv_freq.astype("float32").numpy(),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
