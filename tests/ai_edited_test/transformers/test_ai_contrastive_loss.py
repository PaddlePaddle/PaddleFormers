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


class TestSimpleContrastiveLoss(unittest.TestCase):
    """Tests for SimpleContrastiveLoss."""

    def test_default_temperature(self):
        """Test SimpleContrastiveLoss with default temperature."""
        from paddleformers.transformers.contrastive_loss import SimpleContrastiveLoss

        loss_fn = SimpleContrastiveLoss()
        self.assertEqual(loss_fn.embedding_temperature, 0.02)

    def test_custom_temperature(self):
        """Test SimpleContrastiveLoss with a custom temperature."""
        from paddleformers.transformers.contrastive_loss import SimpleContrastiveLoss

        loss_fn = SimpleContrastiveLoss(embedding_temperature=0.1)
        self.assertEqual(loss_fn.embedding_temperature, 0.1)

    def test_forward_basic(self):
        """Test forward pass with equal-sized q_reps and p_reps (group_size=1)."""
        from paddleformers.transformers.contrastive_loss import SimpleContrastiveLoss

        loss_fn = SimpleContrastiveLoss()
        # batch_size=4, embed_dim=8
        q_reps = paddle.randn([4, 8], dtype="float32")
        p_reps = paddle.randn([4, 8], dtype="float32")
        loss = loss_fn(q_reps, p_reps)
        # loss should be a scalar tensor
        self.assertTrue(paddle.is_tensor(loss))
        self.assertEqual(loss.shape, [])

    def test_forward_with_group_size(self):
        """Test forward pass with group_size > 1 (p_reps larger than q_reps)."""
        from paddleformers.transformers.contrastive_loss import SimpleContrastiveLoss

        loss_fn = SimpleContrastiveLoss()
        # batch_size=2, group_size=3, embed_dim=8
        q_reps = paddle.randn([2, 8], dtype="float32")
        p_reps = paddle.randn([6, 8], dtype="float32")
        loss = loss_fn(q_reps, p_reps)
        self.assertTrue(paddle.is_tensor(loss))
        self.assertEqual(loss.shape, [])

    def test_forward_with_group_size_four(self):
        """Test forward pass with group_size=4."""
        from paddleformers.transformers.contrastive_loss import SimpleContrastiveLoss

        loss_fn = SimpleContrastiveLoss(embedding_temperature=0.05)
        # batch_size=3, group_size=4, embed_dim=16
        q_reps = paddle.randn([3, 16], dtype="float32")
        p_reps = paddle.randn([12, 16], dtype="float32")
        loss = loss_fn(q_reps, p_reps)
        self.assertTrue(paddle.is_tensor(loss))
        self.assertTrue(loss.numpy().item() >= 0)

    def test_forward_deterministic(self):
        """Test forward is deterministic with the same inputs."""
        from paddleformers.transformers.contrastive_loss import SimpleContrastiveLoss

        loss_fn = SimpleContrastiveLoss()
        paddle.seed(42)
        q_reps = paddle.randn([2, 4], dtype="float32")
        p_reps = paddle.randn([2, 4], dtype="float32")
        loss1 = loss_fn(q_reps, p_reps)
        loss2 = loss_fn(q_reps, p_reps)
        np.testing.assert_allclose(loss1.numpy(), loss2.numpy(), rtol=1e-6)

    def test_forward_positive_loss(self):
        """Test that contrastive loss is non-negative."""
        from paddleformers.transformers.contrastive_loss import SimpleContrastiveLoss

        loss_fn = SimpleContrastiveLoss()
        q_reps = paddle.randn([4, 8], dtype="float32")
        p_reps = paddle.randn([4, 8], dtype="float32")
        loss = loss_fn(q_reps, p_reps)
        self.assertGreaterEqual(loss.numpy().item(), 0.0)


class TestMatryoshkaContrastiveLoss(unittest.TestCase):
    """Tests for MatryoshkaContrastiveLoss."""

    def test_default_init(self):
        """Test default initialization with no matryoshka dims."""
        from paddleformers.transformers.contrastive_loss import (
            MatryoshkaContrastiveLoss,
        )

        loss_fn = MatryoshkaContrastiveLoss()
        self.assertEqual(loss_fn.embedding_temperature, 0.02)
        self.assertEqual(loss_fn.embedding_matryoshka_dims, [])

    def test_custom_matryoshka_dims(self):
        """Test initialization with custom matryoshka dims."""
        from paddleformers.transformers.contrastive_loss import (
            MatryoshkaContrastiveLoss,
        )

        dims = [128, 256, 512]
        loss_fn = MatryoshkaContrastiveLoss(embedding_matryoshka_dims=dims)
        self.assertEqual(loss_fn.embedding_matryoshka_dims, dims)

    def test_forward_no_matryoshka_dims(self):
        """Test forward when no matryoshka dims specified (falls back to SimpleContrastiveLoss)."""
        from paddleformers.transformers.contrastive_loss import (
            MatryoshkaContrastiveLoss,
        )

        loss_fn = MatryoshkaContrastiveLoss(embedding_temperature=0.02, embedding_matryoshka_dims=None)
        q_reps = paddle.randn([4, 16], dtype="float32")
        p_reps = paddle.randn([4, 16], dtype="float32")
        loss = loss_fn(q_reps, p_reps)
        self.assertTrue(paddle.is_tensor(loss))
        self.assertEqual(loss.shape, [])

    def test_forward_with_matryoshka_dims(self):
        """Test forward with matryoshka dims (reduced dimension computation)."""
        from paddleformers.transformers.contrastive_loss import (
            MatryoshkaContrastiveLoss,
        )

        dims = [64, 128, 256]
        loss_fn = MatryoshkaContrastiveLoss(embedding_matryoshka_dims=dims, embedding_temperature=0.02)
        # embed_dim must be at least as large as max dim
        q_reps = paddle.randn([4, 512], dtype="float32")
        p_reps = paddle.randn([4, 512], dtype="float32")
        loss = loss_fn(q_reps, p_reps)
        self.assertTrue(paddle.is_tensor(loss))
        self.assertEqual(loss.shape, [])

    def test_forward_with_matryoshka_dims_group_size(self):
        """Test forward with matryoshka dims and group_size > 1."""
        from paddleformers.transformers.contrastive_loss import (
            MatryoshkaContrastiveLoss,
        )

        dims = [32, 64]
        loss_fn = MatryoshkaContrastiveLoss(embedding_matryoshka_dims=dims)
        q_reps = paddle.randn([2, 128], dtype="float32")
        p_reps = paddle.randn([6, 128], dtype="float32")
        loss = loss_fn(q_reps, p_reps)
        self.assertTrue(paddle.is_tensor(loss))
        self.assertEqual(loss.shape, [])

    def test_forward_empty_dims_list(self):
        """Test forward with empty dims list (should behave like no dims)."""
        from paddleformers.transformers.contrastive_loss import (
            MatryoshkaContrastiveLoss,
        )

        loss_fn = MatryoshkaContrastiveLoss(embedding_matryoshka_dims=[])
        q_reps = paddle.randn([3, 8], dtype="float32")
        p_reps = paddle.randn([3, 8], dtype="float32")
        loss = loss_fn(q_reps, p_reps)
        self.assertTrue(paddle.is_tensor(loss))

    def test_forward_positive_loss_with_dims(self):
        """Test that loss with matryoshka dims is non-negative."""
        from paddleformers.transformers.contrastive_loss import (
            MatryoshkaContrastiveLoss,
        )

        loss_fn = MatryoshkaContrastiveLoss(embedding_matryoshka_dims=[64, 128])
        q_reps = paddle.randn([4, 256], dtype="float32")
        p_reps = paddle.randn([4, 256], dtype="float32")
        loss = loss_fn(q_reps, p_reps)
        self.assertGreaterEqual(loss.numpy().item(), 0.0)


class TestSimpleInfclLoss(unittest.TestCase):
    """Tests for SimpleInfclLoss."""

    def test_default_init(self):
        """Test default initialization."""
        from paddleformers.transformers.contrastive_loss import SimpleInfclLoss

        loss_fn = SimpleInfclLoss()
        self.assertEqual(loss_fn.head_dim, 64)

    def test_custom_head_dim(self):
        """Test initialization with custom head_dim."""
        from paddleformers.transformers.contrastive_loss import SimpleInfclLoss

        loss_fn = SimpleInfclLoss(inf_cl_head_dim=128)
        self.assertEqual(loss_fn.head_dim, 128)

    def test_forward_import_error(self):
        """Test that forward raises ImportError when paddleformers_kernel is not available."""
        from paddleformers.transformers.contrastive_loss import SimpleInfclLoss

        loss_fn = SimpleInfclLoss()
        q_reps = paddle.randn([4, 16], dtype="float32")
        p_reps = paddle.randn([4, 16], dtype="float32")
        with self.assertRaises(ImportError) as ctx:
            loss_fn(q_reps, p_reps)
        self.assertIn("Paddlenlp_kernels", str(ctx.exception))


class TestMatryoshkaInfclLoss(unittest.TestCase):
    """Tests for MatryoshkaInfclLoss."""

    def test_default_init(self):
        """Test default initialization."""
        from paddleformers.transformers.contrastive_loss import MatryoshkaInfclLoss

        loss_fn = MatryoshkaInfclLoss()
        self.assertEqual(loss_fn.embedding_matryoshka_dims, [])

    def test_custom_dims(self):
        """Test initialization with custom dims."""
        from paddleformers.transformers.contrastive_loss import MatryoshkaInfclLoss

        dims = [128, 256]
        loss_fn = MatryoshkaInfclLoss(embedding_matryoshka_dims=dims, inf_cl_head_dim=32)
        self.assertEqual(loss_fn.embedding_matryoshka_dims, dims)

    def test_custom_head_dim(self):
        """Test initialization with custom head_dim."""
        from paddleformers.transformers.contrastive_loss import MatryoshkaInfclLoss

        loss_fn = MatryoshkaInfclLoss(inf_cl_head_dim=128)
        self.assertEqual(loss_fn.loss_fn.head_dim, 128)

    def test_none_dims_becomes_empty_list(self):
        """Test that None dims becomes an empty list."""
        from paddleformers.transformers.contrastive_loss import MatryoshkaInfclLoss

        loss_fn = MatryoshkaInfclLoss(embedding_matryoshka_dims=None)
        self.assertEqual(loss_fn.embedding_matryoshka_dims, [])
