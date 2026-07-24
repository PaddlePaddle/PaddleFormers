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


class TestRefinedRcomputeFlashAttentionInit(unittest.TestCase):
    """Tests for RefinedRcomputeFlashAttention initialization."""

    def test_init_creates_queue(self):
        """Test initialization creates a queue."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashAttention,
        )

        rrfa = RefinedRcomputeFlashAttention()
        self.assertIsNotNone(rrfa._hold_tensors_queue)
        self.assertTrue(rrfa._hold_tensors_queue.empty())

    def test_is_callable(self):
        """Test RefinedRcomputeFlashAttention is callable."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashAttention,
        )

        rrfa = RefinedRcomputeFlashAttention()
        self.assertTrue(callable(rrfa))


class TestRefinedRcomputeFlashMaskAttentionInit(unittest.TestCase):
    """Tests for RefinedRcomputeFlashMaskAttention initialization."""

    def test_init_creates_queue(self):
        """Test initialization creates a queue."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskAttention,
        )

        rrfma = RefinedRcomputeFlashMaskAttention()
        self.assertIsNotNone(rrfma._hold_tensors_queue)
        self.assertTrue(rrfma._hold_tensors_queue.empty())

    def test_is_callable(self):
        """Test RefinedRcomputeFlashMaskAttention is callable."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskAttention,
        )

        rrfma = RefinedRcomputeFlashMaskAttention()
        self.assertTrue(callable(rrfma))


class TestRefinedRcomputeFlashMaskCpAttentionInit(unittest.TestCase):
    """Tests for RefinedRcomputeFlashMaskCpAttention initialization."""

    def test_init_creates_queue(self):
        """Test initialization creates a queue."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskCpAttention,
        )

        rrfmcpa = RefinedRcomputeFlashMaskCpAttention()
        self.assertIsNotNone(rrfmcpa._hold_tensors_queue)
        self.assertTrue(rrfmcpa._hold_tensors_queue.empty())


class TestFlashAttnFunctorStructure(unittest.TestCase):
    """Tests for FlashAttnFunctor class structure."""

    def test_is_pylayer(self):
        """Test FlashAttnFunctor is a PyLayer subclass."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            FlashAttnFunctor,
        )

        self.assertTrue(issubclass(FlashAttnFunctor, paddle.autograd.PyLayer))

    def test_has_forward(self):
        """Test FlashAttnFunctor has forward method."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            FlashAttnFunctor,
        )

        self.assertTrue(hasattr(FlashAttnFunctor, "forward"))

    def test_has_backward(self):
        """Test FlashAttnFunctor has backward method."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            FlashAttnFunctor,
        )

        self.assertTrue(hasattr(FlashAttnFunctor, "backward"))


class TestFlashMaskAttnFunctorStructure(unittest.TestCase):
    """Tests for FlashMaskAttnFunctor class structure."""

    def test_is_pylayer(self):
        """Test FlashMaskAttnFunctor is a PyLayer subclass."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            FlashMaskAttnFunctor,
        )

        self.assertTrue(
            issubclass(FlashMaskAttnFunctor, paddle.autograd.PyLayer)
        )

    def test_has_forward(self):
        """Test FlashMaskAttnFunctor has forward method."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            FlashMaskAttnFunctor,
        )

        self.assertTrue(hasattr(FlashMaskAttnFunctor, "forward"))

    def test_has_backward(self):
        """Test FlashMaskAttnFunctor has backward method."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            FlashMaskAttnFunctor,
        )

        self.assertTrue(hasattr(FlashMaskAttnFunctor, "backward"))


if __name__ == "__main__":
    unittest.main()
