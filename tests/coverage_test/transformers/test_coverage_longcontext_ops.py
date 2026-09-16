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

from paddleformers.transformers.ernie4_5_moe_vl.model.longcontext_ops import (
    MaxHeap,
    redistribute_tokens,
)


class TestMaxHeap(unittest.TestCase):
    """Tests for MaxHeap class."""

    def test_empty_heap(self):
        heap = MaxHeap()
        self.assertTrue(heap.is_empty())
        self.assertEqual(len(heap), 0)

    def test_push_and_pop(self):
        heap = MaxHeap()
        heap.push((0, 5))
        heap.push((1, 3))
        heap.push((2, 8))
        self.assertFalse(heap.is_empty())
        self.assertEqual(len(heap), 3)
        # Pop should return the max (based on last element)
        item = heap.pop()
        self.assertEqual(item[1], 8)

    def test_top(self):
        heap = MaxHeap([(0, 5), (1, 3), (2, 8)])
        top = heap.top()
        self.assertEqual(top[1], 8)

    def test_pop_empty_raises(self):
        heap = MaxHeap()
        with self.assertRaises(IndexError):
            heap.pop()

    def test_top_empty_raises(self):
        heap = MaxHeap()
        with self.assertRaises(IndexError):
            heap.top()

    def test_order_preserved(self):
        items = [(0, 3), (1, 1), (2, 2)]
        heap = MaxHeap(items)
        result = []
        while not heap.is_empty():
            result.append(heap.pop())
        # Should pop in descending order by last element
        self.assertEqual([r[1] for r in result], [3, 2, 1])


class TestRedistributeTokens(unittest.TestCase):
    """Tests for redistribute_tokens function."""

    def test_balanced_piles(self):
        """When all piles have the same number, no moves needed."""
        piles = [5, 5, 5, 5]
        moves = redistribute_tokens(piles)
        self.assertEqual(len(moves), 0)

    def test_simple_redistribution(self):
        """Test basic token redistribution."""
        piles = [6, 2, 4, 0]
        moves = redistribute_tokens(piles)
        # Total = 12, target = 3 each
        # Pile 0 has surplus of 3, pile 1 has deficit of 1, pile 2 deficit of 1, pile 3 deficit of 3
        total_moved = sum(m.tokens for m in moves)
        self.assertGreater(total_moved, 0)

    def test_single_pile(self):
        """Single pile needs no redistribution."""
        piles = [10]
        moves = redistribute_tokens(piles)
        self.assertEqual(len(moves), 0)

    def test_even_distribution(self):
        """Test with remainders - some piles get extra tokens."""
        piles = [5, 5, 5, 2]
        moves = redistribute_tokens(piles)
        # Total = 17, target per pile = 4, remainder = 1
        self.assertGreater(len(moves), 0)


if __name__ == "__main__":
    unittest.main()
