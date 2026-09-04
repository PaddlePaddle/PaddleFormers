# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
"""Focused tests for packing=false SFT leftover yield (GLM-4 CI GT path)."""

import inspect
import unittest

from paddleformers.datasets.SFTDataset import BaseSFTDataset


class TestSftNonPackingIteration(unittest.TestCase):
    def test_non_packing_loop_keeps_leftover_yield(self):
        source = inspect.getsource(BaseSFTDataset)
        marker = "Not using packing mode for data iteration."
        start = source.find(marker)
        self.assertGreater(start, 0)
        end = source.find("Using binpacking mode for data iteration.", start)
        self.assertGreater(end, start)
        non_packing = source[start:end]
        self.assertIn("yield batch_sequence", non_packing)
        self.assertIn("if len(batch_sequence) > 0:", non_packing)
        self.assertGreaterEqual(non_packing.count("yield batch_sequence"), 2)
        self.assertIn("self.iter_all_examples = True", non_packing)
