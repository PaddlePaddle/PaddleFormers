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

from paddle.distributed.fleet.meta_parallel.parallel_layers.pp_layers import (
    PipelineLayerChunk,
    SegmentLayers,
)


class TestScheduleChunk(unittest.TestCase):
    """Tests for ScheduleChunk."""


class TestSegmentLayers(unittest.TestCase):
    """Tests for SegmentLayers."""

    def test_segment_layers_creation(self):
        layer = SegmentLayers(layers_desc=[], num_parts=0)
        self.assertIsNotNone(layer)


class TestPipelineLayerChunk(unittest.TestCase):
    """Tests for PipelineLayerChunk."""

    def test_pipeline_layer_chunk_creation(self):
        chunk = PipelineLayerChunk()
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.run_function, [])

    def test_pipeline_layer_chunk_append(self):
        from paddle import nn

        chunk = PipelineLayerChunk()
        layer = nn.Linear(10, 5)
        chunk.append(layer)
        self.assertEqual(len(chunk.run_function), 1)

    def test_pipeline_layer_chunk_iter(self):
        from paddle import nn

        chunk = PipelineLayerChunk()
        layer = nn.Linear(10, 5)
        chunk.append(layer)
        items = list(chunk)
        self.assertEqual(len(items), 1)


class TestBuildLayer(unittest.TestCase):
    """Tests for build_spec_layer."""

    def test_build_layer_with_class(self):
        from paddle import nn
        from paddle.distributed.fleet.meta_parallel import build_spec_layer

        layer = build_spec_layer(nn.Linear, 10, 5)
        self.assertIsInstance(layer, nn.Linear)
