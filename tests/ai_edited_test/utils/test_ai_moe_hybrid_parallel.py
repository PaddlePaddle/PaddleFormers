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
from unittest.mock import MagicMock, patch

from paddleformers.utils.moe_hybrid_parallel_optimizer import MoEHybridParallelClipGrad


class TestMoEHybridParallelClipGradInit(unittest.TestCase):
    def test_init_basic(self):
        mock_clip = MagicMock()
        mock_clip.clip_norm = 1.0
        mock_hcg = MagicMock()
        mock_hcg.get_moe_sharding_parallel_world_size.return_value = 0
        mock_hcg.get_sharding_parallel_world_size.return_value = 1
        mock_hcg.get_model_parallel_world_size.return_value = 1
        mock_hcg.get_pipe_parallel_world_size.return_value = 1

        clip_grad = MoEHybridParallelClipGrad(mock_clip, mock_hcg)
        self.assertEqual(clip_grad._clip, mock_clip)
        self.assertEqual(clip_grad._hcg, mock_hcg)
        self.assertEqual(clip_grad.processed_steps, 0)

    def test_getattr_delegates_to_clip(self):
        mock_clip = MagicMock()
        mock_clip.clip_norm = 1.0
        mock_clip.some_attr = "test_value"
        mock_hcg = MagicMock()
        mock_hcg.get_moe_sharding_parallel_world_size.return_value = 0
        mock_hcg.get_sharding_parallel_world_size.return_value = 1
        mock_hcg.get_model_parallel_world_size.return_value = 1
        mock_hcg.get_pipe_parallel_world_size.return_value = 1

        clip_grad = MoEHybridParallelClipGrad(mock_clip, mock_hcg)
        self.assertEqual(clip_grad.some_attr, "test_value")

    def test_call_delegates_to_dygraph_clip(self):
        mock_clip = MagicMock()
        mock_clip.clip_norm = 1.0
        mock_hcg = MagicMock()
        mock_hcg.get_moe_sharding_parallel_world_size.return_value = 0
        mock_hcg.get_sharding_parallel_world_size.return_value = 1
        mock_hcg.get_model_parallel_world_size.return_value = 1
        mock_hcg.get_pipe_parallel_world_size.return_value = 1

        clip_grad = MoEHybridParallelClipGrad(mock_clip, mock_hcg)
        with patch.object(clip_grad, "_dygraph_clip", return_value=[]) as mock_dygraph:
            clip_grad([("param", "grad")])
            mock_dygraph.assert_called_once_with([("param", "grad")])


class TestMoEHybridParallelClipGradStat(unittest.TestCase):
    def test_stat_dict_exists(self):
        mock_clip = MagicMock()
        mock_clip.clip_norm = 1.0
        mock_hcg = MagicMock()
        mock_hcg.get_moe_sharding_parallel_world_size.return_value = 0
        mock_hcg.get_sharding_parallel_world_size.return_value = 1
        mock_hcg.get_model_parallel_world_size.return_value = 1
        mock_hcg.get_pipe_parallel_world_size.return_value = 1

        clip_grad = MoEHybridParallelClipGrad(mock_clip, mock_hcg)
        self.assertIsInstance(clip_grad.stat, dict)


class TestMoEHybridParallelOptimizerConstants(unittest.TestCase):
    def test_all_exports(self):
        from paddleformers.utils.moe_hybrid_parallel_optimizer import __all__

        self.assertIn("MoEHybridParallelOptimizer", __all__)


if __name__ == "__main__":
    unittest.main()
