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


# Tests for src/paddleformers.fleet/distributed/model.py

import unittest
from unittest import mock


class TestDistributedModel(unittest.TestCase):
    """Tests for the distributed_model function."""

    def _make_mock_hcg(
        self, parallel_mode="PIPELINE_PARALLEL", num_virtual_stages=1
    ):
        """Helper to create a mock HCG object."""
        from paddle.distributed.fleet.base.topology import ParallelMode

        hcg = mock.MagicMock()
        mode = getattr(ParallelMode, parallel_mode, parallel_mode)
        hcg.get_parallel_mode.return_value = mode
        pp_group = mock.MagicMock()
        pp_group.nranks = 2
        hcg.get_pipe_parallel_world_size.return_value = 2
        hcg._hcg = hcg
        return hcg

    def _make_mock_strategy(
        self,
        amp=False,
        use_pure_fp16=False,
        use_pure_bf16=False,
        use_dualpipev=False,
        best_unbalanced_scheduler=False,
        accumulate_steps=4,
        pipeline_configs=None,
    ):
        """Helper to create a mock strategy object."""
        strategy = mock.MagicMock()
        strategy.amp = amp
        if amp:
            strategy.amp_configs = {
                "use_pure_fp16": use_pure_fp16,
                "use_pure_bf16": use_pure_bf16,
                "init_loss_scaling": 1.0,
                "incr_ratio": 2.0,
                "decr_ratio": 0.5,
                "incr_every_n_steps": 1000,
                "decr_every_n_nan_or_inf": 2,
                "use_dynamic_loss_scaling": True,
            }
        if pipeline_configs is None:
            pipeline_configs = {}
        strategy.pipeline_configs = {
            "accumulate_steps": accumulate_steps,
            "micro_batch_size": 1,
            **pipeline_configs,
        }
        strategy.hybrid_configs = {
            "pp_configs": mock.MagicMock(
                use_dualpipev=use_dualpipev,
                best_unbalanced_scheduler=best_unbalanced_scheduler,
            )
        }
        return strategy

    # TODO(hushenwei2000): enable this test after migrate to paddle pp
    # def test_distributed_model_single_rank(self):
    #     """Test distributed_model returns NoPipelineParallel when world_size <= 1."""
    #     from paddle.distributed.fleet import distributed_model

    #     mock_model = mock.MagicMock()
    #     mock_fleet = mock.MagicMock()
    #     mock_strategy = self._make_mock_strategy()

    #     with (
    #         mock.patch("paddle.distributed.get_world_size", return_value=1),
    #         mock.patch("paddle.distributed.fleet.fleet", mock_fleet),
    #     ):
    #         mock_fleet._user_defined_strategy = mock_strategy
    #         with mock.patch(
    #             "paddle.distributed.fleet.meta_parallel.NoPipelineParallel"
    #         ) as mock_nopp:
    #             mock_nopp.return_value = mock_model
    #             result = distributed_model(mock_model)
    #             mock_nopp.assert_called_once()

    def test_distributed_model_not_pipeline_layer_raises(self):
        """Test that non-PipelineLayer model raises AssertionError in pipeline mode."""

        from paddle.distributed.fleet import distributed_model
        from paddle.distributed.fleet.meta_parallel import PipelineLayer

        # Create a regular object that is NOT a PipelineLayer
        mock_model = mock.MagicMock(spec=["get_num_virtual_stages"])
        mock_model.get_num_virtual_stages.return_value = 1
        mock_fleet = mock.MagicMock()
        mock_strategy = self._make_mock_strategy()
        mock_hcg = self._make_mock_hcg("PIPELINE_PARALLEL")

        # Make the model not an instance of PipelineLayer
        with (
            mock.patch("paddle.distributed.get_world_size", return_value=4),
            mock.patch("paddle.distributed.fleet.fleet", mock_fleet),
        ):
            mock_fleet._user_defined_strategy = mock_strategy
            mock_fleet._hcg = mock_hcg
            with (
                mock.patch(
                    "paddle.distributed.fleet.meta_parallel.PipelineLayer",
                    PipelineLayer,
                ),
                mock.patch.object(
                    PipelineLayer, "__instancecheck__", return_value=False
                ),
            ):
                # Not a PipelineLayer instance
                with self.assertRaises(AssertionError):
                    distributed_model(mock_model)
