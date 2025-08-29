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
import sys
import unittest
from pathlib import Path

import paddle
import paddle.distributed.fleet.meta_parallel as mpu
import paddle.nn as nn
from paddle.distributed import fleet
from paddle.incubate.nn import FusedLinear

from paddleformers.nn.linear import Linear  # Replace with your actual module path
from paddleformers.transformers import LlamaConfig
from tests.parallel_launch import TestMultipleGpus
from tests.testing_utils import require_paddle_at_least_2_gpu

sys.path.append(str(Path(__file__).parent.parent))

tp_size = paddle.distributed.get_world_size()
tp_rank = 0
if tp_size > 1:
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": tp_size,
        "pp_degree": 1,
        "sharding_degree": 1,
    }
    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()
    tp_rank = hcg.get_model_parallel_rank()
    mp_group = hcg.get_model_parallel_group()


def _test_create_parallel_linear(config):
    config.in_features = 100
    config.out_features = 100
    # Test creating column parallel linear
    linear = Linear.create(
        in_features=config.in_features,
        out_features=config.out_features,
        linear_type="colwise",
        gather_output=True,
    )

    assert isinstance(linear, mpu.ColumnParallelLinear)

    # Test creating row parallel linear
    linear = Linear.create(
        in_features=config.in_features,
        out_features=config.out_features,
        linear_type="rowwise",
        input_is_parallel=False,
    )

    assert isinstance(linear, mpu.RowParallelLinear)
    config.sequence_parallel = False
    # Test createing squence row parallel linear
    linear = Linear.create(
        in_features=config.in_features,
        out_features=config.out_features,
        linear_type="sequence_rowwise",
        input_is_parallel=True,
    )
    assert isinstance(linear, fleet.utils.sequence_parallel_utils.RowSequenceParallelLinear)
    linear = Linear.create(
        in_features=config.in_features,
        out_features=config.out_features,
        linear_type="sequence_colwise",
        gather_output=False,
    )

    assert isinstance(linear, fleet.utils.sequence_parallel_utils.ColumnSequenceParallelLinear)
    print("paddleformers.nn.Linear: _test_create_parallel_linear passed")


@require_paddle_at_least_2_gpu
class TestLinear(TestMultipleGpus):
    def setUp(self):
        super().setUp()
        self.config = LlamaConfig()
        self.in_features = 10
        self.out_features = 10

    # def test_create_tensor_parallel_linear(self):
    #     self.run_2gpu(__file__)

    def test_create_default_linear(self):
        linear = Linear.create(in_features=self.in_features, out_features=self.out_features, linear_type="default")
        self.assertIsInstance(linear, nn.Linear)

    def test_create_fused_linear(self):
        # Test creating fused linear layer
        linear = Linear.create(in_features=self.in_features, out_features=self.out_features, linear_type="fuse_linear")

        self.assertIsInstance(linear, FusedLinear)

    def test_get_linear_type_default(self):
        # Test linear type detection for default case
        self.config.tensor_parallel_degree = 1
        linear_type = Linear.get_linear_type(self.config)
        self.assertEqual(linear_type, "default")

    def test_get_linear_type_fused(self):
        # Test fused linear type detection
        self.config.tensor_parallel_degree = 1
        self.config.fuse_linear = True
        linear_type = Linear.get_linear_type(self.config)
        self.assertEqual(linear_type, "fuse_linear")

    def test_get_linear_type_parallel(self):
        # Test parallel linear type detection
        self.config.tensor_parallel_degree = 2
        # Test column parallel
        col_type = Linear.get_linear_type(self.config, tp_plan="colwise")
        self.assertEqual(col_type, "colwise")
        # Test row parallel
        row_type = Linear.get_linear_type(self.config, tp_plan="rowwise")
        self.assertEqual(row_type, "rowwise")

    def test_get_linear_kwargs(self):
        col_kwargs = Linear.get_linear_kwargs("colwise", gather_output=True)
        self.assertEqual(col_kwargs["gather_output"], True)

        row_kwargs = Linear.get_linear_kwargs("rowwise", input_is_parallel=False)
        self.assertEqual(row_kwargs["input_is_parallel"], False)


if __name__ == "__main__":
    # config = LlamaConfig()
    # config.tensor_parallel_degree = tp_size
    # _test_create_parallel_linear(config)
    unittest.main()
