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
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        )
    ),
)

import random
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet import distributed_model
from paddle.distributed.fleet.meta_parallel import (
    NoPipelineParallel,
    build_spec_layer,
)

from tests.fleet.multi_card_tests.pipeline_parallel.test_distribute_model import (
    get_simple_spec,
)

batch_size = 8
micro_batch_size = 2


class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples=40, shape=(64, 256)):
        self.num_samples = num_samples
        self.shape = shape

    def __getitem__(self, idx):
        img = np.random.rand(*self.shape).astype("float32")
        label = np.random.randint(0, 10, size=(64,), dtype="int64")
        return img, label

    def __len__(self):
        return self.num_samples


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


class TestDistVppTraining(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 4
        self.num_virtual_pipeline_stages = 2
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": self.pipeline_parallel_size,
            "pp_configs": {"best_unbalanced_scheduler": True},
        }
        strategy.pipeline_configs = {
            "accumulate_steps": batch_size // micro_batch_size,
            "micro_batch_size": micro_batch_size,
        }
        strategy.hybrid_configs["pp_configs"].sync_moment = True
        strategy.hybrid_configs["pp_configs"].sync_param = True
        self.strategy = strategy
        fleet.init(is_collective=True, strategy=strategy)

    def test_vpp_model(self):
        hcg = fleet.get_hybrid_communicate_group()

        set_random_seed(1024)
        simple_spec = get_simple_spec()
        nopp_model = build_spec_layer(simple_spec, num_stages=1)
        nopp_model = NoPipelineParallel(nopp_model, self.strategy)
        nopp_scheduler = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2, 3, 4], values=[0.01, 0.02, 0.03, 0.04], verbose=True
        )
        nopp_optimizer = paddle.optimizer.SGD(
            learning_rate=nopp_scheduler,
            parameters=nopp_model.parameters(),
        )

        seg_method = "layer:Linear"
        vpp_model = build_spec_layer(
            simple_spec,
            topology=hcg.topology(),
            seg_method=seg_method,
            num_stages=self.pipeline_parallel_size,
            num_virtual_pipeline_stages=self.num_virtual_pipeline_stages,
        )

        vpp_scheduler = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2, 3, 4], values=[0.01, 0.02, 0.03, 0.04], verbose=True
        )
        vpp_optimizer = paddle.optimizer.SGD(
            learning_rate=vpp_scheduler, parameters=vpp_model.parameters()
        )
        vpp_model = distributed_model(vpp_model)
        vpp_optimizer = fleet.distributed_optimizer(vpp_optimizer)

        layer_name_proj = {
            "_layers.shared_layers.shared.shared_net.weight": "_layers.shared_layers.shared.shared_net.weight",
            "_layers.shared_layers.shared.shared_net.bias": "_layers.shared_layers.shared.shared_net.bias",
            "_layers.1.0.weight": "_layers.1.weight",
            "_layers.1.0.bias": "_layers.1.bias",
            "_layers.2.0.weight": "_layers.2.weight",
            "_layers.2.0.bias": "_layers.2.bias",
            "_layers.3.0.weight": "_layers.3.weight",
            "_layers.3.0.bias": "_layers.3.bias",
            "_layers.4.0.weight": "_layers.4.weight",
            "_layers.4.0.bias": "_layers.4.bias",
            "_layers.5.0.weight": "_layers.5.weight",
            "_layers.5.0.bias": "_layers.5.bias",
            "_layers.6.0.weight": "_layers.6.weight",
            "_layers.6.0.bias": "_layers.6.bias",
            "_layers.7.2.classify_net.weight": "_layers.9.classify_net.weight",
            "_layers.7.2.classify_net.bias": "_layers.9.classify_net.bias",
        }

        nopp_model_param = {}
        for name, param in nopp_model.named_parameters():
            nopp_model_param[name] = param

        for name, param in vpp_model.named_parameters():
            param.set_value(nopp_model_param[layer_name_proj[name]])

        train_loader = paddle.io.DataLoader(
            RandomDataset(),
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=0,
        )

        for step_id, data in enumerate(train_loader()):
            img = paddle.to_tensor(data[0])
            label = paddle.to_tensor(data[1])
            img.stop_gradient = True
            label.stop_gradient = True

            nopp_loss = nopp_model.train_batch(
                [img, label], nopp_optimizer, nopp_scheduler
            )

            vpp_loss = vpp_model.train_batch(
                [img, label], vpp_optimizer, vpp_scheduler
            )

            print("loss:", nopp_loss.numpy(), vpp_loss.numpy())
            np.testing.assert_allclose(
                nopp_loss.numpy(), vpp_loss.numpy(), rtol=1e-6, atol=1e-8
            )


if __name__ == "__main__":
    unittest.main()
