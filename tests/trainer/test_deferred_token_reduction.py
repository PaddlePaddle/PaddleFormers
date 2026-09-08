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
import unittest
from types import SimpleNamespace

import paddle
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
    DygraphShardingOptimizerV2,
)
from paddlefleet.models.common.language_loss.language_loss import (
    clear_pending_gradient_divisor,
    set_pending_gradient_divisor,
)

from paddleformers.trainer import Trainer


class TestDeferredTokenReduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.set_device(os.environ.get("PADDLE_TEST_DEVICE", "cpu"))

    def tearDown(self):
        clear_pending_gradient_divisor()

    def parameter(self, name, main=True):
        grad = paddle.to_tensor([2.0, 4.0], dtype="float32")
        return SimpleNamespace(name=name, main_grad=grad if main else None, grad=None if main else grad)

    def sharded(self, mapping):
        optimizer = object.__new__(DygraphShardingOptimizerV2)
        optimizer.param2bucket = mapping
        return optimizer

    def bucket(self, degree, use_avg=True):
        return SimpleNamespace(_comm_group=SimpleNamespace(nranks=degree), _use_reduce_avg=use_avg)

    def apply(self, optimizer, parameters):
        trainer = object.__new__(Trainer)
        trainer.optimizer = optimizer
        set_pending_gradient_divisor(4.0)
        trainer._apply_deferred_token_normalization(SimpleNamespace(parameters=lambda: iter(parameters)))

    def values(self, parameter):
        grad = parameter.main_grad if parameter.main_grad is not None else parameter.grad
        return grad.numpy().tolist()

    def test_unsharded_main_and_fallback_grad_keep_token_division(self):
        parameters = [self.parameter("main"), self.parameter("fallback", main=False)]
        self.apply(SimpleNamespace(), parameters)
        self.assertEqual([self.values(p) for p in parameters], [[0.5, 1.0], [0.5, 1.0]])

    def test_per_parameter_avg_and_sum_scale_groups(self):
        for use_avg in (True, False):
            with self.subTest(use_avg=use_avg):
                dense, expert = self.parameter("dense"), self.parameter("expert")
                opt = self.sharded({"dense": [self.bucket(2, use_avg)], "expert": [self.bucket(1, use_avg)]})
                self.apply(SimpleNamespace(_inner_opt=opt), [dense, expert])
                self.assertEqual(self.values(dense), [1.0, 2.0])
                self.assertEqual(self.values(expert), [0.5, 1.0])

    def test_all_mappings_checked_before_first_gradient_changes(self):
        first, missing = self.parameter("first"), self.parameter("missing")
        with self.assertRaisesRegex(RuntimeError, "missing FusedCommBuffer"):
            self.apply(self.sharded({"first": [self.bucket(2)]}), [first, missing])
        self.assertEqual(self.values(first), [2.0, 4.0])
        self.assertEqual(self.values(missing), [2.0, 4.0])

    def test_empty_native_mapping_does_not_use_unsharded_fallback(self):
        parameter = self.parameter("weight")
        with self.assertRaisesRegex(RuntimeError, "missing param2bucket"):
            self.apply(self.sharded({}), [parameter])
        self.assertEqual(self.values(parameter), [2.0, 4.0])

    def test_concrete_wrapper_graph_and_multiple_bucket_degrees(self):
        parameter = self.parameter("weight")
        opt = self.sharded({"weight": [self.bucket(1), self.bucket(2)]})
        wrapper = SimpleNamespace(_inner_opt=SimpleNamespace(), _optimizer=opt)
        wrapper._opt = wrapper
        self.apply(wrapper, [parameter])
        self.assertEqual(self.values(parameter), [1.0, 2.0])

    def test_invalid_degree_does_not_mutate_gradients(self):
        parameter = self.parameter("weight")
        with self.assertRaisesRegex(RuntimeError, "invalid comm group nranks"):
            self.apply(self.sharded({"weight": [self.bucket(0)]}), [parameter])
        self.assertEqual(self.values(parameter), [2.0, 4.0])


if __name__ == "__main__":
    unittest.main()
