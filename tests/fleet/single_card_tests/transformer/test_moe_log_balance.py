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

import paddle

from paddleformers.fleet.training.global_vars import (
    get_global_training_logs,
    set_global_training_logs,
    set_profile_timers,
    unset_global_variables,
)
from paddleformers.fleet.transformer.moe import moe_utils
from paddleformers.fleet.transformer.moe.moe_utils import (
    _all_gather_local_tokens,
    global_moe_balance_training_logs_enabled,
    log_moe_balance,
    log_moe_losses,
)
from paddleformers.fleet.transformer.utils import profile


class DummyLogs(dict):
    def is_moe_balance_logs_enabled(self):
        return True


class DummyDisabledLogs(dict):
    def is_moe_balance_logs_enabled(self):
        return False


class TestMoeBalanceLogging(unittest.TestCase):
    def setUp(self):
        unset_global_variables()
        set_global_training_logs(DummyLogs())

    def tearDown(self):
        unset_global_variables()

    def test_alltoall_tensor_tokens_per_expert(self):
        tokens_per_expert = paddle.to_tensor([2, 4, 6, 8], dtype="int64")

        gathered = _all_gather_local_tokens(tokens_per_expert, None)
        self.assertEqual(gathered.dtype, paddle.int64)
        self.assertEqual(list(gathered.shape), [1, 4])

        log_moe_balance(
            layer_number=0,
            moe_group=None,
            num_experts_per_tok=2,
            tokens_per_expert=tokens_per_expert,
        )
        logs = get_global_training_logs()

        self.assertAlmostEqual(logs["tokens_per_expert_layer_0_mean"], 5.0)
        self.assertAlmostEqual(logs["tokens_per_expert_layer_0_max"], 8.0)
        self.assertAlmostEqual(logs["tokens_per_expert_layer_0_max_mean_ratio"], 1.6)
        self.assertAlmostEqual(logs["tokens_per_expert_avg_layer_0_mean"], 0.5)
        self.assertAlmostEqual(logs["local_tokens_per_card_layer_0_mean"], 20.0)
        self.assertAlmostEqual(logs["local_tokens_per_card_layer_0_max_mean_ratio"], 1.0)

    def test_deepep_python_int_list_tokens_per_expert(self):
        tokens_per_expert = [3, 1, 5, 7]

        captured = {}
        orig_all_gather_local_tokens = moe_utils._all_gather_local_tokens

        def wrapped_all_gather_local_tokens(local_tokens_per_expert, group):
            captured["is_tensor"] = isinstance(local_tokens_per_expert, paddle.Tensor)
            captured["is_cpu"] = local_tokens_per_expert.place.is_cpu_place()
            captured["dtype"] = local_tokens_per_expert.dtype
            captured["shape"] = list(local_tokens_per_expert.shape)
            return orig_all_gather_local_tokens(local_tokens_per_expert, group)

        moe_utils._all_gather_local_tokens = wrapped_all_gather_local_tokens

        try:
            log_moe_balance(
                layer_number=1,
                moe_group=None,
                num_experts_per_tok=4,
                tokens_per_expert=tokens_per_expert,
            )
        finally:
            moe_utils._all_gather_local_tokens = orig_all_gather_local_tokens

        self.assertTrue(captured["is_tensor"])
        self.assertTrue(captured["is_cpu"])
        self.assertEqual(captured["dtype"], paddle.int64)
        self.assertEqual(captured["shape"], [4])

        logs = get_global_training_logs()

        self.assertAlmostEqual(logs["tokens_per_expert_layer_1_mean"], 4.0)
        self.assertAlmostEqual(logs["tokens_per_expert_layer_1_min"], 1.0)
        self.assertAlmostEqual(logs["tokens_per_expert_layer_1_max"], 7.0)
        self.assertAlmostEqual(logs["tokens_per_expert_layer_1_max_mean_ratio"], 1.75)
        self.assertAlmostEqual(logs["tokens_per_expert_avg_layer_1_mean"], 1.0)
        self.assertAlmostEqual(logs["local_tokens_per_card_layer_1_mean"], 16.0)
        self.assertAlmostEqual(logs["local_tokens_per_card_layer_1_max_mean_ratio"], 1.0)

    def test_logs_object_enable_balance_gate(self):
        self.assertTrue(global_moe_balance_training_logs_enabled())

    def test_log_moe_losses_logs_non_none_values(self):
        aux_loss = paddle.to_tensor(1.25, dtype="float32")
        z_loss = paddle.to_tensor(2.5, dtype="float32")

        log_moe_losses(layer_number=3, aux_loss=aux_loss, z_loss=z_loss)
        logs = get_global_training_logs()

        self.assertAlmostEqual(logs["aux_loss"].item(), 1.25)
        self.assertAlmostEqual(logs["aux_loss_layer_3"].item(), 1.25)
        self.assertAlmostEqual(logs["zloss"].item(), 2.5)
        self.assertAlmostEqual(logs["zloss_layer_3"].item(), 2.5)

    def test_log_moe_losses_skips_none_values(self):
        z_loss = paddle.to_tensor(3.5, dtype="float32")

        log_moe_losses(layer_number=5, aux_loss=None, z_loss=z_loss)
        logs = get_global_training_logs()

        self.assertNotIn("aux_loss", logs)
        self.assertNotIn("aux_loss_layer_5", logs)
        self.assertAlmostEqual(logs["zloss"].item(), 3.5)
        self.assertAlmostEqual(logs["zloss_layer_5"].item(), 3.5)

    def test_log_moe_losses_without_layer_number_only_logs_global_keys(self):
        aux_loss = paddle.to_tensor(1.0, dtype="float32")
        z_loss = paddle.to_tensor(2.0, dtype="float32")

        log_moe_losses(layer_number=None, aux_loss=aux_loss, z_loss=z_loss)
        logs = get_global_training_logs()

        self.assertAlmostEqual(logs["aux_loss"].item(), 1.0)
        self.assertAlmostEqual(logs["zloss"].item(), 2.0)
        self.assertNotIn("aux_loss_layer_None", logs)
        self.assertNotIn("zloss_layer_None", logs)

    def test_log_moe_losses_respects_balance_gate(self):
        unset_global_variables()
        set_global_training_logs(DummyDisabledLogs())

        log_moe_losses(
            layer_number=1,
            aux_loss=paddle.to_tensor(1.0, dtype="float32"),
            z_loss=paddle.to_tensor(2.0, dtype="float32"),
        )

        self.assertEqual(get_global_training_logs(), {})

    def test_profile_stops_timer_on_exception(self):
        events = []

        class DummyTimer:
            def start(self):
                events.append("start")

            def stop(self):
                events.append("stop")

        class DummyTimers:
            def __call__(self, name, use_event=True):
                events.append(("timer", name, use_event))
                return DummyTimer()

        unset_global_variables()
        set_profile_timers(DummyTimers())

        with self.assertRaisesRegex(RuntimeError, "boom"), profile("attn"):
            raise RuntimeError("boom")

        self.assertEqual(
            events,
            [("timer", "attn", True), "start", "stop"],
        )


if __name__ == "__main__":
    unittest.main()
