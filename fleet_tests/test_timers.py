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

import time
import unittest

import paddle

from paddleformers.fleet.timers import RuntimeTimer
from paddleformers.fleet.training import get_timers
from paddleformers.fleet.training.initialize import initialize_fleet

strategy = paddle.distributed.fleet.DistributedStrategy()
initialize_fleet(strategy=strategy)


class TestTimers(unittest.TestCase):
    def test_timers(self):
        timers = get_timers()

        timers("operation1").start()
        time.sleep(0.1)
        timers("operation1").stop()

        timers("operation2").start()
        time.sleep(0.05)
        timers("operation2").stop()

        timers.log(["operation1", "operation2"])

    def test_runtime_timer(self):
        runtime_timer = RuntimeTimer()

        runtime_timer.start("operation1")
        time.sleep(0.1)
        runtime_timer.stop()
        runtime_timer.log()

        runtime_timer.start("operation2")
        time.sleep(0.05)
        runtime_timer.stop()
        runtime_timer.log()


if __name__ == "__main__":
    unittest.main()
