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

from __future__ import annotations

import time

import paddle

try:
    from paddle.distributed.fleet.utils.timer_helper import _GPUEventTimer
except ImportError:
    _GPUEventTimer = None

try:
    import wandb
except ImportError:
    wandb = None

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


class _Timer:
    """Profile Timer for recording time taken by forward/ backward/ reduce/ step."""

    def __init__(self, name: str):
        self.name = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self):
        """Start the timer."""
        assert not self.started_, f"{self.name} timer has already started"
        if "cpu" not in paddle.device.get_device():
            paddle.device.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timers."""
        assert self.started_, f"{self.name} timer is not started."
        if "cpu" not in paddle.device.get_device():
            paddle.device.synchronize()
        self.elapsed_ += time.time() - self.start_time
        self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset: bool = True) -> float:
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


if _GPUEventTimer is None:
    _GPUEventTimer = _Timer


class RuntimeTimer:
    """A timer that can be dynamically adjusted during runtime."""

    def __init__(self):
        self.timer = _Timer("RuntimeTimer")

    def start(self, name: str):
        """Start the RuntimeTimer."""
        self.timer.name = name
        self.timer.start()

    def stop(self):
        """Stop the RuntimeTimer."""
        self.timer.stop()

    def log(self):
        """Log, stop and reset the RuntimeTimer."""
        runtime = self.timer.elapsed(reset=True)
        if self.timer.started_ is True:
            self.timer.stop()
        self.timer.reset()

        string = "[timelog] {}: {:.2f}s ({}) ".format(self.timer.name, runtime, time.strftime("%Y-%m-%d %H:%M:%S"))
        print(string)


class Timers:
    """Group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name: str, use_event: bool = False) -> _Timer | _GPUEventTimer:
        clazz = _GPUEventTimer if use_event and paddle.is_compiled_with_cuda() else _Timer
        timer = self.timers.get(name)
        if timer is None:
            timer = clazz(name)
            self.timers[name] = timer
        else:
            assert type(timer) == clazz, f"Invalid timer type: {clazz} vs {type(timer)}"
        return timer

    def write(
        self,
        names: list[str],
        writer: SummaryWriter | wandb,
        iteration: int,
        normalizer: float = 1.0,
        reset: bool = True,
    ):
        """Write timers to a tensorboard writer"""
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer
            if isinstance(writer, SummaryWriter) and SummaryWriter is not None:
                writer.add_scalar("timers/" + name, value, iteration)
            elif writer == wandb and wandb is not None:
                writer.log({"timers/" + name: value}, iteration)

    def log(
        self,
        names: list[str],
        normalizer: float = 1.0,
        reset: bool = True,
    ):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = "time (ms)"
        names = sorted(names)

        time_dict = {}
        for name in names:
            time_dict[name] = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer

        time_dict = sorted(time_dict.items(), key=lambda x: x[1], reverse=True)

        for time_tuple in time_dict:
            name, value = time_tuple
            string += f" | {name} : {value:.2f}"
        print(string)

    def info(self, names: list[str], normalizer: float = 1.0, reset: bool = False) -> dict[str, float]:
        """Return a dict of timers."""
        assert normalizer > 0.0
        time_dict = {}
        for name in names:
            time_dict[name] = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
        time_dict = dict(sorted(time_dict.items(), key=lambda x: x[0], reverse=False))
        return time_dict
