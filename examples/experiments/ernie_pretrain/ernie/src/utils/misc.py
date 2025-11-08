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

import copy
import logging
import re

import numpy as np
import paddle
import paddle.distributed as dist

logger = logging.getLogger(__name__)

try:
    from models.sequence_parallel_utils import get_async_loader
    from paddle.incubate.tensor.manipulation import async_offload
except ImportError:
    get_async_loader = async_offload = None

__all__ = (
    "SmoothedValue",
    "global_training_logs",
)

ZERO = paddle.zeros([], dtype="float32")


class SmoothedValue:
    def __init__(
        self,
        skip_zero,
    ):
        self.total = 0.0
        self.count = 0
        self._skip_zero = skip_zero

    @paddle.no_grad()
    def update(self, value):
        if isinstance(value, paddle.Tensor):
            value = value.astype("float32").detach()
            if value.shape == [1]:
                value = value.squeeze()
            self.count += (value != ZERO).astype("int64") if self._skip_zero else 1
        else:
            self.count += 1
        self.total += value

    @property
    def global_avg(self):
        return self.total / max(self.count, 1e-6)

    def reset(self):
        self.total = 0.0
        self.count = 0


class TrainingLogs:
    _instance = None

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance

    def __init__(self):
        self.meters = {}
        self.snapshot = None
        self._global_meters_keys = []
        self.trainer = None
        self.logging_interval = None
        self._skip_zero_keys = []

    def set_trainer_interval(self, trainer, logging_interval):
        self.trainer = trainer
        self.logging_interval = logging_interval

    @property
    def global_meters_keys(self):
        return self._global_meters_keys

    @global_meters_keys.setter
    def global_meters_keys(self, lst):
        self._global_meters_keys = lst

    def enable_skip_zero(self, keys=[]):
        logger.info("global_training_logs: use skip zero")
        self._skip_zero_keys = keys
        for m in self.meters.keys():
            for k in keys:
                if re.match(k, m):
                    m._skip_zero = True

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def is_enabled(self):
        return self.trainer is None or (self.trainer.state.global_step + 1) % self.logging_interval == 0

    def __setitem__(self, k, v):
        skip_zero = False
        for skip_k in self._skip_zero_keys:
            if re.match(skip_k, k):
                skip_zero = True
        metric = self.meters.setdefault(k, SmoothedValue(skip_zero=skip_zero))
        metric.update(v)

    def __getitem__(self, v):
        return self.meters[v]

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def dict(self, use_async=False):
        avg_metric = {k: v.global_avg for k, v in self.meters.items() if k not in self.global_meters_keys}

        if self.global_meters_keys:
            tensor_lst = []
            for k in self.global_meters_keys:
                v = self.meters[k].global_avg if k in self.meters else -100
                tensor_lst.append(paddle.to_tensor(v, "float32"))
            gathered_v = []
            dist.gather(paddle.stack(tensor_lst), gathered_v, 0)
            if gathered_v:
                for i, k in enumerate(self.global_meters_keys):
                    avg_metric[k] = np.mean([t[i] for t in gathered_v if t[i] != -100]).item()

        if not use_async:
            ret = {k: v.item() if isinstance(v, paddle.Tensor) else v for k, v in avg_metric.items()}
            global_info = {k: v for k, v in ret.items() if k in self.global_meters_keys}
            ret = {
                k: v
                for k, v in ret.items()
                if (k not in self.global_meters_keys) and ((not self.meters[k]._skip_zero) or v != 0.0)
            }
            return ret, global_info
        assert get_async_loader is not None, "async logging requires latest paddle"
        if not avg_metric:
            return lambda: ({}, {})
        keys, values = zip(*avg_metric.items())
        tensor_list = [(i, t) for i, t in enumerate(values) if isinstance(t, paddle.Tensor)]
        if tensor_list:
            async_loader = get_async_loader()
            tensor_id, tensor_list = zip(*tensor_list)
            tensor_list = paddle.stack(tensor_list)
            tensor_list_cpu, task = async_offload(tensor_list, async_loader)
        else:
            task = None

        def _ret():
            nonlocal task, tensor_list_cpu, values
            values = list(values)
            if task:
                task.cpu_wait()
                for i, val in zip(tensor_id, tensor_list_cpu.tolist()):
                    values[i] = val
            ret = dict(zip(keys, values))
            global_info = {k: v for k, v in ret.items() if k in self.global_meters_keys}
            ret = {
                k: v
                for k, v in ret.items()
                if (k not in self.global_meters_keys) and ((not self.meters[k]._skip_zero) or v != 0.0)
            }
            return ret, global_info

        return _ret

    def reset(self):
        for k in list(self.meters.keys()):
            self.meters[k].reset()
            self.meters.pop(k)

    def take_snapshot(self):
        self.snapshot = copy.deepcopy(self.meters)

    def restore_snapshot(self):
        assert self.snapshot is not None, "you should use take_snapshot before restore_snapshot"
        self.meters = copy.deepcopy(self.snapshot)
        self.snapshot = None


global_training_logs = TrainingLogs()
