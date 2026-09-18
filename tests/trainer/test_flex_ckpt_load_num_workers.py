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

"""Unit tests for the ``flex_ckpt_load_num_workers`` knob.

``num_workers`` only exists on newer paddle builds, so the trainer has to decide at runtime
whether the argument can be forwarded to ``dist.load_state_dict``. What is pinned down here:

  1. An explicit ``num_workers`` parameter is detected.
  2. A ``**kwargs``-style signature (a wrapper installed around the real function, e.g. by a
     profiler) counts as supported, because the argument reaches the wrapped callable.
  3. A paddle build without the parameter is detected, so the trainer falls back to the
     serial read instead of raising ``TypeError``.
  4. A callable ``inspect.signature`` cannot introspect is treated as unsupported rather
     than letting the exception escape into the checkpoint load path.
  5. The training argument defaults to 1, i.e. the original serial read.
"""

import inspect
import unittest
from dataclasses import fields
from unittest import mock

import paddle.distributed as dist

from paddleformers.trainer.trainer import _supports_load_num_workers
from paddleformers.trainer.training_args import TrainingArguments


def _load_state_dict_with_num_workers(state_dict, path=None, num_workers=1):
    pass


def _load_state_dict_wrapper(*args, **kwargs):
    pass


def _load_state_dict_legacy(state_dict, path=None, process_group=None):
    pass


class TestSupportsLoadNumWorkers(unittest.TestCase):
    def setUp(self):
        _supports_load_num_workers.cache_clear()

    def tearDown(self):
        _supports_load_num_workers.cache_clear()

    def _probe(self, fn):
        with mock.patch.object(dist, "load_state_dict", fn):
            return _supports_load_num_workers()

    def test_explicit_parameter_is_supported(self):
        self.assertTrue(self._probe(_load_state_dict_with_num_workers))

    def test_var_keyword_signature_is_supported(self):
        self.assertTrue(self._probe(_load_state_dict_wrapper))

    def test_missing_parameter_is_not_supported(self):
        self.assertFalse(self._probe(_load_state_dict_legacy))

    def test_uninspectable_callable_is_not_supported(self):
        with mock.patch.object(inspect, "signature", side_effect=ValueError("no signature")):
            self.assertFalse(_supports_load_num_workers())

    def test_result_is_cached(self):
        self.assertTrue(self._probe(_load_state_dict_with_num_workers))
        # The second probe must not re-inspect: the cached answer wins over the new signature.
        self.assertTrue(self._probe(_load_state_dict_legacy))


class TestFlexCkptLoadNumWorkersArgument(unittest.TestCase):
    def test_defaults_to_serial_read(self):
        field = {f.name: f for f in fields(TrainingArguments)}["flex_ckpt_load_num_workers"]
        self.assertEqual(field.type, int)
        self.assertEqual(field.default, 1)
        self.assertIn("help", field.metadata)


if __name__ == "__main__":
    unittest.main()
