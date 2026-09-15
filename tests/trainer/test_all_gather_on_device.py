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

"""Unit tests for all_gather_on_device and the switch that selects it.

Every collective in reshard/common.py short-circuits at ``nranks < 2``, so bucketing,
packing, the writeback and the fallbacks can all be exercised in one process against a
single-rank stub group.

What is pinned down:
  1. Bit-exact agreement with the bucketed host path it replaces.
  2. No host tensor is produced or consumed -- host input is rejected, and results are
     either markers (written into the parameter) or device tensors.
  3. Values survive the bucket being freed, i.e. the writes really copy.
  4. A key with no matching parameter, or a mismatched shape/dtype, gets its own device
     tensor instead of corrupting a parameter.
  5. The branch is selected by TrainingArguments, and returns None when off so the
     caller falls back to the host path.

Multi-rank equivalence (the broadcast actually lining up across ranks) is not covered
here; it needs a distributed harness.
"""

import inspect
import unittest

import numpy as np
import paddle

from paddleformers.trainer.trainer_utils import _restore_master_weights_2d_on_device
from paddleformers.trainer.training_args import TrainingArguments
from paddleformers.trainer.utils.reshard import common as reshard_common
from paddleformers.trainer.utils.reshard.common import (
    AssignedMasterWeight,
    all_gather_on_device,
    all_gather_state_dict,
    set_device_gather,
)


class _SingleRankGroup:
    nranks = 1
    rank = 0
    id = 0
    ranks = [0]


SHAPES = {
    "p_a": [8, 16],
    "p_b": [4, 32],
    "p_c": [16, 8],
    "p_vec": [64],
    "p_big": [64, 64],
}


def _rand_bf16(shape, seed):
    rng = np.random.RandomState(seed)
    host = paddle.to_tensor(rng.uniform(-1, 1, shape).astype("float32"))
    return paddle.cast(host, paddle.bfloat16)


def _f32(tensor):
    return paddle.cast(tensor, paddle.float32).numpy()


class TestAllGatherOnDevice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() == 0:
            raise unittest.SkipTest("all_gather_on_device is a device-only path")
        paddle.set_device("gpu:0")

    def setUp(self):
        self.group = _SingleRankGroup()
        self._old_bucket = reshard_common._STATE_DICT_BROADCAST_BUCKET_SIZE_BYTES
        # small budget so the fixtures really produce several buckets and chunks
        self.chunk = 4 * 1024

    def tearDown(self):
        reshard_common._STATE_DICT_BROADCAST_BUCKET_SIZE_BYTES = self._old_bucket

    def _sources(self):
        return {name: _rand_bf16(shape, seed) for seed, (name, shape) in enumerate(SHAPES.items())}

    def _sink(self):
        return {name: paddle.full(shape, -7.0, dtype=paddle.bfloat16) for name, shape in SHAPES.items()}

    def _host_reference(self):
        """Ground truth from the untouched bucketed host path."""
        reshard_common._STATE_DICT_BROADCAST_BUCKET_SIZE_BYTES = self.chunk
        reshard_common.set_broadcast_max_chunk_bytes(self.chunk * 2)
        out = all_gather_state_dict({k: v.cpu() for k, v in self._sources().items()}, lambda x: True, self.group)
        return {k: _f32(v) for k, v in out.items()}

    def test_matches_the_host_path_bitwise(self):
        reference = self._host_reference()
        sink = self._sink()
        out = all_gather_on_device(dict(self._sources()), self.group, sink, max_chunk_bytes=self.chunk)

        self.assertEqual(set(out.keys()), set(reference.keys()))
        for name, shape in SHAPES.items():
            self.assertIsInstance(out[name], AssignedMasterWeight, f"{name} did not land in its parameter")
            self.assertEqual(out[name].shape, shape)
            np.testing.assert_array_equal(_f32(sink[name]), reference[name], err_msg=name)

    def test_no_parameter_is_left_unwritten(self):
        sink = self._sink()
        all_gather_on_device(dict(self._sources()), self.group, sink, max_chunk_bytes=self.chunk)
        for name, param in sink.items():
            self.assertFalse(bool(paddle.any(paddle.cast(param, paddle.float32) == -7.0)), name)

    def test_values_survive_the_bucket_being_freed(self):
        sink = self._sink()
        all_gather_on_device(dict(self._sources()), self.group, sink, max_chunk_bytes=self.chunk)
        snapshot = {name: _f32(p) for name, p in sink.items()}
        for _ in range(8):  # churn the allocator so freed buckets get reused
            scratch = paddle.full([1024 * 64], 1234.5, dtype=paddle.float32)
            del scratch
        for name, param in sink.items():
            np.testing.assert_array_equal(_f32(param), snapshot[name], err_msg=name)

    def test_everything_returned_is_on_device(self):
        sink = self._sink()
        sink.pop("p_b")  # force one key onto the standalone path
        out = all_gather_on_device(dict(self._sources()), self.group, sink, max_chunk_bytes=self.chunk)
        self.assertNotIsInstance(out["p_b"], AssignedMasterWeight)
        self.assertFalse(out["p_b"].place.is_cpu_place(), "the fallback tensor must stay on device")

    def test_source_consumed(self):
        sources = dict(self._sources())
        all_gather_on_device(sources, self.group, self._sink(), max_chunk_bytes=self.chunk)
        self.assertEqual(len(sources), 0, "sources must be released as they are packed")

    def test_host_input_is_rejected(self):
        sources = {k: v.cpu() for k, v in self._sources().items()}
        with self.assertRaises(AssertionError):
            all_gather_on_device(sources, self.group, self._sink(), max_chunk_bytes=self.chunk)

    def test_unknown_name_gets_its_own_tensor(self):
        reference = self._host_reference()
        sink = self._sink()
        sink.pop("p_vec")
        out = all_gather_on_device(dict(self._sources()), self.group, sink, max_chunk_bytes=self.chunk)
        np.testing.assert_array_equal(_f32(out["p_vec"]), reference["p_vec"])
        self.assertIsInstance(out["p_a"], AssignedMasterWeight)

    def test_shape_mismatch_does_not_touch_the_parameter(self):
        reference = self._host_reference()
        sink = self._sink()
        sink["p_vec"] = paddle.full([32], -7.0, dtype=paddle.bfloat16)  # wrong length
        out = all_gather_on_device(dict(self._sources()), self.group, sink, max_chunk_bytes=self.chunk)
        self.assertNotIsInstance(out["p_vec"], AssignedMasterWeight)
        np.testing.assert_array_equal(_f32(out["p_vec"]), reference["p_vec"])
        self.assertTrue(bool(paddle.all(paddle.cast(sink["p_vec"], paddle.float32) == -7.0)))

    def test_dtype_mismatch_does_not_touch_the_parameter(self):
        reference = self._host_reference()
        sink = self._sink()
        sink["p_big"] = paddle.full(SHAPES["p_big"], -7.0, dtype=paddle.float32)
        out = all_gather_on_device(dict(self._sources()), self.group, sink, max_chunk_bytes=self.chunk)
        self.assertNotIsInstance(out["p_big"], AssignedMasterWeight)
        np.testing.assert_array_equal(_f32(out["p_big"]), reference["p_big"])
        self.assertTrue(bool(paddle.all(sink["p_big"] == -7.0)))

    def test_empty_sink_returns_plain_device_tensors(self):
        reference = self._host_reference()
        out = all_gather_on_device(dict(self._sources()), self.group, None, max_chunk_bytes=self.chunk)
        for name, value in out.items():
            self.assertNotIsInstance(value, AssignedMasterWeight)
            self.assertFalse(value.place.is_cpu_place())
            np.testing.assert_array_equal(_f32(value), reference[name], err_msg=name)

    def test_bigger_buckets_mean_fewer_broadcasts(self):
        """With no host upload left to amortise, coarser buckets cost strictly less."""
        meta = [(k, ("bfloat16", v, 0)) for k, v in SHAPES.items()]
        few, _ = reshard_common._build_state_dict_broadcast_buckets(meta, 1024 * 1024)
        many, _ = reshard_common._build_state_dict_broadcast_buckets(meta, 1024)
        self.assertLess(len(few), len(many))
        self.assertEqual(len(few), 1, "one owner and one dtype should collapse into one bucket")


class TestDeviceGatherSwitch(unittest.TestCase):
    """The 2D branch is selected by config, and declines cleanly when off.

    Device-free on purpose: this is about the wiring from TrainingArguments down to
    _restore_master_weights_2d_on_device, which returns None whenever the switch is off
    so the caller can fall back to the host path without duplicating the decision.
    """

    def setUp(self):
        self._old = reshard_common._USE_DEVICE_GATHER

    def tearDown(self):
        set_device_gather(self._old)

    def test_defaults_to_the_host_path(self):
        self.assertFalse(TrainingArguments.reshard_master_weight_device_gather)

    def test_setter_drives_the_switch(self):
        set_device_gather(True)
        self.assertTrue(reshard_common.use_device_gather())
        set_device_gather(False)
        self.assertFalse(reshard_common.use_device_gather())

    def test_off_makes_the_2d_restore_decline(self):
        set_device_gather(False)
        self.assertIsNone(_restore_master_weights_2d_on_device({}, _SingleRankGroup(), {}))

    def test_does_not_go_through_all_gather_state_dict(self):
        """The device path reuses the bucket kernel directly, not the host entry point."""
        src = inspect.getsource(reshard_common.all_gather_on_device)
        doc = reshard_common.all_gather_on_device.__doc__ or ""
        body = src.replace(doc, "")  # the docstring mentions the host entry point by name
        self.assertNotIn("all_gather_state_dict", body)
        for helper in ("_build_state_dict_broadcast_buckets", "_iter_state_dict_bucket_chunks"):
            self.assertIn(helper, body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
