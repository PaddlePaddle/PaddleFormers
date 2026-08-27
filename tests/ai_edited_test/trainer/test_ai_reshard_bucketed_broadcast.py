# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for the bucketed broadcast path in trainer/utils/reshard/common.py.

Mirrors the existing single-process reshard unit tests: no real collective is
started. all_gather_state_dict short-circuits the broadcast when group.nranks
< 2, yet still runs the full bucket/pack/unpack/dtype logic, so a fake 1-rank
group lets us exercise the pack/unpack path in-process. The true multi-root
collective sequence (>=2 ranks) is out of scope here and belongs to a
launch-based integration test.
"""

import unittest
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import numpy as np
import paddle

from paddleformers.trainer.utils.reshard import common as reshard_common
from paddleformers.trainer.utils.reshard.common import (
    _iter_state_dict_bucket_chunks,
    _normalize_np_dtype_str,
    all_gather_state_dict,
    set_broadcast_max_chunk_bytes,
)

_DEFAULT_MAX_CHUNK = reshard_common._STATE_DICT_BROADCAST_MAX_CHUNK_BYTES


def _fake_group(nranks=1, rank=0, gid=0):
    g = MagicMock()
    g.nranks = nranks
    g.rank = rank
    g.id = gid
    g.ranks = list(range(nranks))
    return g


def _copy_sd(state_dict):
    # all_gather_state_dict consumes (pops) its input, so hand each run a copy.
    out = OrderedDict()
    for k, v in state_dict.items():
        out[k] = v.copy() if isinstance(v, np.ndarray) else v.clone()
    return out


def _bf16_uint16(np_float_array):
    # Reproduce ShardingIO's paddle.load(return_numpy=True) for a BF16 tensor:
    # the bytes come back as numpy uint16.
    return paddle.to_tensor(np_float_array).astype("bfloat16").numpy()


class TestNormalizeNpDtypeStr(unittest.TestCase):
    def test_uint16_maps_to_bfloat16(self):
        # paddle has no numpy-native bf16; it stores bf16 as uint16 and to_tensor
        # restores bfloat16, so meta must record the paddle-effective dtype.
        self.assertEqual(_normalize_np_dtype_str("uint16"), "bfloat16")

    def test_other_dtypes_unchanged(self):
        for dt in ("float32", "float16", "int64", "bfloat16"):
            self.assertEqual(_normalize_np_dtype_str(dt), dt)


class TestIterBucketChunks(unittest.TestCase):
    def test_oversized_bucket_stays_alone(self):
        # A single bucket larger than max_chunk_bytes is NOT split; it lands in
        # its own chunk (documented limitation, matches the per-tensor path).
        buckets = [{"nbytes": 100}, {"nbytes": 100}, {"nbytes": 5000}, {"nbytes": 100}]
        chunks = list(_iter_state_dict_bucket_chunks(buckets, chunk_size=256, max_chunk_bytes=1000))
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], buckets[0:2])  # small ones aggregate up to cap
        self.assertEqual(chunks[1], [buckets[2]])  # oversized alone (5000 > 1000)
        self.assertEqual(chunks[2], [buckets[3]])

    def test_chunk_size_count_cap(self):
        buckets = [{"nbytes": 1} for _ in range(5)]
        chunks = list(_iter_state_dict_bucket_chunks(buckets, chunk_size=2, max_chunk_bytes=10**9))
        self.assertEqual([len(c) for c in chunks], [2, 2, 1])


class TestBucketedGather(unittest.TestCase):
    """all_gather_state_dict must preserve key/shape/dtype/value."""

    def tearDown(self):
        set_broadcast_max_chunk_bytes(_DEFAULT_MAX_CHUNK)

    def _gather(self, state_dict, filter_func, max_chunk_bytes=None):
        if max_chunk_bytes is not None:
            set_broadcast_max_chunk_bytes(max_chunk_bytes)
        return all_gather_state_dict(_copy_sd(state_dict), filter_func, _fake_group())

    def _assert_matches_input(self, out, sd, keys):
        self.assertEqual(set(out.keys()), set(keys))
        for k in keys:
            t = out[k]
            self.assertEqual(list(t.shape), list(np.asarray(sd[k]).shape), f"shape mismatch for {k}")
            na = t.astype("float32").numpy()
            nb = paddle.to_tensor(sd[k]).astype("float32").numpy()
            np.testing.assert_array_equal(na, nb, err_msg=f"value mismatch for {k}")
        return out

    def test_basic_fp32(self):
        sd = OrderedDict(
            a=np.random.rand(4, 8).astype("float32"),
            b=np.random.rand(16).astype("float32"),
            c=np.random.rand(2, 2, 2).astype("float32"),
        )
        out = self._gather(sd, lambda x: True)
        self._assert_matches_input(out, sd, ["a", "b", "c"])

    def test_bf16_return_numpy_uint16(self):
        # Regression: BF16 checkpoint loaded via return_numpy=True is uint16.
        # Bucketed must normalize dtype to bfloat16 (not trip the pack assert).
        base = np.random.rand(3, 5).astype("float32")
        sd = OrderedDict(w=_bf16_uint16(base), s=np.random.rand(8).astype("float32"))
        self.assertEqual(str(sd["w"].dtype), "uint16")
        out = self._gather(sd, lambda x: True)
        self.assertEqual(str(out["w"].dtype).split(".")[-1], "bfloat16")
        self._assert_matches_input(out, sd, ["w", "s"])

    def test_partial_filter(self):
        sd = OrderedDict(
            keep0=np.random.rand(4).astype("float32"),
            drop=np.random.rand(4).astype("float32"),
            keep1=np.random.rand(4).astype("float32"),
        )
        f = lambda k: k.startswith("keep")
        out = self._gather(sd, f)
        self._assert_matches_input(out, sd, ["keep0", "keep1"])

    def test_empty_and_scalar(self):
        sd = OrderedDict(
            empty=np.zeros([0], dtype="float32"),
            empty2d=np.zeros([2, 0], dtype="float32"),
            scalar=np.asarray(3.14, dtype="float32"),
            normal=np.random.rand(5).astype("float32"),
        )
        out = self._gather(sd, lambda x: True)
        self._assert_matches_input(out, sd, ["empty", "empty2d", "scalar", "normal"])
        self.assertEqual(list(out["scalar"].shape), [])

    def test_oversized_tensor_small_max_chunk(self):
        # Shrink bucket/chunk caps so tiny tensors exercise the multi-chunk and
        # oversized-single-bucket paths with real data (no GB allocations).
        with patch.object(reshard_common, "_STATE_DICT_BROADCAST_BUCKET_SIZE_BYTES", 256):
            sd = OrderedDict(
                big=np.random.rand(400).astype("float32"),  # 1600B > cap -> own bucket
                s0=np.random.rand(8).astype("float32"),
                s1=np.random.rand(8).astype("float32"),
            )
            out = self._gather(sd, lambda x: True, max_chunk_bytes=512)
            self._assert_matches_input(out, sd, ["big", "s0", "s1"])


if __name__ == "__main__":
    unittest.main()
