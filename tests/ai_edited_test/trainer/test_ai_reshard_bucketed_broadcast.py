# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for the bucketed broadcast path in trainer/utils/reshard/common.py.

Mirrors the existing single-process reshard unit tests: no real collective is
started. The broadcast inside each all-gather implementation short-circuits when
group.nranks < 2, yet the full bucket/pack/unpack/dtype logic still runs, so a
fake 1-rank group lets us diff the legacy vs bucketed paths in-process. The
tests call those implementations directly, because the public
all_gather_state_dict skips packing entirely at nranks < 2. The true multi-root
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
    _all_gather_state_dict_bucketed,
    _all_gather_state_dict_legacy,
    _iter_state_dict_bucket_chunks,
    _normalize_np_dtype_str,
    all_gather_state_dict,
    set_broadcast_max_chunk_bytes,
    set_bucketed_broadcast,
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


class TestBucketedMatchesLegacy(unittest.TestCase):
    """all_gather_state_dict must return identical results in both modes."""

    def tearDown(self):
        set_bucketed_broadcast(False)
        set_broadcast_max_chunk_bytes(_DEFAULT_MAX_CHUNK)

    def _gather(self, state_dict, filter_func, bucketed, max_chunk_bytes=None):
        if max_chunk_bytes is not None:
            set_broadcast_max_chunk_bytes(max_chunk_bytes)
        impl = _all_gather_state_dict_bucketed if bucketed else _all_gather_state_dict_legacy
        return impl(_copy_sd(state_dict), filter_func, _fake_group())

    def _assert_same(self, a, b):
        self.assertEqual(set(a.keys()), set(b.keys()))
        for k in a:
            ta, tb = a[k], b[k]
            self.assertEqual(list(ta.shape), list(tb.shape), f"shape mismatch for {k}")
            self.assertEqual(str(ta.dtype), str(tb.dtype), f"dtype mismatch for {k}")
            na = ta.astype("float32").numpy()
            nb = tb.astype("float32").numpy()
            np.testing.assert_array_equal(na, nb, err_msg=f"value mismatch for {k}")
        return a

    def test_basic_fp32(self):
        sd = OrderedDict(
            a=np.random.rand(4, 8).astype("float32"),
            b=np.random.rand(16).astype("float32"),
            c=np.random.rand(2, 2, 2).astype("float32"),
        )
        legacy = self._gather(sd, lambda x: True, bucketed=False)
        bucketed = self._gather(sd, lambda x: True, bucketed=True)
        self._assert_same(legacy, bucketed)
        # also check absolute correctness vs input for one key
        np.testing.assert_array_equal(bucketed["a"].numpy(), sd["a"])

    def test_bf16_return_numpy_uint16(self):
        # Regression: BF16 checkpoint loaded via return_numpy=True is uint16.
        # Bucketed must normalize dtype to bfloat16 (not trip the pack assert).
        base = np.random.rand(3, 5).astype("float32")
        sd = OrderedDict(w=_bf16_uint16(base), s=np.random.rand(8).astype("float32"))
        self.assertEqual(str(sd["w"].dtype), "uint16")
        legacy = self._gather(sd, lambda x: True, bucketed=False)
        bucketed = self._gather(sd, lambda x: True, bucketed=True)
        out = self._assert_same(legacy, bucketed)
        self.assertEqual(str(out["w"].dtype).split(".")[-1], "bfloat16")

    def test_partial_filter(self):
        sd = OrderedDict(
            keep0=np.random.rand(4).astype("float32"),
            drop=np.random.rand(4).astype("float32"),
            keep1=np.random.rand(4).astype("float32"),
        )
        f = lambda k: k.startswith("keep")
        legacy = self._gather(sd, f, bucketed=False)
        bucketed = self._gather(sd, f, bucketed=True)
        self.assertEqual(set(bucketed.keys()), {"keep0", "keep1"})
        self._assert_same(legacy, bucketed)

    def test_empty_and_scalar(self):
        sd = OrderedDict(
            empty=np.zeros([0], dtype="float32"),
            empty2d=np.zeros([2, 0], dtype="float32"),
            scalar=np.asarray(3.14, dtype="float32"),
            normal=np.random.rand(5).astype("float32"),
        )
        legacy = self._gather(sd, lambda x: True, bucketed=False)
        bucketed = self._gather(sd, lambda x: True, bucketed=True)
        self._assert_same(legacy, bucketed)
        self.assertEqual(list(bucketed["scalar"].shape), [])

    def test_cpu_tensor_input(self):
        # CPU paddle tensors take the DLPack view instead of Tensor.numpy(); BF16
        # is the dtype DLPack rejects, so it must still fall back to the copy.
        base = np.random.rand(4, 6).astype("float32")
        sd = OrderedDict(
            w=paddle.to_tensor(base, place=paddle.CPUPlace()),
            w_bf16=paddle.to_tensor(base, place=paddle.CPUPlace()).astype("bfloat16"),
            b=paddle.to_tensor(np.arange(5, dtype="float32"), place=paddle.CPUPlace()),
            empty=paddle.zeros([0], dtype="float32").cpu(),
        )
        out = self._gather(sd, lambda x: True, bucketed=True)
        self.assertEqual(set(out.keys()), set(sd.keys()))
        np.testing.assert_array_equal(out["w"].numpy(), base)
        np.testing.assert_array_equal(out["b"].numpy(), np.arange(5, dtype="float32"))
        self.assertEqual(str(out["w_bf16"].dtype).split(".")[-1], "bfloat16")
        np.testing.assert_array_equal(
            out["w_bf16"].astype("float32").numpy(),
            paddle.to_tensor(base).astype("bfloat16").astype("float32").numpy(),
        )
        self.assertEqual(list(out["empty"].shape), [0])

    def test_oversized_tensor_small_max_chunk(self):
        # Shrink bucket/chunk caps so tiny tensors exercise the multi-chunk and
        # oversized-single-bucket paths with real data (no GB allocations).
        with patch.object(reshard_common, "_STATE_DICT_BROADCAST_BUCKET_SIZE_BYTES", 256):
            sd = OrderedDict(
                big=np.random.rand(400).astype("float32"),  # 1600B > cap -> own bucket
                s0=np.random.rand(8).astype("float32"),
                s1=np.random.rand(8).astype("float32"),
            )
            legacy = self._gather(sd, lambda x: True, bucketed=False, max_chunk_bytes=512)
            bucketed = self._gather(sd, lambda x: True, bucketed=True, max_chunk_bytes=512)
            self._assert_same(legacy, bucketed)


class TestSingleRankFastPath(unittest.TestCase):
    """At nranks < 2 all_gather_state_dict skips packing, but still owes the
    caller the filtering and the numpy-to-tensor normalization."""

    def test_filters_sorts_and_converts(self):
        base = np.random.rand(3, 5).astype("float32")
        sd = OrderedDict(
            keep_w=_bf16_uint16(base),
            drop_me=np.random.rand(4).astype("float32"),
            keep_b=np.random.rand(4).astype("float32"),
        )
        out = all_gather_state_dict(_copy_sd(sd), lambda k: k.startswith("keep"), _fake_group())
        self.assertEqual(list(out.keys()), ["keep_b", "keep_w"])
        for k, v in out.items():
            self.assertIsInstance(v, paddle.Tensor, f"{k} was not converted")
            self.assertTrue(v.place.is_cpu_place(), f"{k} landed on {v.place}")
        self.assertEqual(str(out["keep_w"].dtype).split(".")[-1], "bfloat16")
        np.testing.assert_array_equal(out["keep_b"].numpy(), sd["keep_b"])

    def test_tensor_input_is_not_copied(self):
        t = paddle.to_tensor(np.random.rand(4).astype("float32"))
        out = all_gather_state_dict(OrderedDict(w=t), lambda x: True, _fake_group())
        self.assertIs(out["w"], t)

    def test_equivalent_to_bucketed_impl(self):
        sd = OrderedDict(
            w=_bf16_uint16(np.random.rand(3, 5).astype("float32")),
            b=np.random.rand(4).astype("float32"),
            empty=np.zeros([0], dtype="float32"),
            scalar=np.asarray(3.14, dtype="float32"),
        )
        f = lambda k: k != "b"  # noqa: E731
        fast = all_gather_state_dict(_copy_sd(sd), f, _fake_group())
        packed = _all_gather_state_dict_bucketed(_copy_sd(sd), f, _fake_group())
        self.assertEqual(list(fast.keys()), list(packed.keys()))
        for k in fast:
            self.assertEqual(str(fast[k].dtype), str(packed[k].dtype), f"dtype mismatch for {k}")
            self.assertEqual(str(fast[k].place), str(packed[k].place), f"place mismatch for {k}")
            self.assertEqual(list(fast[k].shape), list(packed[k].shape), f"shape mismatch for {k}")
            np.testing.assert_array_equal(
                fast[k].astype("float32").numpy(),
                packed[k].astype("float32").numpy(),
                err_msg=f"value mismatch for {k}",
            )


if __name__ == "__main__":
    unittest.main()
