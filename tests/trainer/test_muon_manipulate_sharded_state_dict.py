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

"""Unit tests for ZeroCostCheckpointCallbackFcBased._muon_manipulate_sharded_state_dict.

Single-process, no GPU, no distributed init: the method is called unbound with light
fakes standing in for model / MuonShardingOptimizer / TrainingArguments. It only ever
reads ``sw.local_tensor.name``, so the fakes can stay this thin.

What is pinned down here:
  1. replicate_saved_into_local=False reproduces the pre-change behaviour byte for byte
     (2D params owned by another rank are dropped).
  2. replicate_saved_into_local=True keeps them, but the master-weight deletion loop
     still removes every bf16 trainable 2D param, so the only thing that actually
     reaches the file is the natively-fp32 2D params (mHC mapping_proj etc.).
  3. No bf16 2D param survives on any rank. This is what bounds the extra write
     volume; a regression here would mean writing the full bf16 2D set per rank.
  4. With the switch on, the saved fp32 2D set no longer depends on Muon's greedy
     ownership, which is what turns one candidate replica into N.
"""

import unittest
from collections import OrderedDict

import paddle

from paddleformers.trainer.utils.zero_cost_checkpoint import (
    ZeroCostCheckpointCallbackFcBased,
)

_MANIPULATE = ZeroCostCheckpointCallbackFcBased._muon_manipulate_sharded_state_dict


class _FakeTensor:
    def __init__(self, name, dtype):
        self.name = name
        self.dtype = dtype


class _FakeShardedWeight:
    def __init__(self, name, dtype):
        self.local_tensor = _FakeTensor(name, dtype)


class _FakeModel:
    """Returns sharded_state_dict() keyed by structure name, unsorted on purpose."""

    def __init__(self, entries):
        self._entries = entries

    def sharded_state_dict(self):
        return {f"struct.{name}": _FakeShardedWeight(name, dtype) for name, dtype in self._entries}


class _FakeGroup:
    def __init__(self, nranks):
        self.nranks = nranks


class _FakeHcg:
    def __init__(self, nranks):
        self._group = _FakeGroup(nranks)

    def get_sharding_parallel_group(self):
        return self._group


class _FakeInnerOpt:
    def __init__(self, master_weight_names, multi_precision=True):
        self._multi_precision = multi_precision
        self._master_weights = {name: object() for name in master_weight_names}


class _FakeOptimizer:
    def __init__(self, local_2d, all_2d, all_1d, master_weight_names, sharding_rank=0, nranks=1):
        self._sharding_rank = sharding_rank
        self._local_2d = [_FakeTensor(n, paddle.bfloat16) for n in local_2d]
        self._params_2d_by_color = {0: [_FakeTensor(n, paddle.bfloat16) for n in all_2d]}
        self._params_1d = [_FakeTensor(n, paddle.bfloat16) for n in all_1d]
        self._inner_opt = _FakeInnerOpt(master_weight_names)
        self._hcg = _FakeHcg(nranks)


class _FakeArgs:
    def __init__(self, replicate_saved_into_local):
        self.replicate_saved_into_local = replicate_saved_into_local


class _FakeSelf:
    def __init__(self, replicate_saved_into_local):
        self.args = _FakeArgs(replicate_saved_into_local)


class TestMuonManipulateShardedStateDict(unittest.TestCase):
    """The scenario mirrors a large MoE Muon job in miniature."""

    def setUp(self):
        # owned_bf16 : 2D Muon param this rank owns, bf16 -> has a master weight
        # other_bf16 : 2D Muon param another rank owns, bf16 -> has a master weight
        # owned_fp32 : 2D Muon param this rank owns, natively fp32 -> no master weight
        # other_fp32 : 2D Muon param another rank owns, natively fp32 -> no master weight
        #              (this is the mapping_proj.weight / hc_head_fn class of params)
        # oned_bf16  : 1D param, always saved
        # buffer_fp32: neither 2D nor 1D -> falls to the final else-branch
        self.entries = [
            ("owned_bf16", paddle.bfloat16),
            ("other_bf16", paddle.bfloat16),
            ("owned_fp32", paddle.float32),
            ("other_fp32", paddle.float32),
            ("oned_bf16", paddle.bfloat16),
            ("buffer_fp32", paddle.float32),
        ]
        self.local_2d = ["owned_bf16", "owned_fp32"]
        self.all_2d = ["owned_bf16", "other_bf16", "owned_fp32", "other_fp32"]
        self.all_1d = ["oned_bf16"]
        # only the bf16 trainable params have an fp32 master weight
        self.master_weight_names = ["owned_bf16", "other_bf16", "oned_bf16"]

    def _run(self, replicate, sharding_rank=0, nranks=1):
        model = _FakeModel(self.entries)
        optimizer = _FakeOptimizer(
            self.local_2d,
            self.all_2d,
            self.all_1d,
            self.master_weight_names,
            sharding_rank=sharding_rank,
            nranks=nranks,
        )
        out = _MANIPULATE(_FakeSelf(replicate), model, optimizer)
        self.assertIsInstance(out, OrderedDict)
        return {sw.local_tensor.name for sw in out.values()}

    def test_replicate_off_matches_legacy_behaviour(self):
        """Switch off => 2D params owned elsewhere are dropped, as the old `continue` did."""
        names = self._run(replicate=False, sharding_rank=0)
        self.assertEqual(names, {"owned_fp32", "buffer_fp32"})
        # owned_bf16 / oned_bf16 removed by the master-weight deletion loop
        self.assertNotIn("owned_bf16", names)
        self.assertNotIn("oned_bf16", names)
        # the key that used to have a single replica across the whole job
        self.assertNotIn("other_fp32", names)

    def test_replicate_off_non_zero_sharding_rank(self):
        """Non-rank-0 with the switch off must not pick up the else-branch buffer."""
        names = self._run(replicate=False, sharding_rank=3)
        self.assertEqual(names, {"owned_fp32"})

    def test_replicate_on_adds_only_fp32_2d_params(self):
        """Switch on => the previously single-replica fp32 2D param is saved locally too."""
        names = self._run(replicate=True, sharding_rank=0)
        self.assertEqual(names, {"owned_fp32", "other_fp32", "buffer_fp32"})

    def test_replicate_on_still_drops_every_bf16_2d_param(self):
        """Write-volume guard: no bf16 2D param may survive."""
        names = self._run(replicate=True, sharding_rank=7)
        for name, dtype in self.entries:
            if dtype == paddle.bfloat16:
                self.assertNotIn(name, names, f"{name} must not be saved")
        self.assertEqual(names, {"owned_fp32", "other_fp32", "buffer_fp32"})

    def test_replicate_on_delta_is_exactly_the_non_owned_fp32_2d_params(self):
        """The point of the change, stated as a set difference."""
        off = self._run(replicate=False, sharding_rank=5)
        on = self._run(replicate=True, sharding_rank=5)
        self.assertEqual(on - off, {"other_fp32", "buffer_fp32"})
        self.assertEqual(off - on, set())

    def test_every_rank_saves_the_same_fp32_2d_params(self):
        """With the switch on, the fp32 2D set no longer depends on Muon's greedy ownership."""
        per_rank = []
        for rank in range(4):
            # rotate ownership to emulate greedy bin-packing assigning owners differently
            self.local_2d = [["owned_bf16", "owned_fp32"], ["other_bf16"], ["other_fp32"], []][rank]
            per_rank.append({n for n in self._run(replicate=True, sharding_rank=rank) if "fp32" in n})
        for names in per_rank:
            self.assertEqual(names, {"owned_fp32", "other_fp32", "buffer_fp32"})

    def test_multi_precision_off_keeps_everything(self):
        """No master weights at all => deletion loop is skipped, nothing is removed."""
        model = _FakeModel(self.entries)
        optimizer = _FakeOptimizer(self.local_2d, self.all_2d, self.all_1d, [])
        optimizer._inner_opt._multi_precision = False
        names = {sw.local_tensor.name for sw in _MANIPULATE(_FakeSelf(True), model, optimizer).values()}
        self.assertEqual(names, {n for n, _ in self.entries})


if __name__ == "__main__":
    unittest.main(verbosity=2)
