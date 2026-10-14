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

"""Tests for the HF bit-exact gradient clip and its distributed wrappers."""

import itertools
from types import SimpleNamespace

import paddle
import pytest
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.hybrid_parallel_optimizer import (
    HybridParallelClipGrad,
)

from paddleformers.utils.hf_bitexact_clip import (
    HFBitexactClipGradByGlobalNorm,
    _bf16,
    _hf_global_norm,
    hf_norm_partition_size,
    unwrap_hf_bitexact_clip,
    verify_hf_norm_groups_registered,
)
from paddleformers.utils.hf_bitexact_hybrid_clip import (
    HFBitexactHybridParallelClipGrad,
    restore_hf_bitexact_clip,
)
from paddleformers.utils.moe_hybrid_parallel_optimizer import MoEHybridParallelClipGrad


def _param(name, *, is_distributed=False, no_sync=False, need_clip=True, groups=None):
    p = SimpleNamespace(
        name=name,
        is_distributed=is_distributed,
        no_sync=no_sync,
        need_clip=need_clip,
    )
    if groups is not None:
        p.hf_norm_groups = [paddle.to_tensor(c, dtype="int64") for c in groups]
    return p


def _clip(norm=1.0):
    return HFBitexactClipGradByGlobalNorm(norm)


def _find_differing_grouping():
    """Return ``(values, split_w, split_g, (whole, grouped))`` of four positive
    halves where ``bf16(sqrt(sum(v^2)))`` differs from the grouped recipe
    ``bf16(sqrt(sum(bf16(half_norm)^2)))``.

    Searched rather than hardcoded so the case keeps demonstrating a real
    difference if the rounding helpers ever change.
    """
    import math

    candidates = [1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 5.0]

    def bf16(x):
        return float(paddle.to_tensor([x], dtype="float32").astype("bfloat16").astype("float32")[0])

    for a, b, c, d in itertools.product(candidates, repeat=4):
        whole = bf16(math.sqrt(a * a + b * b + c * c + d * d))
        h1 = bf16(math.sqrt(a * a + b * b))
        h2 = bf16(math.sqrt(c * c + d * d))
        grouped = bf16(math.sqrt(h1 * h1 + h2 * h2))
        if whole != grouped and h1 > 0 and h2 > 0 and whole > 0:
            return [a, b, c, d], [0, 1], [2, 3], (whole, grouped)
    return None


class TestHFClipRecipe:
    def test_global_norm_rounds_through_bf16(self):
        # sqrt(2) in BF16 must come from the rounded value, not the FP32 literal.
        gn = _hf_global_norm(paddle.to_tensor([2.0], dtype="float32"))
        assert float(gn) == 1.4140625  # bf16(sqrt(2)), not 1.4142135...

    def test_no_clip_when_need_clip_false(self):
        clip = _clip(0.1)
        g = paddle.to_tensor([1.0, 2.0], dtype="float32")
        p = _param("p", need_clip=False)
        out = clip._dygraph_clip([(p, g)])
        # untouched and in place
        assert out[0][1] is g
        assert float(g.sum()) == 3.0

    def test_scale_clamps_coefficient_at_one(self):
        # ``norm < 1`` makes ``max/denom > 1``; the BF16-rounded coefficient is
        # clamped to 1.0 and the multiplication still happens (bit-identical),
        # so the FP32 grad ends up holding exactly its own BF16 rounding.
        clip = _clip(1.0)
        g = paddle.to_tensor([0.1, 0.2], dtype="float32")
        p = _param("p")
        clip._dygraph_clip([(p, g)])
        expected = _bf16(paddle.to_tensor([0.1], dtype="float32")) + _bf16(paddle.to_tensor([0.2], dtype="float32"))
        assert float(g.sum()) == float(expected)


class TestHFClipNormGroups:
    """``hf_norm_groups`` must change the global norm whenever the reference's
    per-Linear partition does.

    The registration itself happens in PaddleFleet, at model construction
    (``_maybe_tag_qkv_dgrad_groups`` / ``_maybe_tag_in_proj_dgrad_groups`` /
    ``_maybe_tag_up_gate_norm_groups``); this test pins the *contract* the clip
    relies on: a fused parameter that publishes two column groups normed
    separately must produce a different global norm than the unsplit one
    whenever BF16 rounding makes them differ.
    """

    def test_fused_parameter_is_split_like_the_reference(self):
        found = _find_differing_grouping()
        assert found is not None, "no grouping produced a differing global norm"
        values, g1, g2, (whole, grouped) = found

        # unsplit: one BF16 norm over the whole fused tensor
        clip = _clip(1.0)
        p = _param("fused", groups=None)
        g = paddle.to_tensor([values], dtype="float32")
        clip._dygraph_clip([(p, g)])
        assert clip.last_global_norm == whole

        # split: two BF16 norms, squared and summed, like torch's per-nn.Linear
        clip2 = _clip(1.0)
        p2 = _param("fused", groups=[g1, g2])
        g2t = paddle.to_tensor([values], dtype="float32")
        clip2._dygraph_clip([(p2, g2t)])
        assert clip2.last_global_norm == grouped
        # and the two recipes genuinely disagree for this input
        assert whole != grouped


class TestMoEHybridParallelClipHf:
    """The MoE distributed wrapper must keep the HF recipe, not replace it."""

    def _wrapped_clip(self, norm=1.0):
        inner = _clip(norm)
        wrapper = MoEHybridParallelClipGrad(inner, hcg=None, timers=None)
        # no distributed job in unit tests: the reduction step is a no-op, which
        # keeps the *local* part identical to the single-card recipe
        wrapper._global_norm = lambda *a, **k: None
        return inner, wrapper

    def test_distributed_wrapper_keeps_hf_global_norm(self):
        inner, wrapper = self._wrapped_clip()
        g = paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32")
        p = _param("w")
        inner._dygraph_clip([(p, g)])
        single_norm = inner.last_global_norm
        wrapper._dygraph_clip([(p, paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32"))])
        # identical to the single-card clip on the same input
        assert wrapper.stat["global_grad_norm"] == single_norm
        ref = _bf16(paddle.sqrt(paddle.to_tensor([1.0 + 4.0 + 9.0], dtype="float32")))
        assert wrapper.stat["global_grad_norm"] == float(ref)

    def test_distributed_wrapper_hf_uses_bf16_norm_squares(self):
        """The buckets must carry squared *rounded* norms; with a single param
        the sum is the square of the reference's per-tensor norm."""
        inner, wrapper = self._wrapped_clip()
        g = paddle.to_tensor([1.0, 2.0], dtype="float32")
        p = _param("w")
        wrapper._dygraph_clip([(p, g)])
        ref_norm = _bf16(paddle.sqrt(paddle.to_tensor([5.0], dtype="float32")))
        # global = bf16(sqrt(ref_norm^2)) == ref_norm
        assert float(ref_norm) == 2.234375
        assert wrapper.stat["global_grad_norm"] == float(ref_norm)

    def test_distributed_wrapper_scales_bf16_rounded(self):
        inner, wrapper = self._wrapped_clip(norm=0.5)
        g = paddle.to_tensor([4.0, 0.0], dtype="float32")
        p = _param("w")
        out = wrapper._dygraph_clip([(p, g)])
        # coef = bf16(0.5 / bf16(4 + 1e-6)) = 0.125 exactly; scaled = bf16(4*0.125)
        assert float(out[0][1].sum()) == 0.5

    def test_two_params_global_norm_sums_squares_of_rounded_norms(self):
        inner, wrapper = self._wrapped_clip()
        g1 = paddle.to_tensor([1.0, 0.0], dtype="float32")
        g2 = paddle.to_tensor([0.0, 2.0], dtype="float32")
        wrapper._dygraph_clip([(_param("a"), g1), (_param("b"), g2)])
        expected = float(_bf16(paddle.sqrt(paddle.to_tensor([1.0 + 4.0], dtype="float32"))))
        assert wrapper.stat["global_grad_norm"] == expected

    def test_fused_parameter_is_split_in_the_distributed_bucket(self):
        """The per-rank contribution must be the reference's several BF16 norms.

        Norming the fused block as one tensor here would put a different square
        into the all-reduce, so the HF tail after the reduction could not recover
        the reference value no matter how exact the collective is.
        """
        values, g1, g2, (whole, grouped) = _find_differing_grouping()
        inner, wrapper = self._wrapped_clip()
        wrapper._dygraph_clip([(_param("fused", groups=[g1, g2]), paddle.to_tensor([values], dtype="float32"))])
        assert wrapper.stat["global_grad_norm"] == grouped
        assert grouped != whole

    def test_double_wrapped_clip_still_uses_the_hf_recipe(self):
        """``MoEHybridParallelOptimizer`` wraps the already-wrapped ``_grad_clip``
        again for each ``_param_groups`` entry, so the HF clip sits two levels
        down for those groups."""
        inner = _clip(1.0)
        wrapper = MoEHybridParallelClipGrad(
            MoEHybridParallelClipGrad(inner, hcg=None, timers=None), hcg=None, timers=None
        )
        wrapper._global_norm = lambda *a, **k: None
        values, g1, g2, (whole, grouped) = _find_differing_grouping()
        wrapper._dygraph_clip([(_param("fused", groups=[g1, g2]), paddle.to_tensor([values], dtype="float32"))])
        assert wrapper.stat["global_grad_norm"] == grouped


class TestHFBitexactHybridParallelClipGrad:
    """The stock ``HybridParallelClipGrad`` replacement keeps the HF recipe.

    ``_global_norm`` is the only part that talks to the collectives; it is stubbed
    out here, which is exactly the single-rank case (nothing to reduce) and leaves
    the rounding-sensitive parts -- the per-rank BF16 per-tensor norms, the
    reference's split of fused parameters, and torch's coefficient -- under test.
    Real TP/PP/sharding collectives are not covered by these unit tests.
    """

    def _wrapped_clip(self, norm=1.0):
        inner = _clip(norm)
        wrapper = HFBitexactHybridParallelClipGrad(inner, hcg=None, split_norm_comm=False, timers=None)
        wrapper._global_norm = lambda *a, **k: None
        return inner, wrapper

    def test_matches_the_single_card_clip(self):
        inner, wrapper = self._wrapped_clip()
        reference = _clip(1.0)
        reference._dygraph_clip([(_param("w"), paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32"))])
        out = wrapper._dygraph_clip([(_param("w"), paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32"))])
        assert inner.last_global_norm == reference.last_global_norm
        assert inner.last_clip_coef == reference.last_clip_coef
        assert out[0][1] is not None

    def test_fused_parameter_is_split_like_the_reference(self):
        values, g1, g2, (whole, grouped) = _find_differing_grouping()
        inner, wrapper = self._wrapped_clip()
        wrapper._dygraph_clip([(_param("fused", groups=[g1, g2]), paddle.to_tensor([values], dtype="float32"))])
        assert inner.last_global_norm == grouped
        assert grouped != whole

    def test_distributed_and_replicated_params_go_to_separate_buckets(self):
        """``_global_norm`` reduces the two buckets over different groups, so a
        parameter must land in the one matching ``is_distributed``."""
        inner, wrapper = self._wrapped_clip()
        seen = {}

        def record(dist, not_dist):
            seen["dist"] = float(dist)
            seen["not_dist"] = float(not_dist)

        wrapper._global_norm = record
        # 3.0 is exact in BF16, so each bucket holds exactly the square.
        wrapper._dygraph_clip(
            [
                (_param("mp", is_distributed=True), paddle.to_tensor([3.0], dtype="float32")),
                (_param("replicated"), paddle.to_tensor([4.0], dtype="float32")),
            ]
        )
        assert seen == {"dist": 9.0, "not_dist": 16.0}

    def test_pipeline_shared_parameter_counted_once(self):
        inner, wrapper = self._wrapped_clip()
        shared = _param("shared")
        shared.is_firstly_shared = False
        wrapper._dygraph_clip(
            [
                (_param("w"), paddle.to_tensor([3.0], dtype="float32")),
                (shared, paddle.to_tensor([4.0], dtype="float32")),
            ]
        )
        # only the 3.0 contributes: bf16(sqrt(9)) == 3.0, not bf16(sqrt(25)) == 5.0
        assert inner.last_global_norm == 3.0

    def test_scales_bf16_rounded(self):
        inner, wrapper = self._wrapped_clip(norm=0.5)
        g = paddle.to_tensor([4.0, 0.0], dtype="float32")
        out = wrapper._dygraph_clip([(_param("w"), g)])
        assert float(out[0][1].sum()) == 0.5


class TestRestoreHFBitexactClip:
    """``HybridParallelOptimizer`` swaps ``_grad_clip`` for its own wrapper; the
    trainer has to put the HF one back or TP/PP/sharding silently reverts to
    paddle's global-norm formula."""

    @staticmethod
    def _dist_optimizer(clip):
        inner = SimpleNamespace(_grad_clip=HybridParallelClipGrad(clip, hcg=None), _param_groups=None)
        return SimpleNamespace(_inner_opt=inner), inner

    def test_hf_clip_is_restored(self):
        dist_opt, inner = self._dist_optimizer(_clip(1.0))
        assert restore_hf_bitexact_clip(dist_opt) is True
        assert isinstance(inner._grad_clip, HFBitexactHybridParallelClipGrad)

    def test_other_clips_are_left_alone(self):
        dist_opt, inner = self._dist_optimizer(paddle.nn.ClipGradByGlobalNorm(1.0))
        original = inner._grad_clip
        assert restore_hf_bitexact_clip(dist_opt) is False
        assert inner._grad_clip is original

    def test_moe_wrapper_is_left_alone(self):
        """``MoEHybridParallelClipGrad`` already keeps the recipe itself."""
        inner = SimpleNamespace(
            _grad_clip=MoEHybridParallelClipGrad(_clip(1.0), hcg=None, timers=None),
            _param_groups=None,
        )
        dist_opt = SimpleNamespace(_inner_opt=inner)
        original = inner._grad_clip
        assert restore_hf_bitexact_clip(dist_opt) is False
        assert inner._grad_clip is original

    def test_param_groups_are_restored(self):
        group = {"grad_clip": HybridParallelClipGrad(_clip(1.0), hcg=None)}
        inner = SimpleNamespace(_grad_clip=None, _param_groups=[group])
        assert restore_hf_bitexact_clip(SimpleNamespace(_inner_opt=inner)) is True
        assert isinstance(group["grad_clip"], HFBitexactHybridParallelClipGrad)

    def test_double_wrapped_param_group_is_restored(self):
        """``HybridParallelOptimizer`` wraps the already-wrapped ``_grad_clip``
        again for each ``_param_groups`` entry, so a group's clip nests two
        wrappers. Matching one level of ``_clip`` would leave those groups on
        paddle's formula while the rest of the model used torch's."""
        hf = _clip(1.0)
        outer = HybridParallelClipGrad(HybridParallelClipGrad(hf, hcg=None), hcg=None)
        group = {"grad_clip": outer}
        inner = SimpleNamespace(_grad_clip=outer, _param_groups=[group])
        assert restore_hf_bitexact_clip(SimpleNamespace(_inner_opt=inner)) is True
        for restored in (inner._grad_clip, group["grad_clip"]):
            assert isinstance(restored, HFBitexactHybridParallelClipGrad)
            # the double wrap is collapsed: reducing twice would square the norm
            assert restored._clip is hf

    def test_restoring_twice_is_stable(self):
        dist_opt, inner = self._dist_optimizer(_clip(1.0))
        restore_hf_bitexact_clip(dist_opt)
        first = inner._grad_clip
        assert restore_hf_bitexact_clip(dist_opt) is False
        assert inner._grad_clip is first


class TestUnwrapHFBitexactClip:
    def test_finds_the_clip_through_nested_wrappers(self):
        hf = _clip(1.0)
        nested = HybridParallelClipGrad(HybridParallelClipGrad(hf, hcg=None), hcg=None)
        assert unwrap_hf_bitexact_clip(nested) is hf
        assert unwrap_hf_bitexact_clip(hf) is hf

    def test_returns_none_for_other_clips(self):
        assert unwrap_hf_bitexact_clip(None) is None
        assert unwrap_hf_bitexact_clip(paddle.nn.ClipGradByGlobalNorm(1.0)) is None
        plain = HybridParallelClipGrad(paddle.nn.ClipGradByGlobalNorm(1.0), hcg=None)
        assert unwrap_hf_bitexact_clip(plain) is None


class TestNormGroupContract:
    """A PaddleFleet without the registration must fail loudly, not silently norm
    every fused projection as one block."""

    @staticmethod
    def _model(*params):
        return SimpleNamespace(parameters=lambda: list(params))

    def test_partition_size_counts_groups(self):
        plain = _param("plain")
        plain.stop_gradient = False
        fused = _param("fused", groups=[[0, 1], [2, 3]])
        fused.stop_gradient = False
        assert hf_norm_partition_size([plain, fused]) == (2, 3)

    def test_raises_when_nothing_registers_groups(self):
        plain = _param("plain")
        plain.stop_gradient = False
        with pytest.raises(RuntimeError, match="hf_norm_groups"):
            verify_hf_norm_groups_registered(self._model(plain))

    def test_returns_partition_sizes_when_registered(self):
        plain = _param("plain")
        plain.stop_gradient = False
        fused = _param("fused", groups=[[0, 1], [2, 3], [4, 5]])
        fused.stop_gradient = False
        assert verify_hf_norm_groups_registered(self._model(plain, fused)) == (2, 4)

    def test_frozen_parameters_are_ignored(self):
        frozen = _param("frozen", groups=[[0], [1]])
        frozen.stop_gradient = True
        plain = _param("plain")
        plain.stop_gradient = False
        with pytest.raises(RuntimeError, match="hf_norm_groups"):
            verify_hf_norm_groups_registered(self._model(frozen, plain))
