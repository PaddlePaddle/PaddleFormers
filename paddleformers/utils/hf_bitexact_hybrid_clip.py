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

"""Keep the HF bit-exact clip recipe under ``HybridParallelOptimizer``.

``HFBitexactClipGradByGlobalNorm`` derives from ``nn.ClipGradByGlobalNorm``,
because that is what the optimizer machinery dispatches on. The side effect is
that ``HybridParallelOptimizer.__init__`` recognizes it by ``isinstance`` and
replaces ``_grad_clip`` with paddle's own ``HybridParallelClipGrad``, whose
``_dygraph_clip`` recomputes the global norm with paddle's formula. Under TP, PP
or sharding the selected recipe would therefore be dropped without a word, and
the run would look aligned while feeding different values to the optimizer.

``MoEHybridParallelClipGrad`` already handles this for hybrid expert parallel by
recognizing the inner clip. This module does the same for the stock wrapper: it
substitutes a subclass that keeps paddle's collective schedule (the inherited
``_global_norm``, including its ``split_norm_comm`` and shared-parameter
handling) but computes the per-rank contributions and the final coefficient with
torch's recipe.

The only unavoidable deviation from the single-GPU reference is the reduction
order of the all-reduce, which stays FP32 and sums exact squares of BF16 values.
"""

from __future__ import annotations

import paddle
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
    DygraphShardingOptimizer,
    DygraphShardingOptimizerV2,
)
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.hybrid_parallel_optimizer import (
    HybridParallelClipGrad,
)
from paddle.distributed.fleet.utils.hybrid_parallel_util import unwrap_optimizer
from paddle.distributed.fleet.utils.mix_precision_utils import MixPrecisionOptimizer
from paddle.framework import core
from paddle.nn import clip

from .hf_bitexact_clip import (
    _hf_clip_coef,
    _hf_global_norm,
    _hf_scale_grads,
    hf_param_norm_sq,
    unwrap_hf_bitexact_clip,
)
from .log import logger

__all__ = [
    "HFBitexactHybridParallelClipGrad",
    "restore_hf_bitexact_clip",
]

#: The optimizer wrappers paddle unwraps before installing its clip wrapper. The
#: replacement has to unwrap the same way to reach the object paddle mutated.
#: ``MuonShardingOptimizer`` is absent from XPU and older GPU builds.
_WRAPPER_OPTIMIZERS = (
    MixPrecisionOptimizer,
    DygraphShardingOptimizer,
    DygraphShardingOptimizerV2,
)
try:
    from paddle.distributed.fleet.meta_optimizers.muon_sharding_optimizer import (
        MuonShardingOptimizer,
    )

    _WRAPPER_OPTIMIZERS += (MuonShardingOptimizer,)
except ImportError:
    pass


class HFBitexactHybridParallelClipGrad(HybridParallelClipGrad):
    """``HybridParallelClipGrad`` with torch's roundings.

    Only two things change relative to the base class: the per-parameter
    contribution is the sum of squared BF16 per-tensor norms over the reference's
    tensor partition instead of ``clip._squared_l2_norm`` of the whole block, and
    the tail after the all-reduce follows ``clip_grad_norm_`` instead of paddle's
    ``max_norm / max(norm, max_norm)``. The dtype buckets the base class keeps are
    unnecessary here because every contribution is already an FP32 scalar.
    """

    @paddle.no_grad()
    def _dygraph_clip(self, params_grads):
        if self._timers:
            self._timers("dygraph-clip").start()

        sum_square_dist = []
        sum_square_not_dist = []
        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, "need_clip", True) is False:
                continue
            # A parameter shared across pipeline stages is counted on the stage
            # that owns it, exactly as the base class does, so the pp all-reduce
            # in ``_global_norm`` does not add it twice.
            if hasattr(p, "is_firstly_shared") and not getattr(p, "is_firstly_shared", True):
                continue
            merge_grad = g
            if g.type == core.VarDesc.VarType.SELECTED_ROWS:
                merge_grad = clip.merge_selected_rows(g)
                merge_grad = clip.get_tensor_from_selected_rows(merge_grad)
            sum_square = hf_param_norm_sq(p, merge_grad)
            if p.is_distributed:
                sum_square_dist.append(sum_square)
            else:
                sum_square_not_dist.append(sum_square)

        def total(squares):
            if not squares:
                return paddle.zeros((1,), dtype=paddle.float32)
            return paddle.add_n(squares)

        result = self._comm_and_clip(params_grads, total(sum_square_dist), total(sum_square_not_dist))
        if self._timers:
            self._timers("dygraph-clip").stop()
        return result

    def _comm_and_clip(self, params_grads, global_norm_var_dist, global_norm_var_not_dist):
        self._global_norm(global_norm_var_dist, global_norm_var_not_dist)

        global_norm = _hf_global_norm(global_norm_var_dist + global_norm_var_not_dist)
        clip_coef = _hf_clip_coef(global_norm, self.clip_norm)
        self._clip.last_global_norm = float(global_norm)
        self._clip.last_clip_coef = float(clip_coef)
        return _hf_scale_grads(params_grads, clip_coef)


def restore_hf_bitexact_clip(dist_optimizer) -> bool:
    """Swap paddle's clip wrapper back to the HF one on ``dist_optimizer``.

    ``HybridParallelOptimizer.__init__`` has already wrapped ``_grad_clip`` by the
    time the trainer sees the optimizer, so the substitution happens here rather
    than by preventing the wrap. The optimizer is unwrapped the same way paddle
    unwraps it before installing the wrapper, so the replacement lands on the
    object paddle actually mutated. Returns whether anything was replaced, and is
    a no-op unless the user asked for the HF target -- the wrapper it looks for
    only exists around ``HFBitexactClipGradByGlobalNorm``.
    """

    def rewrap(wrapper):
        if type(wrapper) is not HybridParallelClipGrad:
            # Already the HF wrapper, or MoEHybridParallelClipGrad, which keeps
            # the recipe itself.
            return None
        # Not ``wrapper._clip``: paddle wraps the already-wrapped ``_grad_clip``
        # a second time for every ``_param_groups`` entry, so a group's clip is
        # ``HybridParallelClipGrad(HybridParallelClipGrad(hf_clip))``. Rewrapping
        # the innermost HF clip both finds those and collapses the double wrap,
        # which would otherwise reduce the norm twice.
        hf_clip = unwrap_hf_bitexact_clip(wrapper)
        if hf_clip is None:
            return None
        return HFBitexactHybridParallelClipGrad(hf_clip, wrapper._hcg, wrapper.split_norm_comm, wrapper._timers)

    inner_opt = unwrap_optimizer(dist_optimizer._inner_opt, _WRAPPER_OPTIMIZERS)
    replaced = False
    replacement = rewrap(getattr(inner_opt, "_grad_clip", None))
    if replacement is not None:
        inner_opt._grad_clip = replacement
        replaced = True
    for group in getattr(inner_opt, "_param_groups", None) or []:
        if isinstance(group, dict) and "grad_clip" in group:
            replacement = rewrap(group["grad_clip"])
            if replacement is not None:
                group["grad_clip"] = replacement
                replaced = True

    if replaced:
        logger.info(
            "Restored HF bit-exact gradient clipping under HybridParallelOptimizer; "
            "per-rank BF16 per-tensor norms and torch's clip coefficient are preserved."
        )
    return replaced
