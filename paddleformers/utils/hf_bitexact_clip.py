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

"""Gradient clipping that reproduces torch's ``clip_grad_norm_`` bit-for-bit.

Only used when ``config.use_accuracy_compatible="hf"``; the default path is
untouched.

Why a separate clip is needed
-----------------------------
``paddle.nn.ClipGradByGlobalNorm`` and ``torch.nn.utils.clip_grad_norm_`` are the
same *formula* but a different *recipe*, and with a BF16 model the recipe decides
the bits:

===========================  ==================================  ==================================
step                         torch (transformers / accelerate)   paddle (ClipGradByGlobalNorm)
===========================  ==================================  ==================================
operand                      the BF16 ``p.grad``                 the FP32 ``p.main_grad``
per-tensor norm              ``_foreach_norm`` -> **BF16**       no per-tensor norm at all
global accumulation          squares of the BF16 per-tensor      squares of every element, summed
                             norms                               straight into one FP32 accumulator
global norm dtype            **BF16**                            FP32 (or FP64)
coefficient                  ``max_norm / (norm + 1e-6)``,       ``max_norm / max(norm, max_norm)``
                             then ``clamp(max=1.0)``, in BF16    in FP32
scaling                      BF16 grad * BF16 coef -> BF16       FP32 grad * FP32 coef -> FP32
===========================  ==================================  ==================================

The two differ in the value fed to the optimizer, so the trajectories separate on
the first clipped step. This class follows the torch column exactly:

1. per parameter: FP32 sum of squares of the gradient, ``sqrt``, **round to BF16**
   (matches ``torch._foreach_norm`` on a BF16 tensor -- FP32 opmath, BF16 result);
2. stack those BF16 per-tensor norms, FP32 sum of squares, ``sqrt``,
   **round to BF16** (matches ``torch.linalg.vector_norm`` over the stack);
3. ``den = bf16(fp32(norm) + fp32(1e-6))``; ``coef = bf16(fp32(max_norm) / fp32(den))``;
   ``coef = min(coef, 1.0)`` -- torch multiplies by the clamped coefficient
   unconditionally, so there is no "skip when below the threshold" branch;
4. every gradient is scaled with FP32 opmath and **rounded back to BF16**, then
   written into the FP32 ``main_grad`` buffer, so the buffer holds exactly the
   BF16 value torch's optimizer would read.

The pre-clip global norm and the clamped coefficient of the most recent step are
kept on the instance as ``last_global_norm`` / ``last_clip_coef``. Nothing is
printed or written to disk: this class only has to make the training step
match, and per-step dumps are of no use to CI/CE monitoring.
"""

from __future__ import annotations

from typing import Optional

import paddle
import paddle.nn as nn

from .accuracy_target import ACCURACY_TARGET_HF

__all__ = [
    "HFBitexactClipGradByGlobalNorm",
    "hf_bitexact_clip_enabled",
    "hf_norm_partition_size",
    "unwrap_hf_bitexact_clip",
    "verify_hf_norm_groups_registered",
]


def hf_bitexact_clip_enabled(accuracy_target) -> bool:
    """Whether ``accuracy_target`` selects the torch ``clip_grad_norm_`` recipe.

    Takes the value rather than reading a global so the model config stays the
    single source of truth; see ``Trainer._build_grad_clip``. The value reaching
    here has already been canonicalized by ``LlmMetaConfig.set_llm_config``, so a
    plain equality test is enough.
    """
    return accuracy_target == ACCURACY_TARGET_HF


def _bf16(value: paddle.Tensor) -> paddle.Tensor:
    """Round an FP32 scalar/tensor through BF16 and return it as FP32.

    Every ``_foreach_*`` / reduction kernel torch runs on a BF16 tensor allocates
    a BF16 output, so each of them rounds once. Keeping the value in FP32 between
    the roundings (rather than in a BF16 buffer) avoids a second rounding on read
    while still discarding exactly the bits torch discards.
    """
    return value.astype("bfloat16").astype("float32")


def unwrap_hf_bitexact_clip(grad_clip):
    """The HF clip inside ``grad_clip``, or ``None`` if there is none.

    Distributed wrappers keep the clip they replaced in ``_clip``, and paddle
    **nests** them: ``HybridParallelOptimizer.__init__`` wraps
    ``inner_opt._grad_clip`` once for the optimizer and then wraps *that already
    wrapped object* again for every entry of ``_param_groups``. A parameter
    group's clip is therefore ``Wrapper(Wrapper(HFBitexactClipGradByGlobalNorm))``,
    and matching on one level of ``_clip`` would miss it -- the group would fall
    back to paddle's global-norm formula while the rest of the model used torch's.
    """
    while grad_clip is not None:
        if isinstance(grad_clip, HFBitexactClipGradByGlobalNorm):
            return grad_clip
        grad_clip = getattr(grad_clip, "_clip", None)
    return None


# -- shared primitives ---------------------------------------------------------
#
# These are the *recipe* of ``torch.nn.utils.clip_grad_norm_`` and are used by the
# single-card clip below and by both distributed wrappers
# (``MoEHybridParallelClipGrad``, ``HFBitexactHybridParallelClipGrad``) so every
# path keeps the exact same roundings. In the distributed setting each rank holds
# a shard of the model, so the per-tensor BF16 norms are computed locally, their
# squares are what travel through the all-reduce, and the global norm /
# coefficient / scaling steps are shared verbatim.


def hf_norm_groups(param):
    """Column groups ``param`` must be split into before norming, or ``None``.

    PaddleFleet fuses projections the reference keeps separate (GDN's four
    ``in_proj_*``, attention's ``q/k/v_proj``, the shared expert's
    ``gate_proj``/``up_proj``) and publishes the reference's column partition on
    the parameter as ``hf_norm_groups``. Gradient clipping is **partition
    sensitive**: torch takes one BF16 per-tensor norm per ``nn.Linear`` and sums
    their squares, and ``bf16(sqrt(a^2+b^2))^2 != bf16(sqrt(a^2))^2 +
    bf16(sqrt(b^2))^2`` in general -- rounding each sub-norm to BF16 first
    discards different bits than rounding the combined one. Norming each group
    separately takes the global norm over the reference's tensor partition
    rather than the fused one.

    Parameters the reference also keeps fused (the routed experts' batched
    ``gate_up_proj``, the vision tower's ``linear_fc1``) publish nothing and are
    normed whole, which is what the reference does with them.
    """
    return getattr(param, "hf_norm_groups", None)


def hf_param_norms(param, g: paddle.Tensor) -> list:
    """The BF16 per-tensor norms ``param`` contributes, as FP32 scalars.

    Torch's ``_foreach_norm`` on a BF16 tensor uses FP32 opmath and rounds the
    result to BF16, which is the rounding each entry here reproduces. There is
    one entry for an unfused parameter and one per group for a fused one, which
    is what torch produces for the separate ``nn.Linear`` modules the groups
    stand for.
    """
    gf = g.astype("float32")
    groups = hf_norm_groups(param)
    if not groups:
        return [_bf16(paddle.sqrt(paddle.sum(gf * gf, dtype="float32")))]
    flat = gf.reshape([-1, gf.shape[-1]])
    norms = []
    for columns in groups:
        sub = flat.index_select(axis=-1, index=columns)
        norms.append(_bf16(paddle.sqrt(paddle.sum(sub * sub, dtype="float32"))))
    return norms


def hf_param_norm_sq(param, g: paddle.Tensor) -> paddle.Tensor:
    """Sum of the squared BF16 per-tensor norms of ``param``.

    Drop-in replacement for ``clip._squared_l2_norm`` in the distributed
    wrappers: it returns a single FP32 scalar, so their dtype bucketing and
    all-reduce need no change, while a fused parameter still contributes the
    reference's several norms instead of one over the whole block. Squaring a
    BF16 value is exact in FP32, so nothing is lost on the way to the reduction.
    """
    total = None
    for norm in hf_param_norms(param, g):
        squared = norm * norm
        total = squared if total is None else total + squared
    return total


def hf_norm_partition_size(parameters) -> tuple:
    """``(fused_tensors, reference_tensors)`` for ``parameters``.

    The two numbers are the size of the partition the global norm would be taken
    over without and with ``hf_norm_groups``. They differ by exactly the extra
    tensors the reference's separate ``nn.Linear`` modules add, which is the
    quantity that decides the global norm, so logging them makes a missing
    registration visible in the training log instead of only in the loss.
    """
    fused = 0
    reference = 0
    for param in parameters:
        fused += 1
        groups = hf_norm_groups(param)
        reference += len(groups) if groups else 1
    return fused, reference


def verify_hf_norm_groups_registered(model) -> tuple:
    """Fail if no parameter publishes ``hf_norm_groups``.

    The clip needs the reference's tensor partition, and the modules that fuse
    projections are the ones that know it, so they publish it (PaddleFleet's
    ``attention``, ``gated_delta_net`` and ``moe_shared_expert`` do this at
    construction time, guarded by the same accuracy target). If the installed
    PaddleFleet predates that, every fused projection is normed as one block: the
    run does not fail, it just stops being bit-exact once the coefficient drops
    below 1.0. That silent degradation is worse than a crash -- any numbers
    collected from such a run look aligned and are not -- so require at least one
    registration and say what is missing.

    Returns the ``(fused, reference)`` partition sizes for logging.
    """
    parameters = [p for p in model.parameters() if not p.stop_gradient]
    fused, reference = hf_norm_partition_size(parameters)
    if reference == fused:
        raise RuntimeError(
            "use_accuracy_compatible='hf' requires gradient clipping over the reference's "
            "tensor partition, but none of the "
            f"{fused} trainable parameters publishes 'hf_norm_groups'. The fused projections "
            "(attention qkv_proj, gated_delta_net in_proj, the shared expert's up_gate_proj) "
            "register it at construction time; an installed PaddleFleet without that support "
            "would norm each fused block as one tensor and silently lose bit-exactness as soon "
            "as the clip coefficient drops below 1.0. Update PaddleFleet, or set max_grad_norm=0 "
            "to run without clipping."
        )
    return fused, reference


def _hf_global_norm(total_sq: paddle.Tensor) -> paddle.Tensor:
    """``sqrt`` of the summed squared BF16 per-tensor norms, rounded to BF16."""
    return _bf16(paddle.sqrt(paddle.cast(total_sq, "float32")))


def _hf_clip_coef(global_norm: paddle.Tensor, clip_norm: float) -> paddle.Tensor:
    """Torch's coefficient: ``min(bf16(max / bf16(norm + 1e-6)), 1.0)``.

    One rounding per torch kernel, then the unconditional clamp -- torch
    multiplies by the clamped coefficient even when it is 1.0.
    """
    denom = _bf16(global_norm + paddle.full_like(global_norm, float(1e-6)))
    coef = _bf16(paddle.full_like(denom, float(clip_norm)) / denom)
    return paddle.minimum(coef, paddle.ones_like(coef))


def _hf_scale_grads(params_grads, coef: paddle.Tensor):
    """Scale every gradient with FP32 opmath, rounding back to its own dtype.

    Mirrors the tail of ``HFBitexactClipGradByGlobalNorm._dygraph_clip``: torch
    writes the coefficient into the BF16 ``p.grad`` in place, while this
    codebase promotes the FP32 ``main_grad`` back to the activation dtype. The
    FP32 (master) grads are rewritten in place; everything else is returned as a
    new tensor of the original dtype.
    """
    params_and_grads = []
    for p, g in params_grads:
        if g is None or not getattr(p, "need_clip", True):
            params_and_grads.append((p, g))
            continue
        scaled = _bf16(g.astype("float32") * coef)
        if g.dtype == paddle.float32:
            g[:] = scaled
            params_and_grads.append((p, g))
        else:
            params_and_grads.append((p, scaled.astype(g.dtype)))
    return params_and_grads


class HFBitexactClipGradByGlobalNorm(nn.ClipGradByGlobalNorm):
    """``ClipGradByGlobalNorm`` with torch's operation order and roundings."""

    def __init__(self, clip_norm: float, trainer=None) -> None:
        super().__init__(clip_norm)
        self._trainer = trainer
        #: pre-clip global norm of the most recent step (python float, BF16-exact)
        self.last_global_norm: Optional[float] = None
        #: clamped coefficient actually multiplied into the gradients
        self.last_clip_coef: Optional[float] = None

    # -- the clip itself -----------------------------------------------------
    @paddle.no_grad()
    def _dygraph_clip(self, params_grads):
        selected = [(p, g) for p, g in params_grads if g is not None and getattr(p, "need_clip", True)]
        if not selected:
            return params_grads

        # (1) per-tensor norm: FP32 sum of squares -> sqrt -> BF16, over the
        #     REFERENCE's tensor partition (fused projections are split first).
        per_tensor = []
        for p, g in selected:
            per_tensor.extend(hf_param_norms(p, g))

        # (2) global norm over the BF16 per-tensor norms: FP32 accumulate -> sqrt -> BF16.
        stacked = paddle.stack(per_tensor).astype("float32")
        global_norm = _hf_global_norm(paddle.sum(stacked * stacked, dtype="float32"))

        # (3) coefficient, one rounding per torch kernel, then the unconditional clamp.
        clip_coef = _hf_clip_coef(global_norm, self.clip_norm)

        self.last_global_norm = float(global_norm)
        self.last_clip_coef = float(clip_coef)

        # (4) scale with FP32 opmath and round back to BF16, in place.
        return _hf_scale_grads(params_grads, clip_coef)
