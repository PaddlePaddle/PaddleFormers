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

"""FP32 clipping preserves its result across layouts and parameter ownership."""

from types import SimpleNamespace

import paddle
import pytest

from paddleformers.trainer import Trainer
from paddleformers.utils.moe_hybrid_parallel_optimizer import MoEHybridParallelClipGrad
from paddleformers.utils.reproducible_norm import (
    ReproducibleClipGradByGlobalNorm,
    ReproducibleL2Norm,
)


@pytest.fixture(autouse=True)
def norm_device():
    paddle.set_device("gpu:0")


@pytest.mark.parametrize(
    "values,expected",
    [
        ([0.0], 0.0),
        ([3.0, 4.0], 25.0),
        ([4096.0, 1.0], 16777216.0),
        ([4096.0, 1.0, 1.0, 1.0], 16777220.0),
        ([2.0**-70], 2.0**-140),
        ([float("inf")], float("inf")),
        ([3e38], float("inf")),
    ],
)
def test_sum_squares_rounding(values, expected):
    norm = ReproducibleL2Norm()
    _, squared = norm.finish(norm.accumulate(norm.zeros(), norm.tensor(values)))
    assert squared.item() == expected


def test_nan_and_invalid_input():
    norm = ReproducibleL2Norm()
    actual, _ = norm.finish(norm.accumulate(norm.zeros(), norm.tensor([float("inf"), float("nan")])))
    assert paddle.isnan(actual).item()
    with pytest.raises(TypeError, match="FP32"):
        norm.accumulate(norm.zeros(), norm.tensor([1.0]).astype("bfloat16"))
    bins = norm.zeros()
    bins[22] = 2**40 + 1
    with pytest.raises(OverflowError):
        norm.finish(bins)


def test_layout_and_chunk_invariance():
    norm = ReproducibleL2Norm()
    gradient = paddle.arange(1, 12289, dtype="float32").reshape([96, 128]) / 16384
    whole = norm.accumulate(norm.zeros(), gradient)
    transposed = norm.accumulate(norm.zeros(), gradient.transpose([1, 0]), chunk_size=123)
    split = norm.accumulate(norm.zeros(), gradient.flatten()[:1234])
    split += norm.accumulate(norm.zeros(), gradient.flatten()[1234:])
    assert paddle.equal_all(whole, transposed).item()
    assert paddle.equal_all(whole, split).item()


def _parameter(name, **kwargs):
    return SimpleNamespace(
        name=name,
        is_distributed=False,
        need_clip=True,
        no_sync=False,
        _reset_grad_inplace_version=lambda _: None,
        **kwargs,
    )


@pytest.mark.parametrize("nested", [False, True])
def test_hybrid_owner_filter_and_coefficient(nested):
    clip = ReproducibleClipGradByGlobalNorm(1.0)
    if nested:
        clip = MoEHybridParallelClipGrad(clip, hcg=None)
    wrapper = MoEHybridParallelClipGrad(clip, hcg=None)
    wrapper._global_norm = lambda *args: None
    gradients = [paddle.to_tensor([3.0]), paddle.to_tensor([4.0]), paddle.to_tensor([12.0])]
    params = [_parameter("first"), _parameter("second"), _parameter("shared", is_firstly_shared=False)]
    wrapper._dygraph_clip(list(zip(params, gradients)))
    assert wrapper.stat["global_grad_norm"] == 5.0
    expected = paddle.to_tensor([3.0, 4.0, 12.0]) * (1.0 / (paddle.to_tensor([5.0]) + 1e-6))
    assert paddle.equal_all(paddle.concat(gradients), expected).item()


@pytest.mark.parametrize("accuracy,limit,expected", [(False, 1.0, False), (True, 1.0, True), (True, 0.0, False)])
def test_trainer_clip_selection(accuracy, limit, expected):
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(max_grad_norm=limit)
    trainer.model = SimpleNamespace(config=SimpleNamespace(use_accuracy_compatible=accuracy))
    clip = trainer._build_grad_clip()
    assert isinstance(clip, ReproducibleClipGradByGlobalNorm) is expected
    if limit == 0:
        assert clip is None


@pytest.mark.parametrize("moe", [False, True])
def test_trainer_norm_logging_keeps_the_clipping_recipe(monkeypatch, moe):
    from paddle.distributed import fleet
    from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.hybrid_parallel_optimizer import (
        HybridParallelClipGrad,
        HybridParallelOptimizer,
    )

    from paddleformers.utils.reproducible_norm import ReproducibleHybridParallelClipGrad

    inner_clip = ReproducibleClipGradByGlobalNorm(1.0)
    if moe:
        clip = MoEHybridParallelClipGrad(inner_clip, hcg=None)
    else:
        clip = HybridParallelClipGrad(HybridParallelClipGrad(inner_clip, hcg=None), hcg=None)
    # A real Trainer instrumentation wrapper surrounds the norm collective. Only
    # communication is omitted; these tests use one complete owner partition.
    monkeypatch.setattr(ReproducibleHybridParallelClipGrad, "_global_norm", lambda *args: None)
    if moe:
        clip._global_norm = lambda *args: None
    optimizer = HybridParallelOptimizer.__new__(HybridParallelOptimizer)
    optimizer._inner_opt = SimpleNamespace(_grad_clip=clip, _param_groups=None)
    monkeypatch.setattr(fleet, "distributed_optimizer", lambda _: optimizer)
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(use_expert_parallel=False, max_grad_norm=1.0, train_mtp_only=False)
    trainer.optimizer = optimizer
    trainer.global_training_logs = {}
    actual_optimizer = trainer._wrap_distributed_optimizer(optimizer._inner_opt)
    gradient = paddle.to_tensor([3.0, 4.0])
    actual_optimizer._inner_opt._grad_clip._dygraph_clip([(_parameter("w"), gradient)])
    assert trainer.global_training_logs["global_norm"] == 5.0
    expected = paddle.to_tensor([3.0, 4.0]) * (1.0 / (paddle.to_tensor([5.0]) + 1e-6))
    assert paddle.equal_all(gradient, expected).item()
