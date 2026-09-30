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

"""Tests for the HF bit-exact AdamW: triton bypass and optimizer-state step."""

import paddle
import pytest

import paddleformers.utils.optimizer as opt_mod
from paddleformers.utils.optimizer import AdamWCustom

_UNIQUE = iter(range(1000))


def _make_opt(param, *, hf=True, lr=0.1):
    return AdamWCustom(
        quantization_config=None,
        tensorwise_offload_optimizer=False,
        parameters=[param],
        learning_rate=lr,
        multi_precision=True,
        weight_decay=0.0,
        apply_decay_param_fun=None,
        accuracy_target="hf" if hf else "megatron",
    )


@pytest.fixture()
def param():
    p = paddle.create_parameter(
        [4],
        dtype="bfloat16",
        name=f"w{next(_UNIQUE)}",
        default_initializer=paddle.nn.initializer.Constant(1.0),
    )
    return p


def _step(opt, param):
    (param * 2.0).sum().backward()
    opt.step()
    opt.clear_grad()


def _step_acc_value(opt, param):
    return opt._get_accumulator_master("hf_bitexact_step", param).item()


class TestTritonBypass:
    def test_hf_target_never_calls_triton(self, param, monkeypatch):
        calls = []

        def fake_triton(*a, **k):
            calls.append(1)

        monkeypatch.setattr(opt_mod, "adamw_triton", fake_triton)
        opt = _make_opt(param, hf=True)
        _step(opt, param)
        assert calls == [], "HF target must not route through adamw_triton"
        # the python HF recipe did run (parameter actually moved)
        assert float(param[0]) != 1.0

    def test_non_hf_target_keeps_triton(self, param, monkeypatch):
        calls = []

        def fake_triton(*a, **k):
            calls.append(1)

        monkeypatch.setattr(opt_mod, "adamw_triton", fake_triton)
        opt = _make_opt(param, hf=False)
        _step(opt, param)
        assert len(calls) == 1, "non-HF target must keep the triton path"


class TestHFStepInOptimizerState:
    def test_step_accumulator_tracks_updates(self, param):
        opt = _make_opt(param, hf=True)
        assert "hf_bitexact_step" not in str(opt.state_dict())
        _step(opt, param)
        _step(opt, param)
        assert _step_acc_value(opt, param) == 2.0

    def test_step_accumulator_absent_for_non_hf(self, param):
        opt = _make_opt(param, hf=False)
        _step(opt, param)
        assert "hf_bitexact_step" not in str(opt.state_dict())

    def test_state_dict_roundtrip_resumes_step_count(self, param):
        # Reference trajectory: 3 continuous steps.
        ref = _make_opt(param, hf=True)
        _step(ref, param)
        _step(ref, param)
        _step(ref, param)
        ref_p = param.detach().clone()
        assert _step_acc_value(ref, param) == 3.0

        # Interrupted trajectory: 2 steps, save/restore, then 1 more step.
        param.set_value(paddle.full([4], 1.0, dtype="bfloat16"))
        resumed = _make_opt(param, hf=True)
        _step(resumed, param)
        _step(resumed, param)
        state = resumed.state_dict()
        assert state is not None

        resumed.set_state_dict(state)
        assert _step_acc_value(resumed, param) == 2.0, (
            "step count must be restored from optimizer state, "
            "not restart at 1 (bias correction would jump off the "
            "continuous HF trajectory)"
        )
        _step(resumed, param)

        both = paddle.equal_all(param.detach().astype("float32"), ref_p.astype("float32"))
        assert bool(both), "resumed step 3 must equal the continuous step 3"
