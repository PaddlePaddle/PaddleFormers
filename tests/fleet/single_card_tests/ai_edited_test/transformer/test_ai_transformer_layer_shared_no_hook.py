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

# [AI-EDITED] Unit tests for the model-side "no-hook shared color" coloring that
# supports sharding-stage1 comm-overlap when the MTP layer shares (aliases) the
# backbone's last transformer layer weights. Because Paddle now forbids
# reassigning a parameter's ``color``, the coloring is split into two single
# assignment sites:
#   - MoE expert params: colored in MoELayer.set_layer_number/_color_expert_params
#   - dense params:       colored in TransformerLayer._mark_shared_no_hook_params
# 覆盖: is_mtp_shared_last_layer 判定 + moe 专家染色 + dense 染色(跳过已染色).
import types
import unittest

from paddleformers.fleet.transformer.moe.moe_layer import (
    MoELayer,
)
from paddleformers.fleet.transformer.transformer_layer import (
    TransformerLayer,
    is_mtp_shared_last_layer,
)


class _FakeParam:
    """Minimal parameter stand-in carrying a mutable ``color`` attribute."""

    def __init__(self, color=-1, name="p"):
        self.color = color
        self.name = name


def _make_config(**overrides):
    """Build a config namespace with the attrs the logic reads."""
    cfg = {
        "mtp_shared_last_layer": True,
        "stage1_overlap": True,
        "mtp_num_layers": 0,
        "num_nextn_predict_layers": 1,
        "num_hidden_layers": 2,
        "num_empty_layers_add_in_head": 0,
    }
    cfg.update(overrides)
    return types.SimpleNamespace(**cfg)


# PLACEHOLDER_APPEND


class TestIsMtpSharedLastLayer(unittest.TestCase):
    """is_mtp_shared_last_layer 判定逻辑
    The shared-last-layer predicate used by both coloring sites."""

    def test_sharing_disabled(self):
        """mtp_shared_last_layer=False -> False"""
        cfg = _make_config(mtp_shared_last_layer=False)
        self.assertFalse(is_mtp_shared_last_layer(cfg, 1, False))

    def test_stage1_overlap_disabled(self):
        """stage1_overlap=False -> False (无 overlap hook, 无需染色)"""
        cfg = _make_config(stage1_overlap=False)
        self.assertFalse(is_mtp_shared_last_layer(cfg, 1, False))

    def test_stage1_overlap_attr_missing(self):
        """config 缺失 stage1_overlap 属性 (getattr -> False) -> False"""
        cfg = _make_config()
        del cfg.stage1_overlap
        self.assertFalse(is_mtp_shared_last_layer(cfg, 1, False))

    def test_no_mtp_layers(self):
        """mtp_num_layers 与 num_nextn_predict_layers 均为 0 -> False"""
        cfg = _make_config(mtp_num_layers=0, num_nextn_predict_layers=0)
        self.assertFalse(is_mtp_shared_last_layer(cfg, 1, False))

    def test_is_mtp_layer(self):
        """MTP 层自身 (is_mtp_layer=True) -> False (其 param 是别名)"""
        self.assertFalse(is_mtp_shared_last_layer(_make_config(), 1, True))

    def test_not_last_layer(self):
        """非最后一层 -> False (last_layer_number = 2-1+0 = 1)"""
        self.assertFalse(is_mtp_shared_last_layer(_make_config(), 0, False))

    def test_last_layer_via_num_nextn(self):
        """通过 num_nextn_predict_layers 判定 MTP 存在 -> True"""
        cfg = _make_config(mtp_num_layers=0, num_nextn_predict_layers=1)
        self.assertTrue(is_mtp_shared_last_layer(cfg, 1, False))

    def test_last_layer_via_mtp_num_layers(self):
        """通过 mtp_num_layers 判定 MTP 存在 -> True"""
        cfg = _make_config(mtp_num_layers=2, num_nextn_predict_layers=0)
        self.assertTrue(is_mtp_shared_last_layer(cfg, 1, False))

    def test_empty_layers_offset(self):
        """num_empty_layers_add_in_head 影响最后一层编号
        last_layer_number = 2 - 1 + 3 = 4"""
        cfg = _make_config(num_hidden_layers=2, num_empty_layers_add_in_head=3)
        self.assertFalse(is_mtp_shared_last_layer(cfg, 1, False))
        self.assertTrue(is_mtp_shared_last_layer(cfg, 4, False))


# PLACEHOLDER_APPEND2


def _make_moe_stub(
    config, layer_number, is_mtp_layer, params, ep_size=2, fusion=True
):
    """Build a stub 'self' for MoELayer._color_expert_params."""
    experts = types.SimpleNamespace(parameters=lambda: params)
    ns = types.SimpleNamespace(
        config=config,
        layer_number=layer_number,
        is_mtp_layer=is_mtp_layer,
        expert_model_parallel_size=ep_size,
        moe_grad_group="GRAD_GROUP",
    )
    if fusion:
        ns.grouped_gemm_experts = experts
    else:
        ns.experts = experts
    return ns


class TestColorExpertParams(unittest.TestCase):
    """MoELayer._color_expert_params: 专家参数唯一染色点
    Experts get colored exactly once, no-hook only on the shared last layer."""

    def test_last_layer_uses_no_hook(self):
        """最后一层 + mtp_share -> moe_weight_no_hook, 带 moe_grad_group"""
        p = _FakeParam(-1)
        stub = _make_moe_stub(_make_config(), 1, False, [p])
        MoELayer._color_expert_params(stub)
        self.assertEqual(
            p.color, {"color": "moe_weight_no_hook", "group": "GRAD_GROUP"}
        )

    def test_non_last_layer_uses_moe_expert(self):
        """非最后一层 -> 普通 moe_expert"""
        p = _FakeParam(-1)
        stub = _make_moe_stub(_make_config(), 0, False, [p])
        MoELayer._color_expert_params(stub)
        self.assertEqual(
            p.color, {"color": "moe_expert", "group": "GRAD_GROUP"}
        )

    def test_sharing_off_uses_moe_expert(self):
        """未开共享 -> moe_expert"""
        p = _FakeParam(-1)
        cfg = _make_config(mtp_shared_last_layer=False)
        stub = _make_moe_stub(cfg, 1, False, [p])
        MoELayer._color_expert_params(stub)
        self.assertEqual(
            p.color, {"color": "moe_expert", "group": "GRAD_GROUP"}
        )

    def test_stage1_overlap_off_uses_moe_expert(self):
        """stage1_overlap 关闭 -> 最后一层专家也只用普通 moe_expert, 不染 no_hook"""
        p = _FakeParam(-1)
        cfg = _make_config(stage1_overlap=False)
        stub = _make_moe_stub(cfg, 1, False, [p])
        MoELayer._color_expert_params(stub)
        self.assertEqual(
            p.color, {"color": "moe_expert", "group": "GRAD_GROUP"}
        )

    def test_non_fusion_experts_branch(self):
        """无 grouped_gemm_experts 时走 self.experts 分支"""
        p = _FakeParam(-1)
        stub = _make_moe_stub(_make_config(), 1, False, [p], fusion=False)
        MoELayer._color_expert_params(stub)
        self.assertEqual(p.color["color"], "moe_weight_no_hook")

    def test_ep_le_one_noop(self):
        """expert_model_parallel_size <= 1 -> 不染色 (无 EP)"""
        p = _FakeParam(-1)
        stub = _make_moe_stub(_make_config(), 1, False, [p], ep_size=1)
        MoELayer._color_expert_params(stub)
        self.assertEqual(p.color, -1)

    def test_already_colored_param_skipped(self):
        """构造期已染色 (dict color) 的参数被跳过, 不二次赋值
        Params colored at construction (non-shared-MTP case) are skipped so
        Paddle's no-reassign-color constraint is not violated."""
        grp = object()
        p = _FakeParam({"color": "moe_expert", "group": grp})
        # Even on the shared last layer, an already-colored param is left as-is.
        stub = _make_moe_stub(_make_config(), 1, False, [p])
        MoELayer._color_expert_params(stub)
        self.assertEqual(p.color, {"color": "moe_expert", "group": grp})


# PLACEHOLDER_APPEND3


def _make_layer(config, layer_number, is_mtp_layer, params):
    """Build a stub 'self' for TransformerLayer._mark_shared_no_hook_params."""
    return types.SimpleNamespace(
        config=config,
        layer_number=layer_number,
        is_mtp_layer=is_mtp_layer,
        parameters=lambda: params,
    )


def _mark(stub):
    TransformerLayer._mark_shared_no_hook_params(stub)


class TestMarkSharedNoHookDense(unittest.TestCase):
    """TransformerLayer._mark_shared_no_hook_params: 只染色未染色的 dense 参数
    Only uncolored dense params are colored; moe/colored params are skipped."""

    def test_early_return_not_shared_last_layer(self):
        """非共享最后一层 -> dense 不染色 (color 保持 -1)"""
        p = _FakeParam(-1)
        stub = _make_layer(
            _make_config(), layer_number=0, is_mtp_layer=False, params=[p]
        )
        _mark(stub)
        self.assertEqual(p.color, -1)

    def test_stage1_overlap_off_dense_not_colored(self):
        """stage1_overlap 关闭 -> 最后一层 dense 也不染色 (color 保持 -1)"""
        p = _FakeParam(-1)
        cfg = _make_config(stage1_overlap=False)
        stub = _make_layer(cfg, layer_number=1, is_mtp_layer=False, params=[p])
        _mark(stub)
        self.assertEqual(p.color, -1)

    def test_dense_param_gets_no_hook(self):
        """最后一层: 未染色 dense (color=-1) -> dense_weight_no_hook, 无 group"""
        p = _FakeParam(-1)
        stub = _make_layer(
            _make_config(), layer_number=1, is_mtp_layer=False, params=[p]
        )
        _mark(stub)
        self.assertEqual(p.color, {"color": "dense_weight_no_hook"})

    def test_uncolored_param_none_gets_no_hook(self):
        """color 属性缺失 (getattr -> None) 的 dense 也应被染色"""
        p = _FakeParam(-1)
        del p.color  # simulate a param that never had a color attribute
        stub = _make_layer(
            _make_config(), layer_number=1, is_mtp_layer=False, params=[p]
        )
        _mark(stub)
        self.assertEqual(p.color, {"color": "dense_weight_no_hook"})

    def test_moe_colored_param_skipped(self):
        """已染色的 moe 参数 (dict color) 应被跳过, 不重染 (Paddle 禁止二次染色)"""
        moe = _FakeParam({"color": "moe_weight_no_hook", "group": object()})
        stub = _make_layer(
            _make_config(), layer_number=1, is_mtp_layer=False, params=[moe]
        )
        _mark(stub)
        self.assertEqual(moe.color["color"], "moe_weight_no_hook")

    def test_mixed_params(self):
        """dense 与 moe 混合: 仅 dense 被染色, moe 原样保留"""
        dense = _FakeParam(-1)
        moe = _FakeParam({"color": "moe_expert", "group": object()})
        stub = _make_layer(
            _make_config(),
            layer_number=1,
            is_mtp_layer=False,
            params=[dense, moe],
        )
        _mark(stub)
        self.assertEqual(dense.color, {"color": "dense_weight_no_hook"})
        self.assertEqual(moe.color["color"], "moe_expert")


class TestHyperConnectionSharedNoHook(unittest.TestCase):
    """HyperConnectionTransformerLayer: hyper-connection 子模块在 super().__init__()
    之后才创建, 其参数在第一次 _mark_shared_no_hook_params 时还不存在。
    再次调用应给这些后创建的参数染色, 且对已染色的基类参数幂等。
    Regression for the bug where late-created hyper-connection params
    (mapping_proj.weight, alpha_pre/post/res, bias) missed the no-hook color."""

    def test_second_call_colors_late_params(self):
        """第二次调用给后创建的 hyper-connection 参数补染 dense_weight_no_hook"""
        # Phase 1: base params present at super().__init__() time.
        dense = _FakeParam(-1, name="dense")
        moe = _FakeParam(
            {"color": "moe_weight_no_hook", "group": object()}, name="moe"
        )
        base_params = [dense, moe]
        stub = _make_layer(
            _make_config(),
            layer_number=1,
            is_mtp_layer=False,
            params=base_params,
        )
        _mark(stub)
        self.assertEqual(dense.color, {"color": "dense_weight_no_hook"})
        self.assertEqual(moe.color["color"], "moe_weight_no_hook")

        # Phase 2: hyper-connection submodules created -> new uncolored params.
        mapping_proj = _FakeParam(-1, name="mapping_proj.weight")
        alpha = _FakeParam(-1, name="alpha_res")
        hc_bias = _FakeParam(-1, name="bias")
        base_params.extend([mapping_proj, alpha, hc_bias])
        _mark(stub)

        # Late params now colored.
        for p in (mapping_proj, alpha, hc_bias):
            self.assertEqual(
                p.color, {"color": "dense_weight_no_hook"}, msg=p.name
            )
        # Base params unchanged (idempotent / no reassignment).
        self.assertEqual(dense.color, {"color": "dense_weight_no_hook"})
        self.assertEqual(moe.color["color"], "moe_weight_no_hook")

    def test_second_call_noop_when_not_shared(self):
        """非共享最后一层: 两次调用后 hyper-connection 参数仍不染色"""
        dense = _FakeParam(-1)
        params = [dense]
        stub = _make_layer(
            _make_config(), layer_number=0, is_mtp_layer=False, params=params
        )
        _mark(stub)
        mapping_proj = _FakeParam(-1, name="mapping_proj.weight")
        params.append(mapping_proj)
        _mark(stub)
        self.assertEqual(dense.color, -1)
        self.assertEqual(mapping_proj.color, -1)


if __name__ == "__main__":
    unittest.main()
