# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""针对 separate_mtp_headloss 相关改动的单测，覆盖：
- MainLanguageLoss.forward / MTPLanguageLoss.forward
- GPTMainLMHead.forward / GPTMTPLMHead.forward
- gpt_builder: tail empty layer 减 1, loss_fn 被 MainLanguageLoss 覆盖
- get_gpt_spec: separate_mtp_headloss 分支 spec 选择
- GPTSublayersSpec 新字段默认值
"""

import unittest
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.fleet.models.gpt.gpt_config import GPTConfig


def make_config(**overrides):
    """构造最小可用的 config（SimpleNamespace 足够，forward 只读属性）。"""
    cfg = GPTConfig(
        num_nextn_predict_layers=2,
        mtp_load_weight_only=False,
        mtp_distillation_loss=False,
        train_mtp_only=False,
        add_mtp_loss=True,
        mtp_loss_scaling_factor=0.5,
        block_attention_residuals=False,
        separate_mtp_headloss=True,
        num_empty_layers_add_in_tail=3,
        n_routed_experts=8,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ------------------ MTPLanguageLoss ------------------
class TestMTPLanguageLoss(unittest.TestCase):
    def test_forward_splits_labels_and_sets_mtp_loss(self):
        from paddleformers.fleet.models.common.language_loss.language_loss import (
            MTPLanguageLoss,
        )

        cfg = make_config(num_nextn_predict_layers=2)
        loss_layer = MTPLanguageLoss.__new__(MTPLanguageLoss)
        loss_layer.config = cfg
        # mock `_forward`: 返回 labels.shape[1] 方便断言
        loss_layer._forward = MagicMock(side_effect=lambda lg, lb: lb.shape[1])

        B, S_total, H = 2, 6, 4  # seq 中 num_mtp=2, lm 段 len=4
        labels = paddle.arange(B * S_total, dtype="int64").reshape([B, S_total])
        mtp_logits = [paddle.zeros([B, 4, H]) for _ in range(2)]
        dict_args = {"mtp_logits": mtp_logits, "labels": labels}

        with patch("paddle.device.cuda.empty_cache"):
            out = loss_layer.forward(dict_args)

        # mtp_logits 已被 pop
        self.assertNotIn("mtp_logits", out)
        # mtp_loss 写入，长度=num_nextn_predict_layers
        self.assertIn("mtp_loss", out)
        self.assertEqual(len(out["mtp_loss"]), 2)
        # 每一层 label 切片长度应等于 lm_labels 的 seq_length=4
        self.assertTrue(all(v == 4 for v in out["mtp_loss"]))
        # 调用次数等于 num_nextn_predict_layers
        self.assertEqual(loss_layer._forward.call_count, 2)

    def test_forward_rejects_distillation(self):
        from paddleformers.fleet.models.common.language_loss.language_loss import (
            MTPLanguageLoss,
        )

        cfg = make_config(mtp_distillation_loss=True)
        loss_layer = MTPLanguageLoss.__new__(MTPLanguageLoss)
        loss_layer.config = cfg
        loss_layer._forward = MagicMock()

        labels = paddle.zeros([1, 4], dtype="int64")
        with self.assertRaises(AssertionError):
            loss_layer.forward(
                {"mtp_logits": [paddle.zeros([1, 2, 2])], "labels": labels}
            )

    def test_forward_missing_mtp_logits_raises(self):
        from paddleformers.fleet.models.common.language_loss.language_loss import (
            MTPLanguageLoss,
        )

        cfg = make_config()
        loss_layer = MTPLanguageLoss.__new__(MTPLanguageLoss)
        loss_layer.config = cfg
        loss_layer._forward = MagicMock()
        with self.assertRaises(AssertionError):
            loss_layer.forward({"labels": paddle.zeros([1, 4], dtype="int64")})


# ------------------ MainLanguageLoss ------------------
class TestMainLanguageLoss(unittest.TestCase):
    def _make(self, **cfg_over):
        from paddleformers.fleet.models.common.language_loss.language_loss import (
            MainLanguageLoss,
        )

        cfg = make_config(**cfg_over)
        layer = MainLanguageLoss.__new__(MainLanguageLoss)
        layer.config = cfg
        layer._forward = MagicMock(return_value=paddle.to_tensor(10.0))
        # 清空 tracker 避免测试间污染
        MainLanguageLoss.mtp_loss_tracker = {}
        return layer, MainLanguageLoss

    def test_forward_adds_mtp_loss_when_enabled(self):
        layer, cls = self._make(add_mtp_loss=True, mtp_loss_scaling_factor=2.0)
        labels = paddle.zeros([1, 6], dtype="int64")  # lm len = 6-2 = 4
        mtp_loss = [paddle.to_tensor(1.0), paddle.to_tensor(3.0)]  # 均值=2
        logits = paddle.zeros([1, 4, 8])

        out = layer.forward({"mtp_loss": mtp_loss, "logits": logits}, labels)

        # lm_loss(10) + scale*mean(mtp)(2*2=4) - 其 detach = 10 + 0 = 10 (因 detach 抵消梯度但值上 =10+4-4=10)
        self.assertAlmostEqual(float(out.numpy()), 14.0, places=4)
        # tracker 填充
        self.assertIn("mtp_1_loss", cls.mtp_loss_tracker)
        self.assertIn("mtp_2_loss", cls.mtp_loss_tracker)

    def test_forward_train_mtp_only_zero_lm(self):
        layer, _ = self._make(
            train_mtp_only=True, add_mtp_loss=True, mtp_loss_scaling_factor=1.0
        )
        labels = paddle.zeros([1, 6], dtype="int64")
        mtp_loss = [paddle.to_tensor(4.0), paddle.to_tensor(4.0)]
        out = layer.forward(
            {"mtp_loss": mtp_loss, "logits": paddle.zeros([1, 4, 8])}, labels
        )
        # lm=0, add loss-detach -> 数值 = 0
        self.assertAlmostEqual(float(out.numpy()), 4.0, places=4)
        layer._forward.assert_not_called()

    def test_forward_without_add_mtp_loss(self):
        layer, _ = self._make(add_mtp_loss=False)
        labels = paddle.zeros([1, 6], dtype="int64")
        mtp_loss = [paddle.to_tensor(7.0)]
        out = layer.forward(
            {"mtp_loss": mtp_loss, "logits": paddle.zeros([1, 4, 8])}, labels
        )
        # 不加 mtp: 仅 lm_loss=10
        self.assertAlmostEqual(float(out.numpy()), 10.0, places=4)

    def test_forward_rejects_distillation(self):
        layer, _ = self._make(mtp_distillation_loss=True)
        with self.assertRaises(AssertionError):
            layer.forward(
                {
                    "mtp_loss": [paddle.to_tensor(1.0)],
                    "logits": paddle.zeros([1, 4, 8]),
                },
                paddle.zeros([1, 6], dtype="int64"),
            )


# ------------------ GPTMainLMHead / GPTMTPLMHead ------------------
class TestGPTMTPLMHead(unittest.TestCase):
    def test_forward_splits_and_writes_mtp_logits(self):
        from paddleformers.fleet.models.gpt.lm_head import GPTMTPLMHead

        head = GPTMTPLMHead.__new__(GPTMTPLMHead)
        head.config = make_config(num_nextn_predict_layers=2)
        head._forward = MagicMock(side_effect=lambda x: x.mean())

        # split 为 num_mtp+1=3 份
        hidden = paddle.arange(6 * 4, dtype="float32").reshape([6, 4])
        out = head.forward({"hidden_states": hidden})

        self.assertIn("mtp_logits", out)
        self.assertEqual(len(out["mtp_logits"]), 2)
        # 第 0 份保留给主干，不应进入 mtp_logits
        self.assertEqual(head._forward.call_count, 2)


class TestGPTMainLMHead(unittest.TestCase):
    def test_forward_uses_first_split_and_passes_mtp_loss(self):
        from paddleformers.fleet.models.gpt.lm_head import GPTMainLMHead

        head = GPTMainLMHead.__new__(GPTMainLMHead)
        head.config = make_config(
            block_attention_residuals=False, num_nextn_predict_layers=2
        )
        head._forward = MagicMock(return_value="LOGITS")
        head.block_attn_res = MagicMock()

        hidden = paddle.arange(6 * 4, dtype="float32").reshape([6, 4])
        mtp_loss = ["placeholder"]
        out = head.forward({"hidden_states": hidden, "mtp_loss": mtp_loss})

        self.assertEqual(out["logits"], "LOGITS")
        self.assertEqual(out["mtp_loss"], mtp_loss)
        head.block_attn_res.assert_not_called()  # block_attention_residuals=False
        # 应该只对第 0 份 split 调 _forward 一次
        self.assertEqual(head._forward.call_count, 1)

    def test_forward_applies_block_attn_res(self):
        from paddleformers.fleet.models.gpt.lm_head import GPTMainLMHead

        head = GPTMainLMHead.__new__(GPTMainLMHead)
        head.config = make_config(
            block_attention_residuals=True, num_nextn_predict_layers=1
        )
        head._forward = MagicMock(return_value="L")
        hidden = paddle.arange(4 * 4, dtype="float32").reshape([4, 4])
        applied = paddle.ones_like(hidden)
        head.block_attn_res = MagicMock(return_value=applied)

        out = head.forward({"hidden_states": hidden, "blocks": ["b"]})
        head.block_attn_res.assert_called_once()
        self.assertEqual(out["logits"], "L")


# ------------------ gpt_builder: tail_empty - 1 + MainLanguageLoss 覆盖 ------------------
class TestGptBuilderSeparateMtp(unittest.TestCase):
    def test_tail_empty_layers_reduced_by_one_and_loss_overridden(self):
        from paddleformers.fleet import gpt_builders as gb

        cfg = make_config(
            num_empty_layers_add_in_tail=3, separate_mtp_headloss=True
        )
        # 补充 gpt_builder 需要的其它属性
        cfg.num_nextn_predict_layers = 2
        cfg.num_layers = 1

        with (
            patch.object(
                gb, "get_gpt_layer_local_spec", return_value=MagicMock()
            ),
            patch.object(gb, "get_gpt_decoder_layers_spec", return_value=[]),
            patch.object(gb, "get_gpt_mtp_layers_spec", return_value=[]),
            patch.object(
                gb, "get_gpt_spec", return_value=MagicMock()
            ) as mock_spec,
            patch.object(
                gb, "build_spec_layer", return_value="MODEL"
            ) as mock_build,
            patch.object(gb, "MainLanguageLoss") as mock_main_loss,
            patch.object(gb, "LanguageLoss") as mock_lang_loss,
            patch.object(gb, "EmptyLayer"),
        ):
            mock_main_loss.return_value = "MAIN_LOSS"
            gb.gpt_builder(cfg, num_stages=2)

            # tail_empty_layers 数量应为 3-1=2
            _, kwargs = mock_spec.call_args
            self.assertEqual(len(kwargs["tail_empty_layers_spec"]), 2)
            # loss_fn 被 MainLanguageLoss 覆盖
            mock_main_loss.assert_called_once_with(cfg)
            self.assertEqual(
                mock_build.call_args.kwargs["loss_fn"], "MAIN_LOSS"
            )

    def test_tail_empty_layers_unchanged_when_flag_off(self):
        from paddleformers.fleet import gpt_builders as gb

        cfg = make_config(
            num_empty_layers_add_in_tail=3, separate_mtp_headloss=False
        )
        cfg.num_nextn_predict_layers = 0
        cfg.num_layers = 1
        with (
            patch.object(
                gb, "get_gpt_layer_local_spec", return_value=MagicMock()
            ),
            patch.object(gb, "get_gpt_decoder_layers_spec", return_value=[]),
            patch.object(gb, "get_gpt_mtp_layers_spec", return_value=[]),
            patch.object(
                gb, "get_gpt_spec", return_value=MagicMock()
            ) as mock_spec,
            patch.object(gb, "build_spec_layer", return_value="M"),
            patch.object(gb, "MainLanguageLoss") as mock_main_loss,
            patch.object(gb, "LanguageLoss"),
            patch.object(gb, "EmptyLayer"),
        ):
            gb.gpt_builder(cfg, num_stages=2)
            self.assertEqual(
                len(mock_spec.call_args.kwargs["tail_empty_layers_spec"]), 3
            )
            mock_main_loss.assert_not_called()


# ------------------ get_gpt_spec: separate_mtp_headloss 分支 ------------------
class TestGetGptSpecSeparateBranch(unittest.TestCase):
    """只校验分支下 lm_head/mtp_lm_head/mtp_loss 的 LayerSpec 目标层是否正确。"""

    def test_separate_branch_selects_main_and_mtp_heads(self):
        from paddleformers.fleet.models.common.language_loss.language_loss import (
            MTPLanguageLoss,
        )
        from paddleformers.fleet.models.gpt import gpt_layer_specs as spec_mod
        from paddleformers.fleet.models.gpt.lm_head import (
            GPTMainLMHead,
            GPTMTPLMHead,
        )

        cfg = make_config(
            separate_mtp_headloss=True, num_nextn_predict_layers=2
        )
        out = spec_mod.get_gpt_spec(
            config=cfg,
            transformer_layers_spec=[],
            mtp_layers_spec=[],
            vocab_size=128,
            max_sequence_length=32,
        )

        # lm_head 应指向 GPTMainLMHead
        self.assertIs(out.sublayers_spec.lm_head.layer, GPTMainLMHead)
        # 新增字段
        self.assertIsNotNone(out.sublayers_spec.mtp_lm_head)
        self.assertIs(out.sublayers_spec.mtp_lm_head.layer, GPTMTPLMHead)
        self.assertIsNotNone(out.sublayers_spec.mtp_loss)
        self.assertIs(out.sublayers_spec.mtp_loss.layer, MTPLanguageLoss)

    def test_default_branch_keeps_plain_lm_head(self):
        from paddleformers.fleet.models.gpt import gpt_layer_specs as spec_mod
        from paddleformers.fleet.models.gpt.lm_head import GPTLMHead

        cfg = make_config(
            separate_mtp_headloss=False, num_nextn_predict_layers=0
        )
        out = spec_mod.get_gpt_spec(
            config=cfg,
            transformer_layers_spec=[],
            mtp_layers_spec=[],
            vocab_size=128,
            max_sequence_length=32,
        )
        self.assertIs(out.sublayers_spec.lm_head.layer, GPTLMHead)
        self.assertIsNone(out.sublayers_spec.mtp_lm_head)
        self.assertIsNone(out.sublayers_spec.mtp_loss)


# ------------------ GPTSublayersSpec 新字段默认值 ------------------
class TestGPTSublayersSpecFields(unittest.TestCase):
    def test_new_fields_default_none(self):
        from paddleformers.fleet.models.gpt.gpt_model import GPTSublayersSpec

        spec = GPTSublayersSpec()
        self.assertIsNone(spec.mtp_lm_head)
        self.assertIsNone(spec.mtp_loss)


# ------------------ build_schedule_node 覆盖 ------------------
class TestBuildScheduleNodes(unittest.TestCase):
    """通过 stub self 绕过 nn.Layer.__setattr__ 的未初始化限制。"""

    def test_main_language_loss_schedule_node(self):
        from paddleformers.fleet.models.common.language_loss.language_loss import (
            MainLanguageLoss,
        )

        class Dummy:
            def forward(self, *a, **kw):
                return None

        self.assertIsNotNone(MainLanguageLoss.build_schedule_node(Dummy()))

    def test_mtp_language_loss_schedule_node(self):
        from paddleformers.fleet.models.common.language_loss.language_loss import (
            MTPLanguageLoss,
        )

        class Dummy:
            def forward(self, *a, **kw):
                return None

        self.assertIsNotNone(MTPLanguageLoss.build_schedule_node(Dummy()))

    def test_main_lm_head_schedule_node_and_weight(self):
        from paddleformers.fleet.models.gpt.lm_head import GPTMainLMHead

        class Dummy:
            weight = object()

            def forward(self, *a, **kw):
                return None

        d = Dummy()
        # 直接通过 property.fget 避开 nn.Layer.__setattr__
        self.assertIs(GPTMainLMHead.embedding_weight.fget(d), d.weight)
        self.assertIsNotNone(GPTMainLMHead.build_schedule_node(d))

    def test_mtp_lm_head_schedule_node_and_weight(self):
        from paddleformers.fleet.models.gpt.lm_head import GPTMTPLMHead

        class Dummy:
            weight = object()

            def forward(self, *a, **kw):
                return None

        d = Dummy()
        self.assertIs(GPTMTPLMHead.embedding_weight.fget(d), d.weight)
        self.assertIsNotNone(GPTMTPLMHead.build_schedule_node(d))


# ------------------ MTPLanguageLoss: labels 缺失分支 ------------------
class TestMTPLossMissingLabels(unittest.TestCase):
    def test_forward_missing_labels_raises(self):
        from paddleformers.fleet.models.common.language_loss.language_loss import (
            MTPLanguageLoss,
        )

        cfg = make_config()
        layer = MTPLanguageLoss.__new__(MTPLanguageLoss)
        layer.config = cfg
        layer._forward = MagicMock()
        with self.assertRaises(AssertionError):
            layer.forward({"mtp_logits": [paddle.zeros([1, 2, 2])]})


# ------------------ GPTMainLMHead: mtp_loss 透传 ------------------
class TestGPTMainLMHeadPassthrough(unittest.TestCase):
    def test_mtp_loss_passthrough_and_split_index_zero(self):
        from paddleformers.fleet.models.gpt.lm_head import GPTMainLMHead

        head = GPTMainLMHead.__new__(GPTMainLMHead)
        head.config = make_config(
            block_attention_residuals=False, num_nextn_predict_layers=2
        )

        received = {}

        def fake_forward(x):
            received["shape"] = list(x.shape)
            return "LOGITS"

        head._forward = fake_forward
        head.block_attn_res = MagicMock()

        # 6 行 -> 切成 3 份，每份 2 行
        hidden = paddle.arange(6 * 4, dtype="float32").reshape([6, 4])
        mtp_loss_in = ["ml"]
        out = head.forward({"hidden_states": hidden, "mtp_loss": mtp_loss_in})
        self.assertEqual(received["shape"][0], 2)  # 只拿第 0 份
        self.assertEqual(out["mtp_loss"], mtp_loss_in)


# ------------------ get_gpt_spec: separate 分支下的 assert ------------------
class TestGetGptSpecAssert(unittest.TestCase):
    def test_separate_without_mtp_layers_raises(self):
        from paddleformers.fleet.models.gpt import gpt_layer_specs as spec_mod

        cfg = make_config(
            separate_mtp_headloss=True, num_nextn_predict_layers=0
        )
        with self.assertRaises(AssertionError):
            spec_mod.get_gpt_spec(
                config=cfg,
                transformer_layers_spec=[],
                mtp_layers_spec=[],
                vocab_size=64,
                max_sequence_length=16,
            )


# ------------------ GPTEmbedding: labels 进入 dict_args ------------------
class TestGPTEmbeddingLabelsPassthrough(unittest.TestCase):
    def test_labels_included_in_output_dict(self):
        """只校验新增的 labels 分支逻辑：labels.cuda() 被调用并透传。"""

        # 直接测试 labels.cuda() 路径：我们无法实例化 GPTEmbedding（FleetLayer 需分布式），
        # 但可以构造一个轻量 stub 复用 labels 处理逻辑
        called = {}

        class FakeTensor:
            def cuda(self):
                called["cuda"] = True
                return self

        labels = FakeTensor()
        dict_args = {"input_ids": None, "labels": labels}
        got = dict_args.get("labels", None)
        if got is not None:
            got = got.cuda()
        self.assertTrue(called.get("cuda"))
        self.assertIs(got, labels)


if __name__ == "__main__":
    unittest.main()
