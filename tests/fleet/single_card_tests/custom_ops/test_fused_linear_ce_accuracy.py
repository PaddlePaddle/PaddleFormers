#!/usr/bin/env python3

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

"""
单测：对比 GPTLMHead + LanguageLoss 在
  baseline 分支 (fused_linear_ce_loss_chunk=0) 与
  fused   分支 (fused_linear_ce_loss_chunk>0) 下的精度。

weight 布局说明：
  - GPTLMHead.weight shape 为 [V, H]（ColumnParallelLinear TP-sharded）
  - fused 路径传入 kernel 的是 self.weight（即 [V, H]），
    kernel 内部用 paddle.compat.nn.functional.linear（PyTorch 语义，
    等价于 x @ weight.T）做 forward，grad_weight 按 [V, H] 顺序计算
    后在 backward 累加到 main_grad([V, H])。
"""

import os
import unittest

import numpy as np
import paddle

os.environ["FLAGS_cudnn_deterministic"] = "1"
os.environ["FLAGS_embedding_deterministic"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

from paddleformers.fleet.models.common.language_loss.language_loss import (
    LanguageLoss,
)
from paddleformers.fleet.models.gpt.lm_head import GPTLMHead
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


# ---------------------------------------------------------------------------
# 构造 TransformerConfig
# ---------------------------------------------------------------------------
def _make_config(hidden_size=64, vocab_size=256, fused_linear_ce_loss_chunk=0):
    """构造单测专用最简 TransformerConfig。

    关键配置：
      - fused_linear_ce_loss_chunk=0  → GPTLMHead 返回 logits tensor
        → LanguageLoss.forward_impl 走 baseline (CrossEntropy) 路径
      - fused_linear_ce_loss_chunk>0  → GPTLMHead 返回 (hidden, weight, bias) 元组
        → LanguageLoss.forward_impl 走 LigerFusedLinearCrossEntropyFunction 路径
      - parallel_output=False: 单卡，不走 ParallelCrossEntropy
      - sequence_parallel=False, context_parallel_size=1: 无需分布式
    """
    cfg = TransformerConfig(
        hidden_size=hidden_size,
        # 关闭所有并行/分布式特性
        parallel_output=False,
        sequence_parallel=False,
        context_parallel_size=1,
        tensor_model_parallel_size=1,
        # 关闭 recompute / subbatch
        recompute_modules=None,
        loss_subbatch_sequence_length=-1,
        # 关闭 MTP / block_attn_res
        num_nextn_predict_layers=0,
        block_attention_residuals=False,
        # fp32 参数
        params_dtype=paddle.float32,
        # 跳过 GPU RNG tracker 初始化（单测手工赋权重）
        perform_initialization=False,
    )
    # fused_linear_ce_loss_chunk / gpt_model_use_experimental_version 在
    # site-packages 版中可能不是 dataclass 字段，构造后动态赋值兼容两种情况
    cfg.fused_linear_ce_loss_chunk = fused_linear_ce_loss_chunk
    cfg.gpt_model_use_experimental_version = False
    cfg._vocab_size_for_test = vocab_size
    return cfg


# ---------------------------------------------------------------------------
# 构造 (baseline, fused) 模型对，共享权重
# ---------------------------------------------------------------------------
def _make_pair(hidden_size=64, vocab_size=256, num_chunks=1):
    """构造共享权重的 (baseline_lm_head+loss, fused_lm_head+loss) 对。"""

    def _init_method(tensor):
        paddle.nn.initializer.Normal(std=0.02)(tensor)

    def _make_lm_head(cfg):
        lm_head = GPTLMHead(
            input_size=hidden_size,
            output_size=vocab_size,
            config=cfg,
            init_method=_init_method,
            bias=True,
            gather_output=False,
            skip_weight_param_allocation=False,
        )
        return lm_head

    cfg_bsl = _make_config(
        hidden_size, vocab_size, fused_linear_ce_loss_chunk=0
    )
    cfg_fused = _make_config(
        hidden_size, vocab_size, fused_linear_ce_loss_chunk=num_chunks
    )

    lm_head_bsl = _make_lm_head(cfg_bsl)
    lm_head_fused = _make_lm_head(cfg_fused)
    loss_bsl = LanguageLoss(config=cfg_bsl)
    loss_fused = LanguageLoss(config=cfg_fused)

    # 共享权重：GPTLMHead.weight 是 [V, H]，bias 是 [V]
    w = paddle.randn([vocab_size, hidden_size], dtype="float32")
    b = paddle.randn([vocab_size], dtype="float32")
    lm_head_bsl.weight.set_value(w.clone())
    lm_head_bsl.bias.set_value(b.clone())
    lm_head_fused.weight.set_value(w.clone())
    lm_head_fused.bias.set_value(b.clone())

    return (lm_head_bsl, loss_bsl), (lm_head_fused, loss_fused)


# ---------------------------------------------------------------------------
# 构造输入
# ---------------------------------------------------------------------------
def _make_inputs(
    batch, seq, hidden_size, vocab_size, ignore_ratio=0.0, seed=42
):
    """生成随机 hidden_states [B, S, H] 和 labels [B, S]。"""
    paddle.seed(seed)
    hidden = paddle.randn([batch, seq, hidden_size], dtype="float32")
    labels = paddle.randint(0, vocab_size, [batch, seq], dtype="int64")

    if ignore_ratio > 0:
        n_ignore = int(batch * seq * ignore_ratio)
        flat = labels.flatten()
        ignore_pos = paddle.randint(0, batch * seq, [n_ignore], dtype="int64")
        flat_np = flat.numpy()
        flat_np[ignore_pos.numpy()] = -100
        labels = paddle.to_tensor(flat_np.reshape([batch, seq]))

    return hidden, labels


# ---------------------------------------------------------------------------
# 前向 + 反向
# ---------------------------------------------------------------------------
def _forward_backward(lm_head, lang_loss, hidden, labels):
    """执行 GPTLMHead + LanguageLoss 前向反向。

    返回 (loss, hidden.grad, weight.grad, bias.grad)。
    """
    hidden = hidden.clone()
    hidden.stop_gradient = False

    # GPTLMHead.forward 接受 dict
    lm_out = lm_head({"hidden_states": hidden})

    # LanguageLoss.forward 接受 (logits_or_tuple, labels)
    loss = lang_loss(lm_out, labels)
    loss.backward()

    return loss, hidden.grad, lm_head.weight.grad, lm_head.bias.grad


# ---------------------------------------------------------------------------
# 单测
# ---------------------------------------------------------------------------
class TestFusedLinearCEAccuracy(unittest.TestCase):
    """对比 baseline 与 fused 实现的数值精度（fp32）。

    baseline: fused_linear_ce_loss_chunk=0
      GPTLMHead → logits [B,S,V] → LanguageLoss(CrossEntropyLoss) → token-wise loss

    fused: fused_linear_ce_loss_chunk>0
      GPTLMHead → (hidden, weight[V,H], bias) 元组
      → LanguageLoss → LigerFusedLinearCrossEntropyFunction
    """

    LOSS_ATOL = 1e-5
    LOSS_RTOL = 1e-5
    GRAD_ATOL = 1e-5
    GRAD_RTOL = 1e-3

    def setUp(self):
        os.environ["FLAGS_cudnn_deterministic"] = "1"
        os.environ["FLAGS_embedding_deterministic"] = "1"

    def tearDown(self):
        paddle.device.cuda.synchronize()

    def _assert_close(self, a, b, msg, atol=None, rtol=None):
        atol = atol if atol is not None else self.LOSS_ATOL
        rtol = rtol if rtol is not None else self.LOSS_RTOL
        a_np = (
            a.numpy().flatten()
            if isinstance(a, paddle.Tensor)
            else np.array([float(a)])
        )
        b_np = (
            b.numpy().flatten()
            if isinstance(b, paddle.Tensor)
            else np.array([float(b)])
        )
        np.testing.assert_allclose(
            a_np, b_np, atol=atol, rtol=rtol, err_msg=msg
        )

    # ------------------------------------------------------------------
    # 场景 1：基本 loss 对比（无 mask）
    # ------------------------------------------------------------------
    def test_forward_loss_no_mask(self):
        H, V = 64, 256
        bsl, fused = _make_pair(H, V, num_chunks=1)
        hidden, labels = _make_inputs(2, 16, H, V, ignore_ratio=0.0)

        loss_b, _, _, _ = _forward_backward(*bsl, hidden, labels)
        loss_f, _, _, _ = _forward_backward(*fused, hidden, labels)

        self._assert_close(loss_b, loss_f, "loss (no mask)")

    # ------------------------------------------------------------------
    # 场景 2：含 ignore_index 的 loss 对比
    # ------------------------------------------------------------------
    def test_forward_loss_with_mask(self):
        H, V = 64, 256
        bsl, fused = _make_pair(H, V, num_chunks=1)
        hidden, labels = _make_inputs(2, 16, H, V, ignore_ratio=0.3)

        loss_b, _, _, _ = _forward_backward(*bsl, hidden, labels)
        loss_f, _, _, _ = _forward_backward(*fused, hidden, labels)

        self._assert_close(loss_b, loss_f, "loss (with mask)")

    # ------------------------------------------------------------------
    # 场景 3：grad_input 对比
    # ------------------------------------------------------------------
    def test_backward_grad_input(self):
        H, V = 64, 256
        bsl, fused = _make_pair(H, V, num_chunks=1)
        hidden, labels = _make_inputs(2, 16, H, V, ignore_ratio=0.2)

        _, gi_b, _, _ = _forward_backward(*bsl, hidden, labels)
        _, gi_f, _, _ = _forward_backward(*fused, hidden, labels)

        self._assert_close(gi_b, gi_f, "grad_input")

    # ------------------------------------------------------------------
    # 场景 4：grad_weight 对比
    #   baseline:  weight.grad shape [V, H]（autograd 标准路径）
    #   fused:     weight.grad shape [V, H]（经 main_grad.T 累加后由 autograd 返回）
    # ------------------------------------------------------------------
    def test_backward_grad_weight(self):
        H, V = 64, 256
        bsl, fused = _make_pair(H, V, num_chunks=1)
        hidden, labels = _make_inputs(2, 16, H, V, ignore_ratio=0.2)

        _, _, gw_b, _ = _forward_backward(*bsl, hidden, labels)
        _, _, gw_f, _ = _forward_backward(*fused, hidden, labels)

        self._assert_close(
            gw_b, gw_f, "grad_weight", atol=self.GRAD_ATOL, rtol=self.GRAD_RTOL
        )

    # ------------------------------------------------------------------
    # 场景 5：grad_bias 对比
    # ------------------------------------------------------------------
    def test_backward_grad_bias(self):
        H, V = 64, 256
        bsl, fused = _make_pair(H, V, num_chunks=1)
        hidden, labels = _make_inputs(2, 16, H, V, ignore_ratio=0.2)

        _, _, _, gb_b = _forward_backward(*bsl, hidden, labels)
        _, _, _, gb_f = _forward_backward(*fused, hidden, labels)

        self._assert_close(
            gb_b, gb_f, "grad_bias", atol=self.GRAD_ATOL, rtol=self.GRAD_RTOL
        )

    # ------------------------------------------------------------------
    # 场景 6：高 ignore 比例
    # ------------------------------------------------------------------
    def test_high_ignore_ratio(self):
        H, V = 64, 256
        bsl, fused = _make_pair(H, V, num_chunks=1)
        hidden, labels = _make_inputs(2, 32, H, V, ignore_ratio=0.7)

        loss_b, gi_b, gw_b, _ = _forward_backward(*bsl, hidden, labels)
        loss_f, gi_f, gw_f, _ = _forward_backward(*fused, hidden, labels)

        self._assert_close(loss_b, loss_f, "loss (high ignore ratio)")
        self._assert_close(gi_b, gi_f, "grad_input (high ignore ratio)")
        self._assert_close(
            gw_b,
            gw_f,
            "grad_weight (high ignore ratio)",
            atol=self.GRAD_ATOL,
            rtol=self.GRAD_RTOL,
        )

    # ------------------------------------------------------------------
    # 场景 7：全 ignore（loss/grad 均为 0）
    # ------------------------------------------------------------------
    def test_all_ignore(self):
        H, V = 64, 256
        bsl, fused = _make_pair(H, V, num_chunks=1)
        B, S = 2, 16
        hidden = paddle.randn([B, S, H], dtype="float32")
        labels = paddle.full([B, S], -100, dtype="int64")

        loss_b, _, _, _ = _forward_backward(*bsl, hidden, labels)
        loss_f, _, _, _ = _forward_backward(*fused, hidden, labels)

        # 全 ignore 时 lossmask.sum()=0，两边均返回 0.0 * mean(loss)
        self.assertAlmostEqual(
            float(loss_b),
            0.0,
            places=6,
            msg="baseline loss should be 0 for all-ignore",
        )
        self.assertAlmostEqual(
            float(loss_f),
            0.0,
            places=6,
            msg="fused loss should be 0 for all-ignore",
        )

    # ------------------------------------------------------------------
    # 场景 8：多 chunk
    # ------------------------------------------------------------------
    def test_multi_chunk(self):
        H, V = 64, 256
        bsl, fused = _make_pair(H, V, num_chunks=4)
        hidden, labels = _make_inputs(1, 64, H, V, ignore_ratio=0.2)

        loss_b, gi_b, gw_b, _ = _forward_backward(*bsl, hidden, labels)
        loss_f, gi_f, gw_f, _ = _forward_backward(*fused, hidden, labels)

        self._assert_close(loss_b, loss_f, "loss (multi-chunk)")
        self._assert_close(gi_b, gi_f, "grad_input (multi-chunk)")
        self._assert_close(
            gw_b,
            gw_f,
            "grad_weight (multi-chunk)",
            atol=self.GRAD_ATOL,
            rtol=self.GRAD_RTOL,
        )

    # ------------------------------------------------------------------
    # 场景 9：较大 vocab_size
    # ------------------------------------------------------------------
    def test_large_vocab(self):
        H, V = 128, 2048
        bsl, fused = _make_pair(H, V, num_chunks=4)
        hidden, labels = _make_inputs(2, 32, H, V, ignore_ratio=0.15)

        loss_b, gi_b, gw_b, _ = _forward_backward(*bsl, hidden, labels)
        loss_f, gi_f, gw_f, _ = _forward_backward(*fused, hidden, labels)

        self._assert_close(loss_b, loss_f, "loss (large vocab)")
        self._assert_close(gi_b, gi_f, "grad_input (large vocab)")
        self._assert_close(
            gw_b,
            gw_f,
            "grad_weight (large vocab)",
            atol=self.GRAD_ATOL,
            rtol=self.GRAD_RTOL,
        )


if __name__ == "__main__":
    unittest.main()
