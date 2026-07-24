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

"""Online TileLang HySparse training-loop validation (task #2).

Exercises the *production online* HySparse path with both branches routed
through the TileLang backend (``hy_sparse_full_attn_use_tilelang=True`` +
``hy_sparse_block_sparse_use_tilelang=True``)
end-to-end through the real two-layer network wiring, at the online model's
per-head dimensions (Dk = qk_rope+qk_nope = 256, Dv = kv_lora_rank = 448,
H = 64 heads), with the learnable SWA attention-sink bias turned on
(``add_swa_attention_sink_bias=True``), and single-card parallelism
(TP=CP=PP=1).

The path runs across two ``HySparseTransformerLayer`` layers:

  * a *full* layer scores all key blocks and emits ``shared_key`` +
    ``shared_block_indices`` (the shared compressed-KV latent);
  * a *SWA* layer consumes those, running its sliding-window main path plus a
    TileLang block-sparse gather branch, both with their own learnable sink.

What this closes (vs the existing oracle-vs-production integration test, which
pins ``full_recompute=True`` on both stacks, uses H=4 / kv_lora_rank=512, and
sinkless softmax):

  1. gradients on the online path exist, are finite, and are non-zero for
     - the network input hidden state,
     - the full layer's ``kv_a_proj_with_mqa.weight`` (the *entire* shared_key
       latent traces to it -> proves the shared-KV gradient bridge is live),
     - both learnable sink logits (``swa_attn_sink`` / ``sparse_attn_sink``) on
       the SWA layer (proves the sink is actually trainable, not a dead param);
  2. full_recompute ON vs OFF agree to bf16 tolerance on both the SWA output and
     every audited gradient (recompute correctness for the shared-KV path);
  3. one real optimizer step runs, stays finite, and actually moves the sink
     logits off their zero init.

Requires an SM 10.x (Blackwell) device for the TileLang kernels; skips
otherwise. Only the TileLang backend is needed (no FA4/DSA), so this runs
independently of the production-vs-oracle cross-check.
"""

import os
import unittest

os.environ["FLAGS_cudnn_deterministic"] = "True"

import numpy as np
import paddle
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddleformers.fleet.fusions.fused_bias_dropout import get_bias_dropout_add
from paddleformers.fleet.parallel_state import get_context_parallel_world_size
from paddleformers.fleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddleformers.fleet.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
)
from paddleformers.fleet.transformer.dot_product_attention import (
    DotProductAttention,
)
from paddleformers.fleet.transformer.enums import AttnMaskType
from paddleformers.fleet.transformer.multi_latent_attention import (
    MLASelfAttentionSublayersSpec,
    MQASelfAttention,
)
from paddleformers.fleet.transformer.paddle_norm import WrappedPaddleNorm
from paddleformers.fleet.transformer.transformer_config import TransformerConfig
from paddleformers.fleet.transformer.transformer_layer import (
    HySparseTransformerLayer,
    TransformerLayerSublayersSpec,
)


def _tilelang_backend_or_skip(testcase):
    """Skip unless the TileLang HySparse kernels can run (SM 10.x)."""
    if not paddle.device.is_compiled_with_cuda():
        testcase.skipTest("CUDA build of Paddle is required")
    if paddle.device.cuda.device_count() == 0:
        testcase.skipTest("No CUDA device available")
    major = paddle.device.cuda.get_device_capability()[0]
    if major != 10:
        testcase.skipTest(
            f"HySparse TileLang kernels require SM 10.x; got SM {major}.x"
        )
    try:
        from paddleformers.fleet.tilelang_ops.hysparse import (  # noqa: F401
            sliding_window_mqa_attention,
        )
        from paddleformers.fleet.tilelang_ops.hysparse.block_sparse_mqa_tl import (  # noqa: F401
            block_sparse_mqa_attention_tl,
        )
    except (ImportError, RuntimeError) as exc:
        testcase.skipTest(f"HySparse TileLang backend import failed: {exc}")


def _finite(t):
    return bool(paddle.isfinite(t).all())


def _l2(t):
    return float(t.astype("float32").norm().item())


class TestHySparseOnlineTileLangTrainStep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Kept modest for a single card; the *per-head* dims below match the
        # online model, which is what the task pins (Dk=256, Dv=448, H=64).
        cls.batch_size = 1
        cls.seq_len = 2048

        # Online HySparse MQA wiring @ production per-head dims.
        #   Dk (query/key head dim) = qk_rope_head_dim + qk_nope_head_dim
        #                           = 64 + 192 = 256
        #   Dv (shared value latent) = kv_lora_rank = 448
        #   H  (attention heads)     = 64
        # learnable SWA attention-sink bias ON; TileLang backend ON.
        cls.base_config = TransformerConfig(
            hidden_size=2048,
            head_dim=192,
            num_attention_heads=64,
            num_key_value_heads=4,
            gated_attention=True,
            gated_attn_use_q_lora=True,
            q_lora_rank=1024,
            qk_rope_head_dim=64,
            qk_nope_head_dim=192,
            v_head_dim=256,
            kv_lora_rank=448,
            rope_theta=640000,
            use_qk_norm=True,
            multi_latent_attention=True,
            rope_type="rope",
            add_swa_attention_sink_bias=True,
            sliding_window=[128, 128],
            swa_head_dim=192,
            swa_v_head_dim=256,
            swa_num_attention_heads=64,
            swa_num_key_value_heads=4,
            window_attn_skip_freq=2,
            enable_hy_sparse_attention=True,
            hy_sparse_full_attn_use_tilelang=True,
            hy_sparse_block_sparse_use_tilelang=True,
            hy_sparse_block_size=64,
            hy_sparse_topk=16,
        )

        cls.sublayer_spec = MLASelfAttentionSublayersSpec(
            core_attention=DotProductAttention,
            o_proj=RowParallelLinear,
            gate_proj=ColumnParallelLinear,
            q_a_proj=ColumnParallelLinear,
            q_b_proj=ColumnParallelLinear,
            kv_a_proj_with_mqa=ColumnParallelLinear,
            kv_b_proj=ColumnParallelLinear,
            q_a_layernorm=WrappedPaddleNorm,
            kv_a_layernorm=WrappedPaddleNorm,
        )

    # ---- builders -----------------------------------------------------

    def _build_stack(self, config, full_recompute):
        """Build a (full, swa) HySparseTransformerLayer pair."""
        layer_spec = TransformerLayerSublayersSpec(
            self_attn=LayerSpec(
                layer=MQASelfAttention,
                sublayers_spec=self.sublayer_spec,
            ),
            self_attn_bda=get_bias_dropout_add,
        )
        full_layer = HySparseTransformerLayer(
            config, layer_spec, layer_number=0
        )
        full_layer.self_attn.attn_mask_type = AttnMaskType.causal
        full_layer = paddle.amp.decorate(
            full_layer, level="O2", dtype="bfloat16"
        )
        full_layer.full_recompute = full_recompute

        swa_layer = HySparseTransformerLayer(config, layer_spec, layer_number=1)
        swa_layer.self_attn.attn_mask_type = AttnMaskType.causal
        swa_layer = paddle.amp.decorate(swa_layer, level="O2", dtype="bfloat16")
        swa_layer.full_recompute = full_recompute
        return full_layer, swa_layer

    def _inputs(self):
        hidden_states = paddle.randn(
            [self.batch_size, self.seq_len, self.base_config.hidden_size],
            dtype="bfloat16",
        )
        startend = paddle.full(
            [self.batch_size, 1, self.seq_len, 1], self.seq_len, dtype="int32"
        )
        ograd = (
            paddle.randn(
                [self.batch_size, self.seq_len, self.base_config.hidden_size],
                dtype="bfloat16",
            )
            * 1e-2
        )
        return hidden_states, startend, ograd

    def _run_stack(self, full_layer, swa_layer, hidden_states, startend, ograd):
        """full -> SWA forward, backward; returns (swa_out, input_grad)."""
        hs = hidden_states.detach()
        hs.stop_gradient = False
        out_dict = full_layer(
            {
                "hidden_states": hs,
                "attn_mask_startend_row_indices": startend,
            }
        )
        out_dict = swa_layer(out_dict)
        swa_out = out_dict["hidden_states"]
        swa_out.backward(ograd)
        return swa_out.detach(), hs.grad

    # ---- tests --------------------------------------------------------

    def test_online_dims_and_single_card(self):
        """Confirm the wiring matches the online per-head dims and runs on a
        single card (TP=CP=PP=1)."""
        _tilelang_backend_or_skip(self)
        # single process -> no context parallel; MQA forward hard-rejects
        # TP>1 / CP>1, so a successful run also proves TP=CP=1.
        self.assertEqual(get_context_parallel_world_size(), 1)

        cfg = self.base_config
        # Dk = qk_rope + qk_nope = 256; Dv = kv_lora_rank = 448; H = 64.
        self.assertEqual(cfg.qk_rope_head_dim + cfg.qk_nope_head_dim, 256)
        self.assertEqual(cfg.kv_lora_rank, 448)
        self.assertEqual(cfg.num_attention_heads, 64)
        self.assertTrue(cfg.enable_hy_sparse_attention)
        self.assertTrue(cfg.hy_sparse_full_attn_use_tilelang)
        self.assertTrue(cfg.hy_sparse_block_sparse_use_tilelang)
        self.assertTrue(cfg.add_swa_attention_sink_bias)

        paddle.seed(2026)
        model_parallel_cuda_manual_seed(2026)
        full_layer, swa_layer = self._build_stack(cfg, full_recompute=True)
        # full layer scores blocks (is_mqa False -> MLA super().forward);
        # SWA layer runs the MQA sliding-window + block-sparse branches.
        self.assertFalse(full_layer.self_attn.is_mqa)
        self.assertTrue(swa_layer.self_attn.is_mqa)
        self.assertTrue(swa_layer.self_attn.is_swa)
        # learnable sink logits live only on the SWA (MQA) layer, shape [H].
        self.assertIsNotNone(swa_layer.self_attn.swa_attn_sink)
        self.assertIsNotNone(swa_layer.self_attn.sparse_attn_sink)
        self.assertEqual(list(swa_layer.self_attn.swa_attn_sink.shape), [64])
        self.assertEqual(list(swa_layer.self_attn.sparse_attn_sink.shape), [64])

    def test_shared_kv_recompute_grads_and_sink(self):
        """full_recompute ON (the online setting): shared-KV bridge + sink
        gradients exist / finite / non-zero."""
        _tilelang_backend_or_skip(self)
        paddle.seed(2026)
        model_parallel_cuda_manual_seed(2026)

        full_layer, swa_layer = self._build_stack(
            self.base_config, full_recompute=True
        )
        hidden_states, startend, ograd = self._inputs()
        swa_out, igrad = self._run_stack(
            full_layer, swa_layer, hidden_states, startend, ograd
        )

        # forward finite
        self.assertTrue(_finite(swa_out))

        # (1) network input gradient
        self.assertIsNotNone(igrad)
        self.assertTrue(_finite(igrad))
        self.assertGreater(_l2(igrad), 0.0)

        # (2) shared-KV bridge: full layer kv_a_proj_with_mqa.weight grad.
        #     The whole shared_key latent traces to this projection, so a live
        #     grad here proves the full->SWA shared-KV gradient path is wired.
        kv_a_w = full_layer.self_attn.kv_a_proj_with_mqa.weight
        self.assertIsNotNone(kv_a_w.grad)
        self.assertTrue(_finite(kv_a_w.grad))
        self.assertGreater(_l2(kv_a_w.grad), 0.0)

        # (3) learnable attention sinks (both SWA branches) are trainable.
        for name in ("swa_attn_sink", "sparse_attn_sink"):
            sink = getattr(swa_layer.self_attn, name)
            self.assertIsNotNone(sink.grad, f"{name}.grad is None")
            self.assertTrue(_finite(sink.grad), f"{name}.grad not finite")
            self.assertGreater(
                _l2(sink.grad), 0.0, f"{name}.grad is all-zero (dead sink)"
            )

    def test_recompute_on_off_consistency(self):
        """full_recompute ON vs OFF agree (bf16 tol) on SWA output and every
        audited gradient. The OFF path needs ``has_recovered`` monkeypatched
        (see bug note in the task report)."""
        _tilelang_backend_or_skip(self)
        paddle.seed(7)
        model_parallel_cuda_manual_seed(7)

        cfg = self.base_config
        on_full, on_swa = self._build_stack(cfg, full_recompute=True)
        off_full, off_swa = self._build_stack(cfg, full_recompute=False)
        # identical weights across the two stacks.
        off_full.set_state_dict(on_full.state_dict())
        off_swa.set_state_dict(on_swa.state_dict())

        hidden_states, startend, ograd = self._inputs()

        on_out, on_igrad = self._run_stack(
            on_full, on_swa, hidden_states, startend, ograd
        )

        off_out, off_igrad = self._run_stack(
            off_full, off_swa, hidden_states, startend, ograd
        )

        np.testing.assert_allclose(
            on_out.astype("float32").numpy(),
            off_out.astype("float32").numpy(),
            atol=8e-2,
            rtol=8e-2,
        )
        np.testing.assert_allclose(
            on_igrad.astype("float32").numpy(),
            off_igrad.astype("float32").numpy(),
            atol=8e-2,
            rtol=8e-2,
        )
        # sink + shared-KV grads also agree across recompute on/off.
        for layer_on, layer_off, attr in (
            (on_full, off_full, "kv_a_proj_with_mqa"),
        ):
            g_on = layer_on.self_attn.kv_a_proj_with_mqa.weight.grad
            g_off = layer_off.self_attn.kv_a_proj_with_mqa.weight.grad
            np.testing.assert_allclose(
                g_on.astype("float32").numpy(),
                g_off.astype("float32").numpy(),
                atol=8e-2,
                rtol=8e-2,
            )
        for name in ("swa_attn_sink", "sparse_attn_sink"):
            g_on = getattr(on_swa.self_attn, name).grad
            g_off = getattr(off_swa.self_attn, name).grad
            np.testing.assert_allclose(
                g_on.astype("float32").numpy(),
                g_off.astype("float32").numpy(),
                atol=8e-2,
                rtol=8e-2,
            )

    def test_one_optimizer_step_trains_sink(self):
        """A single AdamW step runs, stays finite, and moves the zero-init sink
        logits (proves the learnable sink is actually optimized)."""
        _tilelang_backend_or_skip(self)
        paddle.seed(11)
        model_parallel_cuda_manual_seed(11)

        full_layer, swa_layer = self._build_stack(
            self.base_config, full_recompute=True
        )
        params = full_layer.parameters() + swa_layer.parameters()
        opt = paddle.optimizer.AdamW(
            learning_rate=1e-2, parameters=params, multi_precision=True
        )

        hidden_states, startend, ograd = self._inputs()
        hs = hidden_states.detach()
        hs.stop_gradient = False
        out_dict = swa_layer(
            full_layer(
                {
                    "hidden_states": hs,
                    "attn_mask_startend_row_indices": startend,
                }
            )
        )
        loss = out_dict["hidden_states"].astype("float32").pow(2).mean()
        self.assertTrue(_finite(loss))

        swa_sink_before = (
            swa_layer.self_attn.swa_attn_sink.astype("float32").numpy().copy()
        )
        sparse_sink_before = (
            swa_layer.self_attn.sparse_attn_sink.astype("float32")
            .numpy()
            .copy()
        )

        loss.backward()
        opt.step()
        opt.clear_grad()

        # every parameter stays finite after the step.
        for p in params:
            self.assertTrue(_finite(p), f"param {p.name} went non-finite")

        swa_sink_after = swa_layer.self_attn.swa_attn_sink.astype(
            "float32"
        ).numpy()
        sparse_sink_after = swa_layer.self_attn.sparse_attn_sink.astype(
            "float32"
        ).numpy()

        # sinks started at 0.0; after one step with lr=1e-2 they must move.
        self.assertGreater(
            float(np.abs(swa_sink_after - swa_sink_before).max()),
            0.0,
            "swa_attn_sink did not update",
        )
        self.assertGreater(
            float(np.abs(sparse_sink_after - sparse_sink_before).max()),
            0.0,
            "sparse_attn_sink did not update",
        )


if __name__ == "__main__":
    unittest.main()
