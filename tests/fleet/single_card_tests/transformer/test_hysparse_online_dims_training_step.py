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

"""End-to-end training-step check for the online HySparse TileLang wiring.

This closes the loop for the *production online* configuration
(``conf/online/ernielite_layer43_pretrain_exp2.4.yaml`` +
``model_config_separated/conf/fleet_align/ernielite_layer43_0713_8k_bzz3``):

  * backend  = TileLang oracle for both branches in the training/precision tests,
    plus production FA4 + DSA in the packed runtime-routing test;
  * recompute = full (``recompute_granularity=full``, ``recompute_num_layers=1``)
  * learnable attention sink = ON (``add_swa_attention_sink_bias=True``)
  * parallelism = TP=CP=PP=1 (HySparse hard-rejects TP/CP>1; online sets all 1)

and reproduces the online *key attention dimensions* exactly:

  * H  = num_attention_heads          = 64
  * Dk = qk_rope(64) + qk_nope(192)   = 256   (per-head QK projection dim)
  * Dv = kv_lora_rank                 = 448   (absorbed-MLA MQA value/latent dim)

Only the affordability knobs (hidden_size / seq_len / batch) are shrunk; the
key dims above are the online values. The kv_lora_rank=448 / Dk_mqa=512 shape is
specifically *not* a power-of-two, so this also exercises the TileLang MQA
kernels on the exact non-aligned latent widths the online model uses.

The test runs a full ``full -> SWA`` forward + backward + one AdamW optimizer
step and asserts:

  1. the full layer emits ``shared_key`` and the SWA output is finite;
  2. every probed gradient -- including both learnable sinks -- is finite and
     non-zero (the shared-KV edge and both sink softmaxes are all live);
  3. the optimizer step actually moves the parameters and leaves them finite;
  4. (precision) the full-layer ``kv_a_proj`` and input-hidden gradients agree
     between recompute-ON and recompute-OFF to bf16 tolerance.

Requires an SM 10.x (Blackwell) device with the FA4 FlashMask CUTE backend and
the cuDNN DSA backend available; skips otherwise.
"""

import os
import unittest

os.environ["FLAGS_cudnn_deterministic"] = "True"

import dataclasses

import numpy as np
import paddle
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddleformers.fleet.fusions.fused_bias_dropout import get_bias_dropout_add
from paddleformers.fleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddleformers.fleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddleformers.fleet.transformer.dot_product_attention import DotProductAttention
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

# bf16 relative-L2 tolerance for the recompute-on vs recompute-off comparison.
_BF16_REL_L2_TOL = 0.1
_BACKEND_MATRIX = {
    "TT": (True, True),
    "TF": (True, False),
    "FT": (False, True),
    "FF": (False, False),
}


def _rel_l2(got, ref):
    diff = float(np.linalg.norm((got - ref).ravel()))
    denom = float(np.linalg.norm(ref.ravel()))
    return diff / denom if denom > 1e-12 else diff


def _hysparse_backend_or_skip(testcase):
    """Skip unless the TileLang and production FA4/DSA backends can run."""
    if not paddle.device.is_compiled_with_cuda():
        testcase.skipTest("CUDA build of Paddle is required")
    if paddle.device.cuda.device_count() == 0:
        testcase.skipTest("No CUDA device available")
    major = paddle.device.cuda.get_device_capability()[0]
    if major != 10:
        testcase.skipTest(f"HySparse kernels require SM 10.x; got SM {major}.x")
    try:
        import paddlefleet_ops

        from paddleformers.fleet.cudnn_ops import is_dsa_available
        from paddleformers.fleet.tilelang_ops.hysparse import (  # noqa: F401
            sliding_window_mqa_attention,
        )
        from paddleformers.fleet.tilelang_ops.hysparse.block_sparse_mqa_tl import (  # noqa: F401
            block_sparse_mqa_attention_tl,
        )

        if not paddlefleet_ops.is_flash_mask_available():
            testcase.skipTest("FlashMask (FA4) backend not available")
        if not is_dsa_available():
            testcase.skipTest("cuDNN DSA backend not available")
    except (ImportError, RuntimeError) as exc:
        testcase.skipTest(f"HySparse backend import failed: {exc}")


def _ft_backend_or_skip(testcase):
    """Skip unless the FA4 scorer + TileLang sparse backends can run.

    FT (full=FA4, sparse=TileLang) does not touch the cuDNN DSA backend
    (multi_latent_attention only imports DSA on the ``use_tl=False`` sparse
    branch), so -- unlike :func:`_hysparse_backend_or_skip` -- this helper does
    *not* require ``is_dsa_available``. That lets the FT-only tests run on nodes
    where FA4 + TileLang are present but DSA is not.
    """
    if not paddle.device.is_compiled_with_cuda():
        testcase.skipTest("CUDA build of Paddle is required")
    if paddle.device.cuda.device_count() == 0:
        testcase.skipTest("No CUDA device available")
    major = paddle.device.cuda.get_device_capability()[0]
    if major != 10:
        testcase.skipTest(f"HySparse kernels require SM 10.x; got SM {major}.x")
    try:
        import paddlefleet_ops

        from paddleformers.fleet.tilelang_ops.hysparse import (  # noqa: F401
            sliding_window_mqa_attention,
        )
        from paddleformers.fleet.tilelang_ops.hysparse.block_sparse_mqa_tl import (  # noqa: F401
            block_sparse_mqa_attention_tl,
        )

        if not paddlefleet_ops.is_flash_mask_available():
            testcase.skipTest("FlashMask (FA4) backend not available")
    except (ImportError, RuntimeError) as exc:
        testcase.skipTest(f"HySparse FA4/TileLang backend import failed: {exc}")


def _shape(tensor):
    return list(tensor.shape) if tensor is not None else None


class TestHySparseOnlineDimsTrainingStep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Affordability knobs (NOT online values): a two-layer stack at online
        # attention dims is memory-heavy, so shrink batch/seq/hidden only.
        cls.batch_size = 1
        cls.seq_len = 2048
        cls.hidden_size = 2048

        # Online HySparse MQA wiring, TileLang backend, learnable sink ON.
        # Key dims match production: H=64, Dk=qk_rope(64)+qk_nope(192)=256,
        # Dv=kv_lora_rank=448.
        cls.base_config = TransformerConfig(
            hidden_size=cls.hidden_size,
            head_dim=128,
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
            window_attn_skip_freq=2,
            enable_hy_sparse_attention=True,
            hy_sparse_full_attn_use_tilelang=True,
            hy_sparse_block_sparse_use_tilelang=True,
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

    def _build_stack(self, config, full_recompute):
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

    def _make_inputs(self, doc_lengths=None):
        hidden_states = paddle.randn(
            [self.batch_size, self.seq_len, self.hidden_size], dtype="bfloat16"
        )
        if doc_lengths is None:
            startend = paddle.full(
                [self.batch_size, 1, self.seq_len, 1],
                self.seq_len,
                dtype="int32",
            )
        else:
            self.assertEqual(self.batch_size, 1)
            self.assertEqual(sum(doc_lengths), self.seq_len)
            doc_ends = paddle.zeros([self.seq_len], dtype="int32")
            offset = 0
            for length in doc_lengths:
                doc_ends[offset : offset + length] = offset + length
                offset += length
            startend = doc_ends.reshape([1, 1, self.seq_len, 1])
        ograd = (
            paddle.randn(
                [self.batch_size, self.seq_len, self.hidden_size],
                dtype="bfloat16",
            )
            * 1e-2
        )
        return hidden_states, startend, ograd

    def _forward_backward(self, full_layer, swa_layer, inputs):
        """Full -> SWA forward + backward. Returns (swa_out, input_grad, out_dict)."""
        full_layer.train()
        swa_layer.train()
        hidden_states, startend, ograd = inputs
        hs = hidden_states.detach()
        hs.stop_gradient = False
        out_dict = full_layer(
            {
                "hidden_states": hs,
                "attn_mask_startend_row_indices": startend,
            }
        )
        self.assertIn("shared_key", out_dict)
        self.assertIsNotNone(out_dict["shared_key"])
        out_dict = swa_layer(out_dict)
        swa_out = out_dict["hidden_states"]
        swa_out.backward(ograd)
        return swa_out.detach(), hs.grad

    def _assert_grad_ok(self, name, param):
        self.assertIsNotNone(
            param.grad, f"[{name}] grad is None -- gradient edge is broken"
        )
        g = param.grad.astype("float32")
        finite = bool(paddle.isfinite(g).all())
        norm = float(paddle.linalg.norm(g).item())
        print(
            f"[grad] {name:40s} shape={list(param.shape)} "
            f"finite={finite} norm={norm:.6e}"
        )
        self.assertTrue(finite, f"[{name}] gradient has NaN/Inf")
        self.assertGreater(norm, 0.0, f"[{name}] gradient is all-zero")

    def test_online_tilelang_training_step(self):
        """Full end-to-end: forward + backward + one AdamW step at online dims."""
        _hysparse_backend_or_skip(self)
        paddle.seed(2026)
        model_parallel_cuda_manual_seed(2026)

        full_layer, swa_layer = self._build_stack(
            self.base_config, full_recompute=True
        )
        inputs = self._make_inputs()
        swa_out, input_grad = self._forward_backward(
            full_layer, swa_layer, inputs
        )

        # (1) forward output finite.
        self.assertTrue(
            bool(paddle.isfinite(swa_out.astype("float32")).all()),
            "SWA output has NaN/Inf",
        )
        self.assertTrue(
            bool(paddle.isfinite(input_grad.astype("float32")).all()),
            "input-hidden gradient has NaN/Inf",
        )

        # (2) probed gradients finite + non-zero. The two sinks and the full
        # layer's kv_a_proj (shared_key source) are the load-bearing edges.
        probes = {
            "full.kv_a_proj_with_mqa": full_layer.self_attn.kv_a_proj_with_mqa.weight,
            "full.q_b_proj": full_layer.self_attn.q_b_proj.weight,
            "full.kv_b_proj": full_layer.self_attn.kv_b_proj.weight,
            "full.o_proj": full_layer.self_attn.o_proj.weight,
            "swa.kv_a_proj_with_mqa": swa_layer.self_attn.kv_a_proj_with_mqa.weight,
            "swa.o_proj": swa_layer.self_attn.o_proj.weight,
            "swa.swa_attn_sink": swa_layer.self_attn.swa_attn_sink,
            "swa.sparse_attn_sink": swa_layer.self_attn.sparse_attn_sink,
        }
        for name, param in probes.items():
            self.assertIsNotNone(
                param, f"[{name}] parameter is None (not created)"
            )
            self._assert_grad_ok(name, param)

        # (3) one optimizer step actually moves + keeps params finite.
        params = list(full_layer.parameters()) + list(swa_layer.parameters())
        before = {id(p): p.detach().astype("float32").clone() for p in params}
        opt = paddle.optimizer.AdamW(
            learning_rate=1e-3, parameters=params, multi_precision=True
        )
        opt.step()
        opt.clear_grad()

        moved = 0
        for p in params:
            after = p.detach().astype("float32")
            self.assertTrue(
                bool(paddle.isfinite(after).all()),
                "a parameter became NaN/Inf after the optimizer step",
            )
            if float(paddle.linalg.norm(after - before[id(p)]).item()) > 0.0:
                moved += 1
        print(f"[opt] {moved}/{len(params)} parameters moved after AdamW step")
        self.assertGreater(
            moved, 0, "optimizer step did not update any parameter"
        )

    def test_online_runtime_op_shapes(self):
        """Record + assert the runtime q/shared_key/output shapes the online
        HySparse ops actually see, and confirm H / Dk / Dv.

        Full scorer (full layer, decompressed MHA)  : q,k=[B,S,H,256] Dk=256,
            v=[B,S,H,256] Dv=256, out=[B,S,H,256].
        SWA windowed main path (absorbed MQA)       : q=[B,S,H,512] Dk=512,
            shared_k=[B,S,512], shared_v=[B,S,448] Dv=448, out=[B,S,H,448].
        SWA block-sparse gather (absorbed MQA)       : q=[B,S,H,512] Dk=512,
            shared_key=[B,S,512], out=[B,S,H*448] (pre-flattened).

        Runs a full forward + backward with a random dO and one AdamW step, so
        the shapes are captured on the exact online training path, not a mock.
        """
        _hysparse_backend_or_skip(self)
        paddle.seed(2026)
        model_parallel_cuda_manual_seed(2026)

        import paddleformers.fleet.tilelang_ops.hysparse as hy_pkg
        import paddleformers.fleet.tilelang_ops.hysparse.block_score_mha as bsm_mod
        import paddleformers.fleet.tilelang_ops.hysparse.block_sparse_mqa_tl as bsp_mod

        captured = {}

        orig_full = bsm_mod.block_score_mha_attn_fwd

        def spy_full(q, k, v, *a, **kw):
            out = orig_full(q, k, v, *a, **kw)
            captured["full_scorer"] = {
                "q": _shape(q),
                "k": _shape(k),
                "v": _shape(v),
                "out": _shape(out[0]),
            }
            return out

        orig_swa = hy_pkg.sliding_window_mqa_attention

        def spy_swa(q, k, v, *a, **kw):
            out = orig_swa(q, k, v, *a, **kw)
            captured["swa_windowed"] = {
                "q": _shape(q),
                "shared_k": _shape(k),
                "shared_v": _shape(v),
                "out": _shape(out[0]),
            }
            return out

        orig_bsp = bsp_mod.block_sparse_mqa_attention_tl

        def spy_bsp(q, shared_key_sq, *a, **kw):
            out = orig_bsp(q, shared_key_sq, *a, **kw)
            captured["swa_block_sparse"] = {
                "q": _shape(q),
                "shared_key": _shape(shared_key_sq),
                "out": _shape(out[0]),
            }
            return out

        bsm_mod.block_score_mha_attn_fwd = spy_full
        hy_pkg.sliding_window_mqa_attention = spy_swa
        bsp_mod.block_sparse_mqa_attention_tl = spy_bsp
        try:
            full_layer, swa_layer = self._build_stack(
                self.base_config, full_recompute=True
            )
            inputs = self._make_inputs()
            swa_out, _ = self._forward_backward(full_layer, swa_layer, inputs)
            params = list(full_layer.parameters()) + list(
                swa_layer.parameters()
            )
            opt = paddle.optimizer.AdamW(
                learning_rate=1e-3, parameters=params, multi_precision=True
            )
            opt.step()
            opt.clear_grad()
        finally:
            bsm_mod.block_score_mha_attn_fwd = orig_full
            hy_pkg.sliding_window_mqa_attention = orig_swa
            bsp_mod.block_sparse_mqa_attention_tl = orig_bsp

        for name, shp in captured.items():
            print(f"[runtime-shape] {name}: {shp}")

        B, S = self.batch_size, self.seq_len
        H = self.base_config.num_attention_heads  # 64
        Dk_full = (
            self.base_config.qk_nope_head_dim
            + self.base_config.qk_rope_head_dim
        )  # 256
        Dv_full = self.base_config.v_head_dim  # 256
        Dk_mqa = (
            self.base_config.kv_lora_rank + self.base_config.qk_rope_head_dim
        )  # 512
        Dv_mqa = self.base_config.kv_lora_rank  # 448

        self.assertEqual(H, 64)
        self.assertEqual(Dk_mqa, 512)
        self.assertEqual(Dv_mqa, 448)

        # All three op boundaries must have been exercised.
        for key in ("full_scorer", "swa_windowed", "swa_block_sparse"):
            self.assertIn(key, captured, f"{key} op was never called")

        fs = captured["full_scorer"]
        self.assertEqual(fs["q"], [B, S, H, Dk_full])
        self.assertEqual(fs["k"], [B, S, H, Dk_full])
        self.assertEqual(fs["v"], [B, S, H, Dv_full])
        self.assertEqual(fs["out"], [B, S, H, Dv_full])

        sw = captured["swa_windowed"]
        self.assertEqual(sw["q"], [B, S, H, Dk_mqa])
        self.assertEqual(sw["shared_k"], [B, S, Dk_mqa])
        self.assertEqual(sw["shared_v"], [B, S, Dv_mqa])
        self.assertEqual(sw["out"], [B, S, H, Dv_mqa])

        bs = captured["swa_block_sparse"]
        self.assertEqual(bs["q"], [B, S, H, Dk_mqa])
        self.assertEqual(bs["shared_key"], [B, S, Dk_mqa])
        # block_sparse_mqa_attention_tl returns a pre-flattened [B,S,H*Dv].
        self.assertEqual(bs["out"], [B, S, H * Dv_mqa])

    def test_online_ff_runtime_op_shapes_packed(self):
        """Prove packed online-shape execution routes through FA4 + DSA."""
        _hysparse_backend_or_skip(self)
        paddle.seed(2026)
        model_parallel_cuda_manual_seed(2026)

        import paddleformers.fleet.cudnn_ops as cudnn_pkg
        import paddleformers.fleet.tilelang_ops.hysparse as hy_pkg
        import paddleformers.fleet.tilelang_ops.hysparse.block_score_mha as bsm_mod
        import paddleformers.fleet.tilelang_ops.hysparse.block_sparse_mqa_tl as bsp_mod

        captured = {
            "fa4_calls": 0,
            "dsa_calls": 0,
            "tl_full_calls": 0,
            "tl_sparse_calls": 0,
        }

        orig_fa4 = hy_pkg.block_score_fa4_attn_fwd
        orig_dsa = cudnn_pkg.block_sparse_mqa_attention_dsa
        orig_tl_full = bsm_mod.block_score_mha_attn_fwd
        orig_tl_sparse = bsp_mod.block_sparse_mqa_attention_tl

        def spy_fa4(q, k, v, *args, **kwargs):
            captured["fa4_calls"] += 1
            out = orig_fa4(q, k, v, *args, **kwargs)
            captured["fa4"] = {
                "q": _shape(q),
                "k": _shape(k),
                "v": _shape(v),
                "out": _shape(out[0]),
                "startend": _shape(kwargs.get("startend_row_indices")),
            }
            return out

        def spy_dsa(q, shared_key, *args, **kwargs):
            captured["dsa_calls"] += 1
            out = orig_dsa(q, shared_key, *args, **kwargs)
            captured["dsa"] = {
                "q": _shape(q),
                "shared_key": _shape(shared_key),
                "out": _shape(out[0]),
                "kv_lora_rank": kwargs.get("kv_lora_rank"),
            }
            return out

        def spy_tl_full(*args, **kwargs):
            captured["tl_full_calls"] += 1
            return orig_tl_full(*args, **kwargs)

        def spy_tl_sparse(*args, **kwargs):
            captured["tl_sparse_calls"] += 1
            return orig_tl_sparse(*args, **kwargs)

        hy_pkg.block_score_fa4_attn_fwd = spy_fa4
        cudnn_pkg.block_sparse_mqa_attention_dsa = spy_dsa
        bsm_mod.block_score_mha_attn_fwd = spy_tl_full
        bsp_mod.block_sparse_mqa_attention_tl = spy_tl_sparse
        try:
            cfg = dataclasses.replace(
                self.base_config,
                hy_sparse_full_attn_use_tilelang=False,
                hy_sparse_block_sparse_use_tilelang=False,
            )
            full_layer, swa_layer = self._build_stack(cfg, full_recompute=True)
            inputs = self._make_inputs([257, 511, 63, 1217])
            swa_out, input_grad = self._forward_backward(
                full_layer, swa_layer, inputs
            )
        finally:
            hy_pkg.block_score_fa4_attn_fwd = orig_fa4
            cudnn_pkg.block_sparse_mqa_attention_dsa = orig_dsa
            bsm_mod.block_score_mha_attn_fwd = orig_tl_full
            bsp_mod.block_sparse_mqa_attention_tl = orig_tl_sparse

        self.assertTrue(bool(paddle.isfinite(swa_out.astype("float32")).all()))
        self.assertTrue(
            bool(paddle.isfinite(input_grad.astype("float32")).all())
        )
        # Full recompute executes each backend once in the no-grad forward and
        # once again during backward recomputation.
        self.assertEqual(captured["fa4_calls"], 2)
        self.assertEqual(captured["dsa_calls"], 2)
        self.assertEqual(captured["tl_full_calls"], 0)
        self.assertEqual(captured["tl_sparse_calls"], 0)

        B, S = self.batch_size, self.seq_len
        H = self.base_config.num_attention_heads
        Dk_full = (
            self.base_config.qk_nope_head_dim
            + self.base_config.qk_rope_head_dim
        )
        Dv_full = self.base_config.v_head_dim
        Dk_mqa = (
            self.base_config.kv_lora_rank + self.base_config.qk_rope_head_dim
        )
        Dv_mqa = self.base_config.kv_lora_rank
        self.assertEqual(captured["fa4"]["q"], [B, S, H, Dk_full])
        self.assertEqual(captured["fa4"]["k"], [B, S, H, Dk_full])
        self.assertEqual(captured["fa4"]["v"], [B, S, H, Dv_full])
        self.assertEqual(captured["fa4"]["out"], [B, S, H, Dv_full])
        self.assertEqual(captured["fa4"]["startend"], [B, 1, S, 1])
        self.assertEqual(captured["dsa"]["q"], [B, S, H, Dk_mqa])
        self.assertEqual(captured["dsa"]["shared_key"], [B, S, Dk_mqa])
        self.assertEqual(captured["dsa"]["out"], [B, S, H * Dv_mqa])
        self.assertEqual(captured["dsa"]["kv_lora_rank"], Dv_mqa)
        for name in ("swa_attn_sink", "sparse_attn_sink"):
            self._assert_grad_ok(
                f"ff.{name}", getattr(swa_layer.self_attn, name)
            )
        self._assert_grad_ok(
            "ff.full.kv_a_proj_with_mqa",
            full_layer.self_attn.kv_a_proj_with_mqa.weight,
        )
        print(f"[runtime-ff-packed] {captured}")

    def _run_matrix_cell(self, config, state, inputs):
        full_layer, swa_layer = self._build_stack(config, full_recompute=True)
        full_layer.set_state_dict(state[0])
        swa_layer.set_state_dict(state[1])
        hidden_states, startend, ograd = inputs
        hs = hidden_states.detach()
        hs.stop_gradient = False
        paddle.seed(2026)
        model_parallel_cuda_manual_seed(2026)
        out_dict = full_layer(
            {
                "hidden_states": hs,
                "attn_mask_startend_row_indices": startend,
            }
        )
        out = swa_layer(out_dict)["hidden_states"]
        out_f32 = out.astype("float32")
        result = {
            "output": out_f32.detach().numpy(),
            "mse_loss": float(out_f32.square().mean().item()),
            "cotangent_loss": float(
                (out_f32 * ograd.astype("float32")).sum().item()
            ),
        }
        out.backward(ograd)
        result.update(
            {
                "input_grad": hs.grad.astype("float32").numpy(),
                "key_grad": full_layer.self_attn.kv_a_proj_with_mqa.weight.grad.astype(
                    "float32"
                ).numpy(),
                "swa_sink_grad": swa_layer.self_attn.swa_attn_sink.grad.astype(
                    "float32"
                ).numpy(),
                "sparse_sink_grad": swa_layer.self_attn.sparse_attn_sink.grad.astype(
                    "float32"
                ).numpy(),
            }
        )
        delta_params = {
            "key_delta": full_layer.self_attn.kv_a_proj_with_mqa.weight,
            "sparse_sink_delta": swa_layer.self_attn.sparse_attn_sink,
        }
        before = {
            name: param.detach().astype("float32").numpy().copy()
            for name, param in delta_params.items()
        }
        params = list(full_layer.parameters()) + list(swa_layer.parameters())
        opt = paddle.optimizer.AdamW(
            learning_rate=1e-2, parameters=params, multi_precision=True
        )
        opt.step()
        for name, param in delta_params.items():
            value = param.detach().astype("float32")
            self.assertTrue(bool(paddle.isfinite(value).all()))
            result[name] = value.numpy() - before[name]
        return result

    def _assert_matrix_close(self, label, got, ref):
        self.assertTrue(
            np.isfinite(got).all(), f"{label}: candidate is non-finite"
        )
        self.assertTrue(
            np.isfinite(ref).all(), f"{label}: reference is non-finite"
        )
        rel = _rel_l2(got, ref)
        print(f"[online packed matrix] {label}: rel-L2={rel:.6e}")
        self.assertLessEqual(rel, _BF16_REL_L2_TOL)

    def test_online_packed_backend_matrix(self):
        """Compare TT/TF/FT/FF at exact online attention dimensions."""
        _hysparse_backend_or_skip(self)
        paddle.seed(2026)
        model_parallel_cuda_manual_seed(2026)
        ref_cfg = dataclasses.replace(
            self.base_config,
            hy_sparse_full_attn_use_tilelang=True,
            hy_sparse_block_sparse_use_tilelang=True,
        )
        ref_full, ref_swa = self._build_stack(ref_cfg, full_recompute=True)
        for name in ("swa_attn_sink", "sparse_attn_sink"):
            sink = getattr(ref_swa.self_attn, name)
            sink.set_value(paddle.randn(sink.shape, dtype=sink.dtype) * 0.5)
        state = (
            {
                name: value.detach().clone()
                for name, value in ref_full.state_dict().items()
            },
            {
                name: value.detach().clone()
                for name, value in ref_swa.state_dict().items()
            },
        )
        del ref_full, ref_swa
        inputs = self._make_inputs([257, 511, 63, 1217])
        results = {}
        for label, (full_tl, sparse_tl) in _BACKEND_MATRIX.items():
            config = dataclasses.replace(
                self.base_config,
                hy_sparse_full_attn_use_tilelang=full_tl,
                hy_sparse_block_sparse_use_tilelang=sparse_tl,
            )
            results[label] = self._run_matrix_cell(config, state, inputs)

        fields = (
            "output",
            "input_grad",
            "key_grad",
            "swa_sink_grad",
            "sparse_sink_grad",
            "key_delta",
            "sparse_sink_delta",
        )
        for label in ("TF", "FT", "FF"):
            for field in fields:
                self._assert_matrix_close(
                    f"{label} vs TT [{field}]",
                    results[label][field],
                    results["TT"][field],
                )
            for loss_name in ("mse_loss", "cotangent_loss"):
                got = results[label][loss_name]
                ref = results["TT"][loss_name]
                rel = abs(got - ref) / max(abs(ref), 1e-12)
                print(
                    f"[online packed matrix] {label} vs TT [{loss_name}]: "
                    f"got={got:.8e} ref={ref:.8e} rel={rel:.6e}"
                )
                self.assertLessEqual(rel, _BF16_REL_L2_TOL)

    # FT packed docs: two length-1 docs plus several block-unaligned lengths
    # (65/63/130/511/1277 are none of them multiples of block_B=64), summing to
    # the full seq_len so nonzero-bos, ragged, short documents are exercised.
    _FT_PACKED_DOCS = [1, 1, 65, 63, 130, 511, 1277]

    def test_online_ft_runtime_dispatch_and_topk_identity(self):
        """FT (full=FA4, sparse=TileLang) packed-doc runtime routing + TopK.

        Proves under full recompute that:
          * FA4 scorer runs exactly twice (no-grad fwd + recompute bwd),
          * TileLang sparse gather runs exactly twice,
          * the TileLang scorer and cuDNN DSA are never touched,
          * the block-score / TopK selection produced by the no-grad forward
            pass and by the recompute pass are bit-identical, and the indices
            actually handed to the sparse op match them bit-for-bit.
        """
        _ft_backend_or_skip(self)
        self.assertEqual(sum(self._FT_PACKED_DOCS), self.seq_len)
        paddle.seed(2026)
        model_parallel_cuda_manual_seed(2026)

        import paddleformers.fleet.cudnn_ops as cudnn_pkg
        import paddleformers.fleet.tilelang_ops.hysparse as hy_pkg
        import paddleformers.fleet.tilelang_ops.hysparse.block_score_mha as bsm_mod
        import paddleformers.fleet.tilelang_ops.hysparse.block_sparse_mqa_tl as bsp_mod

        cap = {"fa4": 0, "tl_sparse": 0, "tl_full": 0, "dsa": 0}
        topk_caps = []  # (grad_enabled, indices_np) per scorer pass
        sparse_idx_caps = []  # indices_np handed to the sparse op per pass

        orig_fa4 = hy_pkg.block_score_fa4_attn_fwd
        orig_topk = hy_pkg.select_topk_blocks
        orig_tl_sparse = bsp_mod.block_sparse_mqa_attention_tl
        orig_tl_full = bsm_mod.block_score_mha_attn_fwd
        orig_dsa = cudnn_pkg.block_sparse_mqa_attention_dsa

        def spy_fa4(q, k, v, *a, **kw):
            cap["fa4"] += 1
            out = orig_fa4(q, k, v, *a, **kw)
            cap["fa4_shapes"] = {
                "q": _shape(q),
                "k": _shape(k),
                "v": _shape(v),
                "out": _shape(out[0]),
                "startend": _shape(kw.get("startend_row_indices")),
            }
            return out

        def spy_topk(*a, **kw):
            idx = orig_topk(*a, **kw)
            topk_caps.append((paddle.is_grad_enabled(), idx.numpy().copy()))
            return idx

        def spy_tl_sparse(q, shared_key_sq, block_indices, *a, **kw):
            cap["tl_sparse"] += 1
            sparse_idx_caps.append(block_indices.numpy().copy())
            out = orig_tl_sparse(q, shared_key_sq, block_indices, *a, **kw)
            cap["tl_sparse_shapes"] = {
                "q": _shape(q),
                "shared_key": _shape(shared_key_sq),
                "indices": _shape(block_indices),
                "out": _shape(out[0]),
            }
            return out

        def spy_tl_full(*a, **kw):
            cap["tl_full"] += 1
            return orig_tl_full(*a, **kw)

        def spy_dsa(*a, **kw):
            cap["dsa"] += 1
            return orig_dsa(*a, **kw)

        hy_pkg.block_score_fa4_attn_fwd = spy_fa4
        hy_pkg.select_topk_blocks = spy_topk
        bsp_mod.block_sparse_mqa_attention_tl = spy_tl_sparse
        bsm_mod.block_score_mha_attn_fwd = spy_tl_full
        cudnn_pkg.block_sparse_mqa_attention_dsa = spy_dsa
        try:
            cfg = dataclasses.replace(
                self.base_config,
                hy_sparse_full_attn_use_tilelang=False,
                hy_sparse_block_sparse_use_tilelang=True,
            )
            full_layer, swa_layer = self._build_stack(cfg, full_recompute=True)
            inputs = self._make_inputs(self._FT_PACKED_DOCS)
            swa_out, input_grad = self._forward_backward(
                full_layer, swa_layer, inputs
            )
        finally:
            hy_pkg.block_score_fa4_attn_fwd = orig_fa4
            hy_pkg.select_topk_blocks = orig_topk
            bsp_mod.block_sparse_mqa_attention_tl = orig_tl_sparse
            bsm_mod.block_score_mha_attn_fwd = orig_tl_full
            cudnn_pkg.block_sparse_mqa_attention_dsa = orig_dsa

        print(f"[ft-dispatch] calls={cap}")
        self.assertTrue(bool(paddle.isfinite(swa_out.astype("float32")).all()))
        self.assertTrue(
            bool(paddle.isfinite(input_grad.astype("float32")).all())
        )

        # (1) dispatch counts: FA4 + TileLang sparse each twice, no TL scorer,
        # no DSA.
        self.assertEqual(cap["fa4"], 2, "FA4 scorer should run fwd + recompute")
        self.assertEqual(
            cap["tl_sparse"], 2, "TileLang sparse should run fwd + recompute"
        )
        self.assertEqual(cap["tl_full"], 0, "TileLang scorer must not be used")
        self.assertEqual(cap["dsa"], 0, "DSA must not be used on the FT path")

        # (2) exact online shapes at the op boundary.
        B, S = self.batch_size, self.seq_len
        H = self.base_config.num_attention_heads
        Dk_full = (
            self.base_config.qk_nope_head_dim
            + self.base_config.qk_rope_head_dim
        )
        Dv_full = self.base_config.v_head_dim
        Dk_mqa = (
            self.base_config.kv_lora_rank + self.base_config.qk_rope_head_dim
        )
        Dv_mqa = self.base_config.kv_lora_rank
        topk = self.base_config.hy_sparse_topk
        fa4 = cap["fa4_shapes"]
        self.assertEqual(fa4["q"], [B, S, H, Dk_full])
        self.assertEqual(fa4["k"], [B, S, H, Dk_full])
        self.assertEqual(fa4["v"], [B, S, H, Dv_full])
        self.assertEqual(fa4["out"], [B, S, H, Dv_full])
        self.assertEqual(fa4["startend"], [B, 1, S, 1])
        tls = cap["tl_sparse_shapes"]
        self.assertEqual(tls["q"], [B, S, H, Dk_mqa])
        self.assertEqual(tls["shared_key"], [B, S, Dk_mqa])
        self.assertEqual(tls["indices"], [B, S, topk])
        self.assertEqual(tls["out"], [B, S, H * Dv_mqa])

        # (3) TopK identity across the no-grad forward and the recompute pass.
        self.assertEqual(
            len(topk_caps), 2, "scorer TopK should be computed twice"
        )
        grad_flags = [flag for flag, _ in topk_caps]
        self.assertIn(False, grad_flags, "one scorer pass must be no-grad")
        self.assertIn(True, grad_flags, "one scorer pass must be recompute")
        idx0, idx1 = topk_caps[0][1], topk_caps[1][1]
        self.assertEqual(idx0.shape, (B, S, topk))
        self.assertEqual(str(idx0.dtype), "int32")
        self.assertTrue(
            np.array_equal(idx0, idx1),
            "no-grad vs recompute TopK block indices differ",
        )
        # indices actually consumed by the sparse op match the scorer output
        # bit-for-bit on every pass.
        self.assertEqual(len(sparse_idx_caps), 2)
        for k, si in enumerate(sparse_idx_caps):
            self.assertTrue(
                np.array_equal(si, idx0),
                f"sparse-op indices (pass {k}) differ from scorer TopK",
            )
        print(
            "[ft-dispatch] TopK bit-identical across no-grad/recompute "
            f"(grad_flags={grad_flags}); sparse-op indices match scorer"
        )

    def test_online_ft_recompute_precision_packed(self):
        """FT recompute on/off numerical consistency at packed docs.

        Same object, same weights, same input and random dO; only
        ``full_recompute`` is toggled. Seeds finite but extreme sink biases and
        checks that output, input grad, the full-layer kv_a_proj grad and both
        learnable-sink grads are finite and agree to bf16 rel-L2 tolerance.
        """
        _ft_backend_or_skip(self)
        self.assertEqual(sum(self._FT_PACKED_DOCS), self.seq_len)
        paddle.seed(2026)
        model_parallel_cuda_manual_seed(2026)

        cfg = dataclasses.replace(
            self.base_config,
            hy_sparse_full_attn_use_tilelang=False,
            hy_sparse_block_sparse_use_tilelang=True,
        )
        full_layer, swa_layer = self._build_stack(cfg, full_recompute=False)
        # Finite but extreme sink biases: a wide symmetric span (well inside the
        # exp range so gradients stay nonzero, unlike the -1e30 sentinel).
        swa_sink = swa_layer.self_attn.swa_attn_sink
        sparse_sink = swa_layer.self_attn.sparse_attn_sink
        swa_sink.set_value(
            paddle.linspace(-4.0, 4.0, swa_sink.shape[0]).astype(swa_sink.dtype)
        )
        sparse_sink.set_value(
            paddle.linspace(4.0, -4.0, sparse_sink.shape[0]).astype(
                sparse_sink.dtype
            )
        )
        inputs = self._make_inputs(self._FT_PACKED_DOCS)

        def _run():
            swa_out, igrad = self._forward_backward(
                full_layer, swa_layer, inputs
            )
            return {
                "output": swa_out.astype("float32").numpy(),
                "input_grad": igrad.astype("float32").numpy(),
                "key_grad": full_layer.self_attn.kv_a_proj_with_mqa.weight.grad.astype(
                    "float32"
                ).numpy(),
                "swa_sink_grad": swa_layer.self_attn.swa_attn_sink.grad.astype(
                    "float32"
                ).numpy(),
                "sparse_sink_grad": swa_layer.self_attn.sparse_attn_sink.grad.astype(
                    "float32"
                ).numpy(),
            }

        def _clear():
            for p in list(full_layer.parameters()) + list(
                swa_layer.parameters()
            ):
                p.clear_gradient()

        # recompute OFF reference.
        off = _run()
        _clear()
        # same object, only toggle recompute ON.
        full_layer.full_recompute = True
        swa_layer.full_recompute = True
        on = _run()

        fields = (
            "output",
            "input_grad",
            "key_grad",
            "swa_sink_grad",
            "sparse_sink_grad",
        )
        for field in fields:
            g_off, g_on = off[field], on[field]
            self.assertTrue(
                np.isfinite(g_off).all(), f"{field}: recompute-off non-finite"
            )
            self.assertTrue(
                np.isfinite(g_on).all(), f"{field}: recompute-on non-finite"
            )
            rel = _rel_l2(g_on, g_off)
            print(
                f"[ft-recompute packed] {field}: rel-L2 on-vs-off = {rel:.6e}"
            )
            self.assertLessEqual(
                rel,
                _BF16_REL_L2_TOL,
                f"[{field}] FT recompute on/off mismatch rel-L2={rel:.6e}",
            )
        # sinks must carry a real (nonzero) gradient at these bias values, so
        # the rel-L2 checks above are not comparing degenerate all-zero vectors.
        for field in ("swa_sink_grad", "sparse_sink_grad"):
            self.assertGreater(
                float(np.linalg.norm(off[field].ravel())),
                0.0,
                f"{field}: sink gradient is all-zero (degenerate comparison)",
            )

    def test_online_recompute_precision(self):
        """Full-layer kv_a_proj + input-hidden grads agree recompute on vs off."""
        _hysparse_backend_or_skip(self)
        paddle.seed(2026)
        model_parallel_cuda_manual_seed(2026)
        inputs = self._make_inputs()

        # recompute OFF reference.
        full_off, swa_off = self._build_stack(
            self.base_config, full_recompute=False
        )
        _, igrad_off = self._forward_backward(full_off, swa_off, inputs)
        kv_off = full_off.self_attn.kv_a_proj_with_mqa.weight.grad.astype(
            "float32"
        )
        igrad_off = igrad_off.astype("float32")

        # recompute ON, identical weights.
        full_on, swa_on = self._build_stack(
            self.base_config, full_recompute=True
        )
        full_on.set_state_dict(full_off.state_dict())
        swa_on.set_state_dict(swa_off.state_dict())
        _, igrad_on = self._forward_backward(full_on, swa_on, inputs)
        kv_on = full_on.self_attn.kv_a_proj_with_mqa.weight.grad.astype(
            "float32"
        )
        igrad_on = igrad_on.astype("float32")

        for name, g_on, g_off in [
            ("kv_a_proj.weight", kv_on, kv_off),
            ("input_hidden", igrad_on, igrad_off),
        ]:
            self.assertTrue(bool(paddle.isfinite(g_on).all()))
            self.assertTrue(bool(paddle.isfinite(g_off).all()))
            denom = float(paddle.linalg.norm(g_off).item())
            rel = float(paddle.linalg.norm(g_on - g_off).item()) / max(
                denom, 1e-12
            )
            print(f"[recompute {name}] rel-L2 on-vs-off = {rel:.6e}")
            self.assertLessEqual(
                rel,
                _BF16_REL_L2_TOL,
                f"[{name}] recompute-on/off mismatch rel-L2={rel:.6e} "
                f"exceeds bf16 tol {_BF16_REL_L2_TOL}",
            )


if __name__ == "__main__":
    unittest.main()
