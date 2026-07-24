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

"""Shared-KV recompute gradient check for the two-layer HySparse wiring.

Background
----------
On the HySparse path a *full* attention layer emits ``shared_key`` (a slice of
its compressed KV latent, ``concat(kv_compressed, k_pos_emb)``) plus the top-k
``shared_block_indices``; a downstream *SWA* layer consumes both in its
block-sparse branch. ``shared_key`` therefore traces entirely back to the full
layer's ``kv_a_proj_with_mqa`` weight.

Under ``full_recompute`` the full layer's forward re-runs inside a ``no_grad``
recompute region, so ``kv_compressed`` (and hence ``shared_key``) would default
to ``stop_gradient=True`` and sever the sparse-branch gradient edge. The linchpin
guard in ``HySparseTransformerLayer._forward_impl``::

    if self.training and not paddle.is_grad_enabled():
        shared_key.stop_gradient = False

flips it back so the recompute PyLayer registers a backward edge for the
secondary ``shared_key`` output. This test verifies that the full layer's
``kv_a_proj_with_mqa.weight`` gradient actually arrives *only* via the SWA
sparse branch, and that it is finite, non-zero, and agrees between the
recompute-on and recompute-off paths.

Isolation
---------
After running the full layer we detach the dense hidden-state path
(``out_dict['hidden_states']``) so no gradient can reach ``kv_a_proj`` through
the full layer's own attention/MLP output. Only ``shared_key`` keeps a live
graph into the SWA layer, so any non-zero ``kv_a_proj`` gradient is proof the
shared-KV edge survives.

Coverage: {production FA4/DSA, TileLang oracle} x {recompute off, recompute on}.

Requires an SM 10.x (Blackwell) device with the FA4 FlashMask CUTE backend and
the cuDNN DSA backend available; skips otherwise.
"""

import os
import unittest

os.environ["FLAGS_cudnn_deterministic"] = "True"

import dataclasses

import paddle
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddleformers.fleet.fusions.fused_bias_dropout import get_bias_dropout_add
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

# bf16 relative-L2 tolerance for the recompute-on vs recompute-off comparison.
_BF16_REL_L2_TOL = 0.1


def _hysparse_backend_or_skip(testcase):
    """Skip unless BOTH the production FA4/DSA and TileLang backends can run."""
    if not paddle.device.is_compiled_with_cuda():
        testcase.skipTest("CUDA build of Paddle is required")
    if paddle.device.cuda.device_count() == 0:
        testcase.skipTest("No CUDA device available")
    major = paddle.device.cuda.get_device_capability()[0]
    if major != 10:
        testcase.skipTest(
            f"HySparse FA4/DSA + TileLang require SM 10.x; got SM {major}.x"
        )
    try:
        import paddlefleet_ops

        if not paddlefleet_ops.is_flash_mask_available():
            testcase.skipTest("FlashMask (FA4) backend not available")
        from paddleformers.fleet.cudnn_ops import is_dsa_available

        if not is_dsa_available():
            testcase.skipTest("cuDNN DSA backend not available")
    except (ImportError, RuntimeError):
        testcase.skipTest("HySparse FA4/DSA backend import failed")


class TestHySparseRecomputeSharedKVGrad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_size = 2
        cls.seq_len = 4096

        # base config == the production HySparse MQA wiring (oracle OFF).
        cls.base_config = TransformerConfig(
            hidden_size=1536,
            head_dim=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            gated_attention=True,
            qk_rope_head_dim=64,
            qk_nope_head_dim=192,
            v_head_dim=256,
            kv_lora_rank=512,
            rope_theta=5000000,
            use_qk_norm=True,
            multi_latent_attention=True,
            rope_type="rope",
            add_swa_attention_sink_bias=False,
            sliding_window=[128, 128],
            window_attn_skip_freq=2,
            enable_hy_sparse_attention=True,
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
        """Build a (full, swa) HySparseTransformerLayer pair for a config."""
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

    def _shared_kv_grad(
        self, full_layer, swa_layer, hidden_states, startend, ograd
    ):
        """Full -> (detach dense path) -> SWA -> backward.

        Returns the full layer's ``kv_a_proj_with_mqa.weight`` gradient, which
        under the isolation below can only have arrived through the ``shared_key``
        edge into the SWA sparse branch.
        """
        full_layer.train()
        swa_layer.train()

        hs = hidden_states.detach()
        hs.stop_gradient = False
        out_dict = full_layer(
            {
                "hidden_states": hs,
                "attn_mask_startend_row_indices": startend,
            }
        )

        # Isolation: sever the dense hidden-state path so no gradient reaches
        # kv_a_proj through the full layer's own attention/MLP output. Keep the
        # shared_key graph alive as the sole route back into the full layer. The
        # explicit flag does not recreate a missing recompute PyLayer edge; if
        # the production linchpin failed, the probed weight grad remains absent.
        self.assertIn("shared_key", out_dict)
        self.assertIsNotNone(out_dict["shared_key"])
        out_dict["hidden_states"] = out_dict["hidden_states"].detach()
        out_dict["shared_key"].stop_gradient = False

        out_dict = swa_layer(out_dict)
        swa_out = out_dict["hidden_states"]
        swa_out.backward(ograd)
        return full_layer.self_attn.kv_a_proj_with_mqa.weight.grad

    def _make_inputs(self):
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

    def _grad_for_mode(
        self, use_tilelang, full_recompute, inputs, ref_state=None
    ):
        """Run one (backend, recompute-mode) cell; return grad + state dict.

        ``ref_state`` (if given) is loaded into the freshly built stack so the
        recompute-on and recompute-off cells share identical weights, making
        their gradients directly comparable.
        """
        cfg = dataclasses.replace(
            self.base_config,
            hy_sparse_full_attn_use_tilelang=use_tilelang,
            hy_sparse_block_sparse_use_tilelang=use_tilelang,
        )
        full_layer, swa_layer = self._build_stack(cfg, full_recompute)
        if ref_state is not None:
            full_layer.set_state_dict(ref_state[0])
            swa_layer.set_state_dict(ref_state[1])
        hidden_states, startend, ograd = inputs
        grad = self._shared_kv_grad(
            full_layer, swa_layer, hidden_states, startend, ograd
        )
        state = (full_layer.state_dict(), swa_layer.state_dict())
        return grad, state

    def _assert_grad_ok(self, backend, mode, grad):
        self.assertIsNotNone(
            grad,
            f"[{backend}/{mode}] kv_a_proj_with_mqa.weight.grad is None -- "
            "shared_key gradient edge is broken",
        )
        grad_f = grad.astype("float32")
        finite = bool(paddle.isfinite(grad_f).all())
        norm = float(paddle.linalg.norm(grad_f).item())
        print(
            f"[{backend}/{mode}] shape={list(grad.shape)} dtype={grad.dtype} "
            f"finite={finite} norm={norm:.6e}"
        )
        self.assertTrue(finite, f"[{backend}/{mode}] gradient has NaN/Inf")
        self.assertGreater(
            norm, 0.0, f"[{backend}/{mode}] gradient is all-zero"
        )
        return grad_f

    def _check_backend(self, backend, use_tilelang, inputs):
        # recompute OFF is the reference; recompute ON reuses its weights.
        grad_off, ref_state = self._grad_for_mode(
            use_tilelang, full_recompute=False, inputs=inputs
        )
        g_off = self._assert_grad_ok(backend, "recompute_off", grad_off)

        grad_on, _ = self._grad_for_mode(
            use_tilelang,
            full_recompute=True,
            inputs=inputs,
            ref_state=ref_state,
        )
        g_on = self._assert_grad_ok(backend, "recompute_on", grad_on)

        denom = float(paddle.linalg.norm(g_off).item())
        rel = float(paddle.linalg.norm(g_on - g_off).item()) / max(denom, 1e-12)
        print(f"[{backend}] recompute_on vs off rel-L2={rel:.6e}")
        self.assertLessEqual(
            rel,
            _BF16_REL_L2_TOL,
            f"[{backend}] recompute-on/off gradient mismatch rel-L2={rel:.6e} "
            f"exceeds bf16 tol {_BF16_REL_L2_TOL}",
        )

    def test_production_shared_kv_grad_recompute(self):
        _hysparse_backend_or_skip(self)
        paddle.seed(2026)
        model_parallel_cuda_manual_seed(2026)
        self._check_backend(
            "production_fa4_dsa", use_tilelang=False, inputs=self._make_inputs()
        )

    def test_tilelang_shared_kv_grad_recompute(self):
        _hysparse_backend_or_skip(self)
        paddle.seed(2026)
        model_parallel_cuda_manual_seed(2026)
        self._check_backend(
            "tilelang_oracle", use_tilelang=True, inputs=self._make_inputs()
        )


if __name__ == "__main__":
    unittest.main()
