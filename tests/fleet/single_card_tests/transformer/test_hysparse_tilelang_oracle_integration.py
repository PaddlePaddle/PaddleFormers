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

"""Network-level integration check for the split HySparse backend switches on
the two-layer HySparse wiring.

The HySparse path runs across two transformer layers:

  * a *full* layer (``HySparseTransformerLayer`` w/ ``full_recompute``) scores all
    key blocks and emits ``shared_key`` + ``shared_block_indices``;
  * a *SWA* layer consumes those, gathering the selected blocks in its
    block-sparse branch.

Each of the two branches picks its operator backend *independently* via two
config flags (the old monolithic ``hy_sparse_use_tilelang`` is now split):

  * ``hy_sparse_full_attn_use_tilelang``     -- full-attn block-score branch:
        True  -> TileLang ``block_score_mha_attn_fwd`` (oracle)
        False -> FA4 fused ``block_score_fa4_attn_fwd`` (production);
  * ``hy_sparse_block_sparse_use_tilelang``  -- block-sparse gather branch:
        True  -> TileLang ``block_sparse_mqa_attention_tl`` (oracle)
        False -> cuDNN-DSA ``block_sparse_mqa_attention_dsa`` (production).

That yields a 2x2 backend matrix, labelled ``<full><sparse>`` with
``T``=TileLang, ``F``=production:

  * ``TT`` -- pure TileLang oracle (both branches independently numeric-audited);
  * ``TF`` -- TileLang scorer  + DSA gather   (isolates the gather backend);
  * ``FT`` -- FA4 scorer       + TileLang gather (isolates the scorer backend);
  * ``FF`` -- pure production   (FA4 scorer + DSA gather) -- the *suspected* path.

Each op is already cross-checked op-by-op at bf16 precision (fwd+bwd, exact
``block_logit`` and TopK-index bridge). This test closes the loop *in the real
network wiring*: it builds the identical two-layer stack four times (same
weights, same input, same random dO), runs one stack per matrix cell, and
compares the final SWA-layer output, the input-hidden-state gradient, the
shared-key producer gradient, and (when the fixture enables them) the learnable
sink gradients against the audited ``TT`` oracle -- using a finite check plus a
magnitude-sensitive relative-L2 metric. Because ``FT``/``TF`` each differ from
``FF`` in exactly one branch, an ``FF`` divergence localises to the scorer
(``FF`` vs ``TF``) or the gather (``FF`` vs ``FT``); a pairwise rel-L2 matrix is
printed to aid that triage.

Requires an SM 10.x (Blackwell) device with the FA4 FlashMask CUTE backend and
the cuDNN DSA backend available; skips otherwise (all four cells need at least
one of FA4/DSA, and the full matrix needs both).
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


# Backend matrix cells, labelled ``<full><sparse>`` where the tuple is
# (hy_sparse_full_attn_use_tilelang, hy_sparse_block_sparse_use_tilelang):
#   T=TileLang oracle, F=production (FA4 scorer / cuDNN-DSA gather).
BACKEND_MATRIX = {
    "TT": (True, True),  # pure oracle (audit reference)
    "TF": (True, False),  # TileLang scorer + DSA gather
    "FT": (False, True),  # FA4 scorer + TileLang gather
    "FF": (False, False),  # pure production (suspected)
}


def _to_np(tensor):
    """Detach a tensor to a float32 numpy array (None passes through)."""
    if tensor is None:
        return None
    return tensor.detach().astype("float32").numpy()


def _rel_l2(got, ref):
    """Magnitude-sensitive relative-L2: ||got-ref|| / ||ref|| (scale aware).

    Falls back to the absolute error norm when the reference is a (near-)zero
    tensor so a degenerate zero-gradient does not divide by zero.
    """
    diff = float(np.linalg.norm((got - ref).ravel()))
    denom = float(np.linalg.norm(ref.ravel()))
    return diff / denom if denom > 1e-12 else diff


class TestHySparseTileLangOracleIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_size = 2
        cls.seq_len = 4096
        cls.run_seed = 2026

        # base config == the production HySparse MQA wiring. Learnable sink bias
        # is enabled so the four-way matrix also exercises (and compares) the
        # windowed + block-sparse sink gradients -- the finite-sink gather path
        # is the most backend-divergence-prone part of the FF production route.
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
            add_swa_attention_sink_bias=True,
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

    def _build_stack(self, config):
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
        full_layer.full_recompute = True

        swa_layer = HySparseTransformerLayer(config, layer_spec, layer_number=1)
        swa_layer.self_attn.attn_mask_type = AttnMaskType.causal
        swa_layer = paddle.amp.decorate(swa_layer, level="O2", dtype="bfloat16")
        # full_recompute short-circuits the ``has_recovered()`` recovery-window
        # check in HySparseTransformerLayer.forward; set it on both layers so the
        # test exercises the operator swap without depending on recovery state.
        swa_layer.full_recompute = True
        return full_layer, swa_layer

    def _seed_sinks(self, swa_layer):
        """Give the SWA layer's learnable sinks a material (non-zero) value.

        Sinks initialise to 0.0; a zero sink still contributes ``exp(0)=1`` to
        the denominator but keeps every cell's sink identical, so we seed a
        randn value (later mirrored to the other cells via ``set_state_dict``)
        to make the finite-sink path -- and its gradient -- non-degenerate and
        genuinely backend-sensitive.
        """
        for name in ("swa_attn_sink", "sparse_attn_sink"):
            param = getattr(swa_layer.self_attn, name, None)
            if param is not None:
                param.set_value(
                    paddle.randn(param.shape, dtype=param.dtype) * 0.5
                )

    def _grad_bundle(self, full_layer, swa_layer, hs):
        """Collect the comparison tensors as float32 numpy arrays.

        * ``key_grad``  -- full-layer ``kv_a_proj_with_mqa`` weight gradient; the
          whole ``shared_key`` (concat of kv_compressed + k_pos_emb) traces to
          it, so this is the shared-key producer gradient that flows back from
          the downstream SWA block-sparse branch.
        * ``swa_attn_sink`` / ``sparse_attn_sink`` -- SWA learnable-sink grads
          (present only when ``add_swa_attention_sink_bias`` is on).
        """
        bundle = {"input_grad": _to_np(hs.grad)}
        key_param = full_layer.self_attn.kv_a_proj_with_mqa.weight
        bundle["key_grad"] = _to_np(key_param.grad)
        for name in ("swa_attn_sink", "sparse_attn_sink"):
            param = getattr(swa_layer.self_attn, name, None)
            bundle[name] = _to_np(param.grad) if param is not None else None
        return bundle

    def _run_stack(self, full_layer, swa_layer, hidden_states, startend, ograd):
        """Full -> SWA forward, backward for one matrix cell.

        Returns ``(swa_out_np, grad_bundle)``. The RNG is reseeded per cell so
        any stochastic op (e.g. bias-dropout residual) draws an identical mask
        across backends, isolating the comparison to the operator swap.
        """
        paddle.seed(self.run_seed)
        model_parallel_cuda_manual_seed(self.run_seed)
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
        return _to_np(swa_out), self._grad_bundle(full_layer, swa_layer, hs)

    def _assert_close(self, tag, got, ref, rel_tol=0.1):
        """Finite + magnitude-sensitive relative-L2 gate; returns the rel-L2.

        ``allclose`` is deliberately *not* the hard gate for gradients: several
        of these (notably ``key_grad``) are heavy near-cancellations whose tiny
        entries fail element-wise bf16 rtol while the aggregate is sound, so
        rel-L2 is the meaningful magnitude-sensitive signal.
        """
        self.assertIsNotNone(got, f"{tag}: value missing on candidate cell")
        self.assertIsNotNone(ref, f"{tag}: value missing on reference cell")
        self.assertTrue(
            np.isfinite(got).all(), f"{tag}: candidate has non-finite entries"
        )
        self.assertTrue(
            np.isfinite(ref).all(), f"{tag}: reference has non-finite entries"
        )
        rl2 = _rel_l2(got, ref)
        self.assertLessEqual(
            rl2, rel_tol, f"{tag}: rel-L2 {rl2:.3e} exceeds tol {rel_tol}"
        )
        return rl2

    def test_backend_matrix_ff(self):
        _hysparse_backend_or_skip(self)
        paddle.seed(2026)
        model_parallel_cuda_manual_seed(2026)

        # Build one stack per matrix cell, all wired to the same fixture.
        stacks = {}
        for label, (full_tl, sparse_tl) in BACKEND_MATRIX.items():
            cfg = dataclasses.replace(
                self.base_config,
                hy_sparse_full_attn_use_tilelang=full_tl,
                hy_sparse_block_sparse_use_tilelang=sparse_tl,
            )
            stacks[label] = self._build_stack(cfg)

        # Reference cell (pure oracle) sets the shared weights + seeded sinks;
        # every other cell is copied bit-exactly from it so the ONLY difference
        # across cells is the operator backend, not the parameters.
        ref_full, ref_swa = stacks["TT"]
        self._seed_sinks(ref_swa)
        ref_state_full = ref_full.state_dict()
        ref_state_swa = ref_swa.state_dict()
        for label, (full_layer, swa_layer) in stacks.items():
            if label == "TT":
                continue
            full_layer.set_state_dict(ref_state_full)
            swa_layer.set_state_dict(ref_state_swa)

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

        # Run every cell on the identical input + random dO.
        outputs, bundles = {}, {}
        for label, (full_layer, swa_layer) in stacks.items():
            outputs[label], bundles[label] = self._run_stack(
                full_layer, swa_layer, hidden_states, startend, ograd
            )

        # Quantities compared against the audited TT oracle.
        fields = ["output", "input_grad", "key_grad"]
        if self.base_config.add_swa_attention_sink_bias:
            fields += ["swa_attn_sink", "sparse_attn_sink"]

        def _value(label, field):
            return (
                outputs[label] if field == "output" else bundles[label][field]
            )

        # Primary gate: TF / FT / FF each vs the TT oracle.
        print("\n[HySparse 2x2 backend matrix] rel-L2 vs TT oracle:")
        for label in ("TF", "FT", "FF"):
            rows = []
            for field in fields:
                rl2 = self._assert_close(
                    f"{label} vs TT [{field}]",
                    _value(label, field),
                    _value("TT", field),
                )
                rows.append(f"{field}={rl2:.2e}")
            print(f"  {label}: " + "  ".join(rows))

        # Localisation aid for a suspected FF anomaly: FF differs from TF only
        # in the scorer (FA4 vs TileLang) and from FT only in the gather (DSA vs
        # TileLang), so a large single-column rel-L2 pinpoints the branch.
        print("[HySparse 2x2 backend matrix] FF localisation rel-L2:")
        for base, isolates in (
            ("TF", "scorer(FA4 vs TL)"),
            ("FT", "gather(DSA vs TL)"),
        ):
            rows = [
                f"{field}={_rel_l2(_value('FF', field), _value(base, field)):.2e}"
                for field in fields
            ]
            print(f"  FF vs {base} [isolates {isolates}]: " + "  ".join(rows))

        # bf16 element-wise agreement of the production (FF) output vs the
        # oracle -- kept as an extra, stricter-shaped check on the final output.
        np.testing.assert_allclose(
            _value("FF", "output"),
            _value("TT", "output"),
            atol=8e-2,
            rtol=8e-2,
        )


if __name__ == "__main__":
    unittest.main()
