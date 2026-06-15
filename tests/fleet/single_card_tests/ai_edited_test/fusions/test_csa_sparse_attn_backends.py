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

"""Verify TileLang and cuDNN CSA sparse attention backends agree."""

import unittest

import paddle

try:
    import paddlefleet_ops

    from paddleformers.fleet.tilelang_ops.attn import sparse_mqa

    _HAS_FLASH_MLA = paddlefleet_ops.is_flash_mla_available() and sparse_mqa._flash_mla_sparse_fwd is not None
except (ImportError, RuntimeError, AttributeError):
    _HAS_FLASH_MLA = False

try:
    import cudnn  # noqa: F401

    from paddleformers.fleet.cudnn_ops import csa_sparse_attn_bwd_cudnn

    _HAS_CUDNN_SPARSE_BWD = callable(csa_sparse_attn_bwd_cudnn)
except (ImportError, ModuleNotFoundError, RuntimeError, AttributeError):
    _HAS_CUDNN_SPARSE_BWD = False


TEST_CASES = [
    (1, 128, 256, 64, 512, 64, "small-basic"),
    (2, 128, 256, 64, 512, 64, "batch2"),
    (4, 128, 256, 64, 512, 64, "batch4"),
    (1, 64, 128, 64, 512, 64, "tiny-seq"),
    (1, 256, 512, 64, 512, 128, "medium-seq"),
    (1, 512, 1024, 64, 512, 128, "large-seq"),
    (1, 1, 256, 64, 512, 64, "single-token"),
    (2, 1, 128, 64, 512, 64, "single-token-batch2"),
    (2, 256, 512, 64, 512, 192, "large-topk-192"),
    (1, 128, 512, 64, 512, 256, "large-topk-256"),
    (8, 64, 128, 64, 512, 64, "batch8"),
    (1, 1, 64, 64, 512, 64, "minimal"),
]

COS_THRESHOLDS = {
    "out": 0.99,
    "dq": 0.95,
    "dkv": 0.95,
    "d_sink": 0.95,
}


def cosine_sim(a, b):
    a_f = a.flatten().cast("float32")
    b_f = b.flatten().cast("float32")
    return float(paddle.nn.functional.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)))


def max_abs_diff(a, b):
    return float((a.cast("float32") - b.cast("float32")).abs().max())


def make_inputs(batch_size, seq_len, kv_seq_len, num_heads, head_dim, topk):
    q = paddle.randn([batch_size, seq_len, num_heads, head_dim]).cast("bfloat16")
    q.stop_gradient = False

    kv = paddle.randn([batch_size, kv_seq_len, head_dim]).cast("bfloat16")
    kv.stop_gradient = False

    attn_sink = paddle.randn([num_heads]).cast("float32") * 0.1
    attn_sink.stop_gradient = False

    topk_idxs = paddle.randint(0, kv_seq_len, [batch_size, seq_len, topk]).cast("int32")
    softmax_scale = 1.0 / (head_dim**0.5)
    return q, kv, attn_sink, topk_idxs, softmax_scale


def run_forward_backward(q, kv, attn_sink, topk_idxs, softmax_scale, backend):
    from paddleformers.fleet.fusions.csa_sparse_attn import csa_sparse_attn

    q_c = q.detach().clone()
    q_c.stop_gradient = False
    kv_c = kv.detach().clone()
    kv_c.stop_gradient = False
    attn_sink_c = attn_sink.detach().clone()
    attn_sink_c.stop_gradient = False

    out = csa_sparse_attn(q_c, kv_c, attn_sink_c, topk_idxs, softmax_scale, backend=backend)
    out.sum().backward()

    return out, q_c.grad, kv_c.grad, attn_sink_c.grad


def run_single_shape(batch_size, seq_len, kv_seq_len, num_heads, head_dim, topk):
    q, kv, attn_sink, topk_idxs, softmax_scale = make_inputs(
        batch_size, seq_len, kv_seq_len, num_heads, head_dim, topk
    )

    out_tl, dq_tl, dkv_tl, dsink_tl = run_forward_backward(
        q, kv, attn_sink, topk_idxs, softmax_scale, backend="tilelang"
    )
    out_cu, dq_cu, dkv_cu, dsink_cu = run_forward_backward(q, kv, attn_sink, topk_idxs, softmax_scale, backend="cudnn")

    if dsink_tl is None or dsink_cu is None:
        return False, {"d_sink": None}

    metrics = {
        "out": (cosine_sim(out_tl, out_cu), max_abs_diff(out_tl, out_cu)),
        "dq": (cosine_sim(dq_tl, dq_cu), max_abs_diff(dq_tl, dq_cu)),
        "dkv": (cosine_sim(dkv_tl, dkv_cu), max_abs_diff(dkv_tl, dkv_cu)),
        "d_sink": (
            cosine_sim(dsink_tl, dsink_cu),
            max_abs_diff(dsink_tl, dsink_cu),
        ),
    }
    passed = all(metrics[name][0] > COS_THRESHOLDS[name] for name in COS_THRESHOLDS)
    return passed, metrics


@unittest.skipUnless(
    paddle.is_compiled_with_cuda(),
    "CSA sparse attention backend comparison requires CUDA",
)
@unittest.skipUnless(
    _HAS_FLASH_MLA and _HAS_CUDNN_SPARSE_BWD,
    "CSA sparse attention backend comparison requires FlashMLA and cuDNN sparse backward",
)
class TestCSASparseAttentionBackends(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            paddle.set_device("gpu:0")
        except Exception as exc:
            raise unittest.SkipTest(f"gpu:0 is not available: {exc}")
        paddle.seed(2026)

    def test_tilelang_and_cudnn_backends_match(self):
        for case in TEST_CASES:
            (
                batch_size,
                seq_len,
                kv_seq_len,
                num_heads,
                head_dim,
                topk,
                label,
            ) = case
            with self.subTest(label=label):
                passed, metrics = run_single_shape(
                    batch_size,
                    seq_len,
                    kv_seq_len,
                    num_heads,
                    head_dim,
                    topk,
                )
                self.assertTrue(passed, f"{label} metrics={metrics}")
                paddle.device.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
