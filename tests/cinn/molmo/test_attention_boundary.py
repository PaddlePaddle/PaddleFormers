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

"""Check boundary equivalence and restoration without external assets."""
import unittest
from types import SimpleNamespace

import numpy as np
import paddle

from paddleformers.transformers.molmo import modeling
from tests.cinn.molmo.molmo_attention_boundary import (
    install,
    rope_with_explicit_head_axis,
)


class AttentionBoundaryTest(unittest.TestCase):
    def test_rope_matches_production_for_both_layouts(self):
        q = paddle.randn([2, 4, 3, 8])
        k = paddle.randn([2, 4, 3, 8])
        cos = paddle.randn([2, 3, 8])
        sin = paddle.randn([2, 3, 8])
        for rope_impl in ("interleave", "half"):
            expected = modeling.apply_rotary_pos_emb(q, k, cos, sin, rope_impl)
            for coefficients in ((cos, sin), (cos.unsqueeze(1), sin.unsqueeze(1))):
                actual = rope_with_explicit_head_axis(q, k, *coefficients, rope_impl=rope_impl)
                for left, right in zip(expected, actual):
                    np.testing.assert_array_equal(left.numpy(), right.numpy())

    def test_install_restores_global_and_instance_methods(self):
        class Attention(paddle.nn.Layer):
            def forward(self, value):
                return value

        attn = Attention()
        model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)]))
        original_rope = modeling.apply_rotary_pos_emb
        original_forward = attn.forward
        restore = install(model, compile_core=False)
        try:
            self.assertIs(modeling.apply_rotary_pos_emb, rope_with_explicit_head_axis)
            self.assertTrue(hasattr(attn, "_cinn_test_core"))
            with self.assertRaisesRegex(RuntimeError, "already patched"):
                install(model, compile_core=False)
        finally:
            restore()
        self.assertIs(modeling.apply_rotary_pos_emb, original_rope)
        self.assertEqual(attn.forward, original_forward)
        self.assertFalse(hasattr(attn, "_cinn_test_core"))
