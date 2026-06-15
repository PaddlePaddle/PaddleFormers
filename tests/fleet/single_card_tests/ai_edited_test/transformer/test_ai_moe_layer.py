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
import contextlib
import io
import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import paddle

from paddleformers.fleet.transformer.moe import moe_layer
from paddleformers.fleet.transformer.moe.moe_layer import (
    GradDtypeGuard,
    GradDtypeUnguard,
    MoESublayers,
)
from paddleformers.fleet.transformer.transformer_layer import TransformerLayer


class GradContext:
    def __init__(self):
        self.grad_consistent = None

    def set_grad_in_dtype_consistent(self, value):
        self.grad_consistent = value


class TestMoELayerGradDtypeHelpers(unittest.TestCase):
    def test_grad_dtype_guard_apply_preserves_gradient(self):
        x = paddle.to_tensor([1.0, 2.0], dtype="float32")
        x.stop_gradient = False

        status, saved = GradDtypeGuard.apply(x, "float32")
        y = GradDtypeUnguard.apply(status, saved)
        y.sum().backward()

        self.assertEqual(status.shape, [0])
        self.assertEqual(status.dtype, paddle.float32)
        self.assertEqual(y.numpy().tolist(), [1.0, 2.0])
        self.assertEqual(x.grad.numpy().tolist(), [1.0, 1.0])

    def test_direct_forward_and_backward_methods(self):
        ctx = GradContext()
        x = paddle.to_tensor([3.0], dtype="float32")

        status, saved = GradDtypeGuard.forward(ctx, x, "float32")
        self.assertEqual(status.shape, [0])
        self.assertIs(saved["x"], x)
        self.assertIs(GradDtypeGuard.backward(ctx, x), x)

        unguard_ctx = GradContext()
        restored = GradDtypeUnguard.forward(unguard_ctx, status, saved)
        self.assertIs(restored, x)
        self.assertFalse(unguard_ctx.grad_consistent)
        self.assertIs(GradDtypeUnguard.backward(unguard_ctx, x), x)


class TestMoESublayersDataclass(unittest.TestCase):
    def test_default_and_custom_mlp_spec(self):
        class MLPObject:
            pass

        self.assertIsNone(MoESublayers().mlp_spec)
        self.assertIs(MoESublayers(mlp_spec=MLPObject).mlp_spec, MLPObject)


class TestLogMoEMD5(unittest.TestCase):
    def setUp(self):
        self.old_log = moe_layer._LOG_LAYER_MD5
        self.old_experimental = TransformerLayer._gpt_model_use_experimental_version
        self.old_skip = TransformerLayer._skip_mtp_probes

    def tearDown(self):
        moe_layer._LOG_LAYER_MD5 = self.old_log
        TransformerLayer._gpt_model_use_experimental_version = self.old_experimental
        TransformerLayer._skip_mtp_probes = self.old_skip

    def test_log_moe_md5_skip_probe_branch(self):
        moe_layer._LOG_LAYER_MD5 = True
        TransformerLayer._gpt_model_use_experimental_version = True
        TransformerLayer._skip_mtp_probes = True

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            moe_layer._log_moe_md5(paddle.to_tensor([1.0], dtype="float32"), "hidden", 7)

        self.assertEqual(captured.getvalue(), "")

    def test_log_moe_md5_prints_rank_layer_name_and_shape(self):
        moe_layer._LOG_LAYER_MD5 = True
        TransformerLayer._gpt_model_use_experimental_version = True
        TransformerLayer._skip_mtp_probes = False

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            moe_layer._log_moe_md5(paddle.to_tensor([1.0, 2.0], dtype="float32"), "hidden", 7)

        output = captured.getvalue()
        self.assertIn("[MD5 MoE]", output)
        self.assertIn("Rank=0", output)
        self.assertIn("Layer=7", output)
        self.assertIn("hidden", output)
        self.assertIn("shape=[2]", output)


if __name__ == "__main__":
    unittest.main()
