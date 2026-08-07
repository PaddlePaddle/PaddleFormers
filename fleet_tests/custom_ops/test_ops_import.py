# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import unittest


class TestOpsImport(unittest.TestCase):
    TARGET_OPS = [
        "fuse_transpose_split_fp8_quant",
        "tokens_unzip_gather",
        "tokens_unzip_slice",
        "tokens_unzip_stable",
        "tokens_zip_prob",
        "tokens_zip_prob_seq_subbatch",
        "tokens_zip_unique_add",
        "tokens_zip_unique_add_subbatch",
        "fused_swiglu_scale",
        "fused_swiglu_scale_bwd",
        "fuse_weighted_swiglu_fp8_quant",
        "fuse_stack_transpose_fp8_quant",
        "fuse_stack_fp8_quant",
    ]

    def setUp(self):
        try:
            import paddlefleet_ops

            self.ops = paddlefleet_ops
        except Exception as e:
            self.fail(f"Failed to import paddlefleet_ops: {e}")

    def test_import_ops(self):
        self.assertIsNotNone(self.ops, "Failed to import paddlefleet_ops")

    def test_ops_submodule_availability(self):
        if self.ops is None:
            self.skipTest(
                "paddlefleet_ops not available. Skipping op availability tests."
            )
        else:
            self.assertIsNotNone(
                self.ops,
                "paddlefleet_ops is None, expected it to be loaded.",
            )

    def test_tokens_ops_availability(self):
        if self.ops is None:
            self.skipTest(
                "paddlefleet_ops not available. Skipping tokens_ ops availability tests."
            )
            return

        missing_ops = []
        for op_name in self.TARGET_OPS:
            if not hasattr(self.ops, op_name):
                missing_ops.append(op_name)

        if missing_ops:
            self.fail(
                f"The following operators are missing from paddlefleet_ops "
                f"(C++ extension likely not compiled correctly or is outdated): {', '.join(missing_ops)}"
            )


class TestDeepGEMMImport(unittest.TestCase):
    def test_deep_gemm_import(self):
        import paddlefleet_ops
        from paddlefleet_ops.deep_gemm import (  # noqa: F401
            cublaslt_gemm_tn,
            set_num_sms,
        )

        print(paddlefleet_ops.deep_gemm)

    def test_error_import(self):
        with self.assertRaises(ImportError):
            from paddlefleet_ops.deep_gemm import xxxx  # noqa: F401


class TestDeepEPImport(unittest.TestCase):
    def test_deep_gemm_import(self):
        import paddlefleet_ops
        from paddlefleet_ops.deep_ep import (  # noqa: F401
            Buffer,
            Config,
            EventOverlap,
            topk_idx_t,
        )

        print(paddlefleet_ops.deep_ep)

    def test_error_import(self):
        with self.assertRaises(ImportError):
            from paddlefleet_ops.deep_ep import xxxx  # noqa: F401


class TestFastHadamardTransformImport(unittest.TestCase):
    def test_fast_hadamard_transform_import(self):
        import paddlefleet_ops
        from paddlefleet_ops.fast_hadamard_transform import (
            hadamard_transform,
        )

        self.assertTrue(callable(hadamard_transform))
        print(paddlefleet_ops.fast_hadamard_transform)

    def test_error_import(self):
        with self.assertRaises(ImportError):
            from paddlefleet_ops.fast_hadamard_transform import (
                xxxx,  # noqa: F401
            )


if __name__ == "__main__":
    unittest.main()
