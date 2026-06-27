# Copyright (c) 2026 PaddleFaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless distributed on the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest

try:
    from paddleformers.fleet._extensions import ops  # noqa: F401

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False


@unittest.skipUnless(_MODULE_AVAILABLE, "paddleformers.fleet._extensions not available")
class TestExtensionsOpsModule(unittest.TestCase):
    """Tests for paddleformers.fleet._extensions.ops module functions."""

    def test_module_imports(self):
        """The _extensions.ops module should be importable."""
        from paddleformers.fleet._extensions import ops

        self.assertIsNotNone(ops)

    def test_module_has_tokens_unzip_gather(self):
        """Module should have tokens_unzip_gather function."""
        from paddleformers.fleet._extensions import ops

        self.assertTrue(hasattr(ops, "tokens_unzip_gather"))

    def test_module_has_fused_swiglu_scale(self):
        """Module should have fused_swiglu_scale function."""
        from paddleformers.fleet._extensions import ops

        self.assertTrue(hasattr(ops, "fused_swiglu_scale"))

    def test_module_has_fused_swiglu_scale_bwd(self):
        """Module should have fused_swiglu_scale_bwd function."""
        from paddleformers.fleet._extensions import ops

        self.assertTrue(hasattr(ops, "fused_swiglu_scale_bwd"))

    def test_module_has_fuse_stack_transpose_fp8_quant(self):
        """Module should have fuse_stack_transpose_fp8_quant function."""
        from paddleformers.fleet._extensions import ops

        self.assertTrue(hasattr(ops, "fuse_stack_transpose_fp8_quant"))

    def test_module_has_tokens_unzip_stable(self):
        """Module should have tokens_unzip_stable function."""
        from paddleformers.fleet._extensions import ops

        self.assertTrue(hasattr(ops, "tokens_unzip_stable"))

    def test_module_has_tokens_zip_prob(self):
        """Module should have tokens_zip_prob function."""
        from paddleformers.fleet._extensions import ops

        self.assertTrue(hasattr(ops, "tokens_zip_prob"))

    def test_module_has_fused_apply_rotary_pos_emb_vision(self):
        """Module should have fused_apply_rotary_pos_emb_vision function."""
        from paddleformers.fleet._extensions import ops

        self.assertTrue(hasattr(ops, "fused_apply_rotary_pos_emb_vision"))

    def test_module_has_filter_scores(self):
        """Module should have filter_scores function."""
        from paddleformers.fleet._extensions import ops

        self.assertTrue(hasattr(ops, "filter_scores"))

    def test_module_has_router_metadata(self):
        """Module should have router_metadata function."""
        from paddleformers.fleet._extensions import ops

        self.assertTrue(hasattr(ops, "router_metadata"))

    def test_module_has_count_cumsum(self):
        """Module should have count_cumsum function."""
        from paddleformers.fleet._extensions import ops

        self.assertTrue(hasattr(ops, "count_cumsum"))

    def test_module_has_fuse_stack_fp8_quant(self):
        """Module should have fuse_stack_fp8_quant function."""
        from paddleformers.fleet._extensions import ops

        self.assertTrue(hasattr(ops, "fuse_stack_fp8_quant"))

    def test_module_has_tokens_unzip_slice(self):
        """Module should have tokens_unzip_slice function."""
        from paddleformers.fleet._extensions import ops

        self.assertTrue(hasattr(ops, "tokens_unzip_slice"))

    def test_module_has_tokens_zip_unique_add(self):
        """Module should have tokens_zip_unique_add function."""
        from paddleformers.fleet._extensions import ops

        self.assertTrue(hasattr(ops, "tokens_zip_unique_add"))

    def test_module_has_merge_subbatch_cast(self):
        """Module should have merge_subbatch_cast function."""
        from paddleformers.fleet._extensions import ops

        self.assertTrue(hasattr(ops, "merge_subbatch_cast"))

    def test_module_has_fuse_weighted_swiglu_fp8_quant(self):
        """Module should have fuse_weighted_swiglu_fp8_quant function."""
        from paddleformers.fleet._extensions import ops

        self.assertTrue(hasattr(ops, "fuse_weighted_swiglu_fp8_quant"))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddleformers.fleet._extensions not available")
class TestExtensionsOpsFunctionSignatures(unittest.TestCase):
    """Tests for ops function signatures and basic properties."""

    def test_tokens_unzip_gather_is_callable(self):
        """tokens_unzip_gather should be callable."""
        from paddleformers.fleet._extensions.ops import tokens_unzip_gather

        self.assertTrue(callable(tokens_unzip_gather))

    def test_fused_swiglu_scale_is_callable(self):
        """fused_swiglu_scale should be callable."""
        from paddleformers.fleet._extensions.ops import fused_swiglu_scale

        self.assertTrue(callable(fused_swiglu_scale))

    def test_filter_scores_is_callable(self):
        """filter_scores should be callable."""
        from paddleformers.fleet._extensions.ops import filter_scores

        self.assertTrue(callable(filter_scores))

    def test_router_metadata_is_callable(self):
        """router_metadata should be callable."""
        from paddleformers.fleet._extensions.ops import router_metadata

        self.assertTrue(callable(router_metadata))

    def test_count_cumsum_is_callable(self):
        """count_cumsum should be callable."""
        from paddleformers.fleet._extensions.ops import count_cumsum

        self.assertTrue(callable(count_cumsum))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddleformers.fleet._extensions not available")
class TestExtensionsOpsBootstrap(unittest.TestCase):
    """Tests for the bootstrap mechanism of _extensions.ops."""

    def test_so_path_exists(self):
        """The ops_pd_.so shared library should exist."""
        import paddleformers.fleet._extensions

        cur_dir = os.path.dirname(
            os.path.abspath(paddleformers.fleet._extensions.__file__)
        )
        so_path = os.path.join(cur_dir, "ops_pd_.so")
        self.assertTrue(
            os.path.exists(so_path), f"SO file not found at {so_path}"
        )


if __name__ == "__main__":
    unittest.main()
