# Copyright (c) 2026 PaddleFleet Authors. All Rights Reserved.
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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import unittest

try:
    from paddleformers.fleet._extensions import ops  # noqa: F401

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False


@unittest.skipUnless(_MODULE_AVAILABLE, "paddleformers.fleet._extensions not available")
class TestExtensionsOpsFunctionCalls(unittest.TestCase):
    """Tests for _extensions.ops function call patterns."""

    def test_fused_swiglu_scale_signature(self):
        """fused_swiglu_scale should accept x and scale arguments."""
        import inspect

        from paddleformers.fleet._extensions.ops import fused_swiglu_scale

        sig = inspect.signature(fused_swiglu_scale)
        params = list(sig.parameters.keys())
        self.assertIn("x", params)
        self.assertIn("scale", params)

    def test_fused_swiglu_scale_bwd_signature(self):
        """fused_swiglu_scale_bwd should accept x, scale, dout arguments."""
        import inspect

        from paddleformers.fleet._extensions.ops import fused_swiglu_scale_bwd

        sig = inspect.signature(fused_swiglu_scale_bwd)
        params = list(sig.parameters.keys())
        self.assertIn("x", params)
        self.assertIn("scale", params)
        self.assertIn("dout", params)

    def test_filter_scores_signature(self):
        """filter_scores should accept probs and indices arguments."""
        import inspect

        from paddleformers.fleet._extensions.ops import filter_scores

        sig = inspect.signature(filter_scores)
        params = list(sig.parameters.keys())
        self.assertIn("probs", params)
        self.assertIn("indices", params)

    def test_router_metadata_signature(self):
        """router_metadata should accept topkrouterindices, expertfrequencyoffset, k."""
        import inspect

        from paddleformers.fleet._extensions.ops import router_metadata

        sig = inspect.signature(router_metadata)
        params = list(sig.parameters.keys())
        self.assertTrue(len(params) >= 3)

    def test_count_cumsum_signature(self):
        """count_cumsum should accept x, e, do_cumsum arguments."""
        import inspect

        from paddleformers.fleet._extensions.ops import count_cumsum

        sig = inspect.signature(count_cumsum)
        params = list(sig.parameters.keys())
        self.assertTrue(len(params) >= 3)

    def test_tokens_zip_prob_signature(self):
        """tokens_zip_prob should accept unzipped_prob, zipped_expertwise_rowmap, dispatched_indices."""
        import inspect

        from paddleformers.fleet._extensions.ops import tokens_zip_prob

        sig = inspect.signature(tokens_zip_prob)
        params = list(sig.parameters.keys())
        self.assertTrue(len(params) >= 3)

    def test_merge_subbatch_cast_signature(self):
        """merge_subbatch_cast should accept x and dtype arguments."""
        import inspect

        from paddleformers.fleet._extensions.ops import merge_subbatch_cast

        sig = inspect.signature(merge_subbatch_cast)
        params = list(sig.parameters.keys())
        self.assertIn("x", params)
        self.assertIn("dtype", params)

    def test_tokens_zip_unique_add_signature(self):
        """tokens_zip_unique_add should accept appropriate arguments."""
        import inspect

        from paddleformers.fleet._extensions.ops import tokens_zip_unique_add

        sig = inspect.signature(tokens_zip_unique_add)
        params = list(sig.parameters.keys())
        self.assertTrue(len(params) >= 4)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddleformers.fleet._extensions not available")
class TestExtensionsOpsUnifiedDecorator(unittest.TestCase):
    """Tests that ops functions are decorated with @unified."""

    def test_fused_swiglu_scale_has_unified(self):
        """fused_swiglu_scale should be decorated with @unified."""
        from paddleformers.fleet._extensions.ops import fused_swiglu_scale

        # The @unified decorator wraps the function
        self.assertTrue(callable(fused_swiglu_scale))

    def test_filter_scores_has_unified(self):
        """filter_scores should be decorated with @unified."""
        from paddleformers.fleet._extensions.ops import filter_scores

        self.assertTrue(callable(filter_scores))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddleformers.fleet._extensions not available")
class TestExtensionsInitModule(unittest.TestCase):
    """Tests for paddleformers.fleet._extensions.__init__ module."""

    def test_init_module_imports(self):
        """The _extensions module should be importable."""
        from paddleformers.fleet import _extensions

        self.assertIsNotNone(_extensions)

    def test_init_module_has_ops(self):
        """The _extensions module should expose ops."""
        from paddleformers.fleet._extensions import ops

        self.assertIsNotNone(ops)


if __name__ == "__main__":
    unittest.main()
