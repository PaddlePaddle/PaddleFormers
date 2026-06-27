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


class TestFusedRmsNorm(unittest.TestCase):
    """Tests for fused_rms_norm module."""

    def test_fused_rms_norm_import(self):
        """Test fused_rms_norm module can be imported."""
        from paddleformers.fleet.fusions import fused_rms_norm

        self.assertIsNotNone(fused_rms_norm)

    def test_fused_rms_norm_has_function(self):
        """Test fused_rms_norm has expected functions."""
        from paddleformers.fleet.fusions import fused_rms_norm

        # Module should have some attributes
        self.assertIsNotNone(fused_rms_norm)


class TestFusedLayerNorm(unittest.TestCase):
    """Tests for fused_layer_norm module."""

    def test_fused_layer_norm_import(self):
        """Test fused_layer_norm module can be imported."""
        from paddleformers.fleet.fusions import fused_layer_norm

        self.assertIsNotNone(fused_layer_norm)


class TestFusedBiasDropout(unittest.TestCase):
    """Tests for fused_bias_dropout module."""

    def test_module_import(self):
        """Test fused_bias_dropout module can be imported."""
        from paddleformers.fleet.fusions import fused_bias_dropout

        self.assertIsNotNone(fused_bias_dropout)

    def test_has_bias_dropout_add_func(self):
        """Test module has _bias_dropout_add_func."""
        from paddleformers.fleet.fusions.fused_bias_dropout import (
            _bias_dropout_add_func,
        )

        self.assertTrue(callable(_bias_dropout_add_func))


class TestFusedSwigluScale(unittest.TestCase):
    """Tests for fused_swiglu_scale module."""

    def test_module_import(self):
        """Test fused_swiglu_scale module can be imported."""
        from paddleformers.fleet.fusions import fused_swiglu_scale

        self.assertIsNotNone(fused_swiglu_scale)

    def test_has_forward_and_backward(self):
        """Test module has forward and backward functions."""
        from paddleformers.fleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
            fused_swiglu_scale_forward,
        )

        self.assertTrue(callable(fused_swiglu_scale_forward))
        self.assertTrue(callable(fused_swiglu_scale_backward))


class TestFusedSoftmaxModule(unittest.TestCase):
    """Tests for fused_softmax module."""

    def test_module_import(self):
        """Test fused_softmax module can be imported."""
        from paddleformers.fleet.fusions import fused_softmax

        self.assertIsNotNone(fused_softmax)

    def test_has_softmax_one(self):
        """Test module has SoftmaxOne class."""
        from paddleformers.fleet.fusions.fused_softmax import SoftmaxOne

        self.assertTrue(callable(SoftmaxOne))

    def test_has_fused_scale_mask_softmax(self):
        """Test module has FusedScaleMaskSoftmax class."""
        from paddleformers.fleet.fusions.fused_softmax import FusedScaleMaskSoftmax

        self.assertTrue(callable(FusedScaleMaskSoftmax))


if __name__ == "__main__":
    unittest.main()
