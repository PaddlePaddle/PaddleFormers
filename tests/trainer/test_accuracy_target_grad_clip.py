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
"""How ``use_accuracy_compatible`` interacts with ``max_grad_norm``.

The Megatron alignment suite compares against a reference that is run without
gradient clipping, so ``Trainer.__init__`` forces the threshold off for that
target. ``max_grad_norm`` defaults to 1.0, which means a config that merely does
not mention it still clips: on PaddleFleet's ``GLM45Air_EP2`` alignment case the
real global norm is ~90, so every gradient gets rescaled by ~0.01, step 1 still
matches bit-for-bit and the comparison diverges from step 2 onwards.

The ``"hf"`` target is exempt because its reference *does* clip and
``Trainer._build_grad_clip()`` supplies the recipe that reproduces it.
"""

import shutil
import tempfile
import unittest

from paddleformers.trainer import Trainer, TrainingArguments
from tests.trainer.trainer_utils import RegressionModelConfig, RegressionPretrainedModel


class TestAccuracyTargetGradClip(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def _max_grad_norm(self, accuracy_target, **kwargs):
        config = RegressionModelConfig()
        config.use_accuracy_compatible = accuracy_target
        model = RegressionPretrainedModel(config)
        args = TrainingArguments(self.output_dir, report_to=[], bf16=True, **kwargs)
        return Trainer(model=model, args=args).args.max_grad_norm

    def test_megatron_target_disables_clipping(self):
        """The default 1.0 must not survive into a Megatron-aligned run."""
        self.assertEqual(self._max_grad_norm("megatron"), 0.0)

    def test_bare_true_disables_clipping(self):
        """``True`` is the historical spelling of the Megatron target."""
        self.assertEqual(self._max_grad_norm(True), 0.0)

    def test_explicit_threshold_is_still_overridden(self):
        """Alignment beats an explicit threshold; the warning says so."""
        self.assertEqual(self._max_grad_norm("megatron", max_grad_norm=5.0), 0.0)

    def test_already_off_is_left_alone(self):
        """``MinimaxV2.5_EP2.yaml`` spells this out; it must not error."""
        self.assertEqual(self._max_grad_norm("megatron", max_grad_norm=0.0), 0.0)

    def test_hf_target_keeps_clipping(self):
        """torch's reference clips, so zeroing this would drop the aligned step."""
        self.assertEqual(self._max_grad_norm("hf"), 1.0)
        self.assertEqual(self._max_grad_norm("hf", max_grad_norm=5.0), 5.0)

    def test_default_target_keeps_clipping(self):
        """A run that targets nothing keeps the user's threshold."""
        self.assertEqual(self._max_grad_norm(False), 1.0)

    def test_untouched_config_keeps_clipping(self):
        """``PretrainedConfig`` defaults the field to ``False``, i.e. off."""
        config = RegressionModelConfig()
        self.assertIs(config.use_accuracy_compatible, False)
        model = RegressionPretrainedModel(config)
        args = TrainingArguments(self.output_dir, report_to=[], bf16=True)
        self.assertEqual(Trainer(model=model, args=args).args.max_grad_norm, 1.0)


if __name__ == "__main__":
    unittest.main()
