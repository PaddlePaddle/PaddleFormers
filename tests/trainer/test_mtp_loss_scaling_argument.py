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

import unittest

from paddleformers.trainer import TrainingArguments
from paddleformers.trainer.argparser import PdArgumentParser


class TestMTPLossScalingArgument(unittest.TestCase):
    def test_explicit_weight_and_zero_are_parseable(self):
        parser = PdArgumentParser((TrainingArguments,))
        for value in ("0.1", "0.0"):
            with self.subTest(value=value):
                args = parser.parse_args(["--output_dir", "/tmp/mtp-argument", "--mtp_loss_scaling_factor", value])
                self.assertEqual(args.mtp_loss_scaling_factor, float(value))

    def test_omitted_weight_preserves_model_default(self):
        parser = PdArgumentParser((TrainingArguments,))
        args = parser.parse_args(["--output_dir", "/tmp/mtp-argument"])
        self.assertIsNone(args.mtp_loss_scaling_factor)


if __name__ == "__main__":
    unittest.main()
