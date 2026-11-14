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

"""Checkpoint rejection must precede parameter mutation."""
import unittest
from unittest.mock import patch

import numpy as np
import paddle

from tests.cinn.checkpoint_utils import strict_set_state_dict


class StrictCheckpointTest(unittest.TestCase):
    def test_valid_checkpoint(self):
        model = paddle.nn.Linear(2, 3)
        state = {name: paddle.zeros_like(value) for name, value in model.state_dict().items()}
        strict_set_state_dict(model, state)
        for value in model.state_dict().values():
            np.testing.assert_array_equal(value.numpy(), np.zeros(value.shape))

    def test_invalid_checkpoint_never_reaches_loader(self):
        model = paddle.nn.Linear(2, 3)
        original = {name: value.numpy().copy() for name, value in model.state_dict().items()}
        invalid = [
            {"weight": original["weight"]},
            {**original, "unknown": np.zeros([1])},
            {**original, "weight": np.zeros([3, 2])},
            {**original, "weight": None},
            None,
        ]
        for state in invalid:
            with self.subTest(state=state), patch.object(model, "set_state_dict") as loader:
                with self.assertRaisesRegex(ValueError, "Checkpoint"):
                    strict_set_state_dict(model, state)
                loader.assert_not_called()
            for name, value in model.state_dict().items():
                np.testing.assert_array_equal(value.numpy(), original[name])

    def test_loader_report_is_checked(self):
        model = paddle.nn.Linear(2, 3)
        with patch.object(model, "set_state_dict", return_value=(["weight"], [])):
            with self.assertRaisesRegex(ValueError, "skipped parameters"):
                strict_set_state_dict(model, model.state_dict())
