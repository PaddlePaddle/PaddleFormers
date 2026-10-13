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
from types import MethodType

from paddleformers.transformers.gpt_provider import _copy_model_attributes


class SourceModel:
    def forward(self):
        return self.marker


class TargetModel(SourceModel):
    pass


class TestGPTProviderMethodBinding(unittest.TestCase):
    def test_forward_uses_converted_instance_state(self):
        source = SourceModel()
        source.marker = "original"
        target = TargetModel()
        _copy_model_attributes(source, target)
        target.marker = "wrapped"
        self.assertIs(target.forward.__self__, target)
        self.assertEqual(target.forward(), "wrapped")
        self.assertEqual(source.forward(), "original")

    def test_instance_methods_rebound_but_foreign_methods_preserved(self):
        source = SourceModel()
        source.marker = "original"
        foreign = SourceModel()
        foreign.marker = "foreign"
        source.local_hook = MethodType(lambda model: model.marker, source)
        source.foreign_hook = foreign.forward
        target = TargetModel()
        _copy_model_attributes(source, target)
        target.marker = "wrapped"
        self.assertEqual(target.local_hook(), "wrapped")
        self.assertIs(target.local_hook.__self__, target)
        self.assertEqual(target.foreign_hook(), "foreign")
        self.assertIs(target.foreign_hook.__self__, foreign)


if __name__ == "__main__":
    unittest.main()
