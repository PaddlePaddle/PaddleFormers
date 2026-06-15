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
import unittest
from collections import namedtuple

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import paddle

from paddleformers.fleet.models.multimodal import llava_model
from paddleformers.fleet.models.multimodal.llava_model import LLaVAModel, pixel_shuffle


class MinimalLLaVA:
    pass


class RecorderLayer:
    def __init__(self, value="shared"):
        self.value = value
        self.inputs = []

    def shared_embedding_or_output_weight(self):
        return self.value

    def set_input_tensor(self, value):
        self.inputs.append(value)


class Param:
    def __init__(self):
        self.stop_gradient = False


class ParamModule:
    def __init__(self):
        self.param = Param()

    def parameters(self):
        return [self.param]


class TestLLaVAModelUtilities(unittest.TestCase):
    def test_shared_embedding_or_output_weight_respects_decoder_flag(self):
        model = MinimalLLaVA()
        model.add_decoder = True
        model.language_model = RecorderLayer("weight")

        self.assertEqual(LLaVAModel.shared_embedding_or_output_weight(model), "weight")

        model.add_decoder = False
        self.assertIsNone(LLaVAModel.shared_embedding_or_output_weight(model))

    def test_set_input_tensor_routes_to_active_chunk(self):
        marker = paddle.ones([1], dtype="float32")
        model = MinimalLLaVA()
        model.vision_model = RecorderLayer()
        model.language_model = RecorderLayer()
        model.add_encoder = True
        model.add_decoder = True
        model.pre_process = False

        LLaVAModel.set_input_tensor(model, marker)
        self.assertIs(model.vision_model.inputs[-1], marker)

        model.add_decoder = False
        LLaVAModel.set_input_tensor(model, [marker])
        self.assertIs(model.vision_model.inputs[-1], marker)

        model.add_encoder = False
        model.pre_process = True
        LLaVAModel.set_input_tensor(model, marker)
        self.assertIs(model.encoder_hidden_state, marker)

        model.pre_process = False
        LLaVAModel.set_input_tensor(model, marker)
        self.assertIs(model.language_model.inputs[-1], marker)

        with self.assertRaises(AssertionError):
            LLaVAModel.set_input_tensor(model, [marker, marker])

    def test_freeze_selected_modules_sets_stop_gradient(self):
        model = MinimalLLaVA()
        model.language_model = ParamModule()
        model.vision_model = ParamModule()
        model.vision_projection = ParamModule()

        LLaVAModel.freeze(
            model,
            freeze_language_model=True,
            freeze_vision_model=False,
            freeze_vision_projection=True,
        )

        self.assertTrue(model.language_model.param.stop_gradient)
        self.assertFalse(model.vision_model.param.stop_gradient)
        self.assertTrue(model.vision_projection.param.stop_gradient)

    def test_load_state_dict_hooks_remove_expected_keys_only(self):
        incompatible_type = namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])
        incompatible = incompatible_type(
            missing_keys=["vision_projection.weight", "language.weight"],
            unexpected_keys=["decoder.extra_state", "decoder.weight"],
        )

        llava_model._load_state_dict_hook_ignore_param_names(["vision_projection.weight"], None, incompatible)
        llava_model._load_state_dict_hook_ignore_extra_state(None, incompatible)

        self.assertEqual(incompatible.missing_keys, ["language.weight"])
        self.assertEqual(incompatible.unexpected_keys, ["decoder.weight"])

    def test_pixel_shuffle_versions_keep_shape_and_reorder_differently(self):
        x = paddle.arange(16, dtype="float32").reshape([1, 4, 4])

        version_one = pixel_shuffle(x, scale_factor=0.5, version=1)
        version_two = pixel_shuffle(x, scale_factor=0.5, version=2)

        self.assertEqual(version_one.shape, [1, 1, 16])
        self.assertEqual(version_two.shape, [1, 1, 16])
        self.assertEqual(
            version_one.numpy().tolist()[0][0],
            [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
            ],
        )
        self.assertEqual(version_two.numpy().tolist(), version_one.numpy().tolist())


if __name__ == "__main__":
    unittest.main()
