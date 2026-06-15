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
import types
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import paddle

from paddleformers.fleet.models.multimodal import llava_model
from paddleformers.fleet.models.multimodal.llava_model import LLaVAModel


class Config:
    def __init__(self, model_type="clip"):
        self.language_model_type = ""
        self.sequence_parallel = False
        self.tp_comm_overlap = False
        self.context_parallel_size = 1
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.vision_model_type = model_type
        self.hidden_size = 2


class PGCollection:
    tp = None


class RecordingGPT:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        RecordingGPT.calls.append(kwargs)

    def shared_embedding_or_output_weight(self):
        return "shared"


class RecordingClip:
    calls = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.class_token_len = kwargs.get("class_token_len", 1)
        RecordingClip.calls.append(kwargs)

    def parameters(self):
        return []

    def set_input_tensor(self, value):
        self.input_tensor = value


class RecordingRadio(RecordingClip):
    calls = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        RecordingRadio.calls.append(kwargs)


class RecordingProjector:
    calls = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        RecordingProjector.calls.append((args, kwargs))

    def state_dict(self):
        return {"weight": object(), "bias": object()}

    def parameters(self):
        return []


class MinimalLLaVA:
    def _process_embedding_token_parallel(self, *args, **kwargs):
        return LLaVAModel._process_embedding_token_parallel(self, *args, **kwargs)

    def _preprocess_data(self, *args, **kwargs):
        return LLaVAModel._preprocess_data(self, *args, **kwargs)

    def _apply_tile_tagging(self, *args, **kwargs):
        return LLaVAModel._apply_tile_tagging(self, *args, **kwargs)


class RecordingLanguageModel:
    def __init__(self):
        self.forward_inputs = []
        self.embedding_inputs = []

    def embedding(self, input_ids, position_ids=None):
        self.embedding_inputs.append((input_ids, position_ids))
        values = paddle.cast(input_ids, "float32").unsqueeze(-1)
        values = paddle.concat([values, values + 0.5], axis=-1)
        return values.transpose([1, 0, 2]).contiguous()

    def __call__(self, **kwargs):
        self.forward_inputs.append(kwargs)
        return kwargs


class RecordingVisionModel:
    def __init__(self, output):
        self.output = output
        self.class_token_len = 0
        self.inputs = []

    def __call__(self, images):
        self.inputs.append(images)
        return self.output


class RecordingProjection:
    def __call__(self, embeddings):
        return embeddings[:, :, :2] + 10.0


class TestLLaVAInitAndExtraBranchesNoMock(unittest.TestCase):
    def setUp(self):
        self.old_gpt = llava_model.GPTModel
        self.old_clip = llava_model.CLIPViTModel
        self.old_radio = llava_model.RADIOViTModel
        self.old_projector = llava_model.MultimodalProjector
        self.old_pg = llava_model.ProcessGroupCollection
        self.old_hf = llava_model.has_config_logger_enabled
        self.old_log_config = llava_model.log_config_to_disk
        self.old_num_embeddings = llava_model.get_num_image_embeddings
        RecordingGPT.calls = []
        RecordingClip.calls = []
        RecordingRadio.calls = []
        RecordingProjector.calls = []
        llava_model.GPTModel = RecordingGPT
        llava_model.CLIPViTModel = RecordingClip
        llava_model.RADIOViTModel = RecordingRadio
        llava_model.MultimodalProjector = RecordingProjector
        llava_model.ProcessGroupCollection.use_mpu_process_groups = lambda: PGCollection()
        llava_model.has_config_logger_enabled = lambda config: True
        self.logged = []
        llava_model.log_config_to_disk = lambda *args, **kwargs: self.logged.append((args, kwargs))
        llava_model.get_num_image_embeddings = lambda *args, **kwargs: 4

    def tearDown(self):
        llava_model.GPTModel = self.old_gpt
        llava_model.CLIPViTModel = self.old_clip
        llava_model.RADIOViTModel = self.old_radio
        llava_model.MultimodalProjector = self.old_projector
        llava_model.ProcessGroupCollection = self.old_pg
        llava_model.has_config_logger_enabled = self.old_hf
        llava_model.log_config_to_disk = self.old_log_config
        llava_model.get_num_image_embeddings = self.old_num_embeddings

    def _build_model(self, vision_type="clip", **kwargs):
        return LLaVAModel(
            Config(),
            object(),
            8,
            16,
            Config(vision_type),
            object(),
            False,
            Config(),
            object(),
            pg_collection=PGCollection(),
            **kwargs,
        )

    def test_init_clip_radio_siglip_and_parallel_assertion_paths(self):
        model = self._build_model("siglip")
        self.assertEqual(model.img_seq_len, 4)
        self.assertEqual(RecordingClip.calls[-1]["class_token_len"], 0)
        self.assertTrue(self.logged)

        radio = self._build_model("radio-g", pixel_shuffle=True)
        self.assertEqual(RecordingRadio.calls[-1]["class_token_len"], 5)
        self.assertTrue(RecordingRadio.calls[-1]["embedder_bias"])
        self.assertEqual(RecordingProjector.calls[-1][0][3], 8)
        self.assertEqual(radio.shared_embedding_or_output_weight(), "shared")

        radio_plain = self._build_model("radio")
        self.assertEqual(RecordingRadio.calls[-1]["class_token_len"], 8)
        self.assertFalse(RecordingRadio.calls[-1]["embedder_bias"])
        self.assertIs(radio_plain.vision_model.kwargs["ln_post_impl"], None)
        self.assertFalse(RecordingRadio.calls[-1]["use_mask_token"])

        cradio = self._build_model("cradio-g", allow_missing_vision_projection_checkpoint=True)
        self.assertEqual(RecordingRadio.calls[-1]["class_token_len"], 8)
        self.assertFalse(RecordingRadio.calls[-1]["embedder_bias"])
        self.assertIs(cradio.vision_model.kwargs["ln_post_impl"], None)
        self.assertFalse(RecordingRadio.calls[-1]["use_mask_token"])

        bad_config = Config()
        bad_config.sequence_parallel = True
        with self.assertRaises(AssertionError):
            LLaVAModel(
                bad_config,
                object(),
                8,
                16,
                Config("clip"),
                object(),
                False,
                Config(),
                object(),
                pg_collection=PGCollection(),
            )

    def test_process_embedding_context_parallel_branches(self):
        model = MinimalLLaVA()
        model.pre_process = True
        model.post_process = True
        model.sequence_parallel_lm = True
        model.tensor_model_parallel_size_lm = 2
        model.context_parallel_lm = 2
        model.tp_comm_overlap_lm = False
        embeddings = paddle.ones([2, 8, 2], dtype="float32")

        result = model._process_embedding_token_parallel(embeddings, None, None, None)
        self.assertIs(result[0], embeddings)

        model.sequence_parallel_lm = False
        result = model._process_embedding_token_parallel(embeddings, None, None, None)
        self.assertIs(result[0], embeddings)

        model.sequence_parallel_lm = True
        model.context_parallel_lm = 1
        model.tp_comm_overlap_lm = True
        model._language_max_sequence_length = 16
        with self.assertRaises(AssertionError):
            model._process_embedding_token_parallel(paddle.ones([4, 2, 2], dtype="float32"), None, None, None)

    def test_hf_language_and_vision_model_branches(self):
        calls = []
        module = types.ModuleType("paddleformers.fleet.models.huggingface.module")

        def build_hf_model(config, model_type=None):
            calls.append((config, model_type))
            return RecordingGPT()

        module.build_hf_model = build_hf_model
        old_module = sys.modules.get("paddleformers.fleet.models.huggingface.module")
        had_module = "paddleformers.fleet.models.huggingface.module" in sys.modules
        try:
            sys.modules["paddleformers.fleet.models.huggingface.module"] = module
            language_config = Config()
            language_config.language_model_type = "hf://tiny-lm"
            LLaVAModel(
                language_config,
                object(),
                8,
                16,
                Config("hf://tiny-vision"),
                object(),
                False,
                Config(),
                object(),
                pg_collection=PGCollection(),
            )
        finally:
            if had_module:
                sys.modules["paddleformers.fleet.models.huggingface.module"] = old_module
            else:
                sys.modules.pop("paddleformers.fleet.models.huggingface.module", None)

        self.assertEqual(calls[0][1], "hf://tiny-lm")
        self.assertIsNone(calls[1][1])
        self.assertEqual(calls[2][1], "hf://tiny-vision")

    def test_forward_default_tiles_cache_reuse_and_parallel_processing(self):
        model = MinimalLLaVA()
        model.add_encoder = True
        model.add_decoder = True
        model.pre_process = True
        model.post_process = True
        model._drop_vision_class_token = False
        model._pixel_shuffle = False
        model._tile_tags = None
        model.image_token_index = -200
        model.img_seq_len = 1
        model._language_is_pipeline_parallel = False
        model._language_max_sequence_length = 4
        model.context_parallel_lm = 1
        model.sequence_parallel_lm = True
        model.tensor_model_parallel_size_lm = 1
        model.tp_comm_overlap_lm = False
        model.language_model = RecordingLanguageModel()
        model.vision_model = RecordingVisionModel(paddle.arange(2, dtype="float32").reshape([1, 1, 2]))
        model.vision_projection = RecordingProjection()
        context = type("Context", (), {"key_value_memory_dict": {}})()

        output, loss_mask = LLaVAModel.forward(
            model,
            images=paddle.ones([1, 3, 2, 2], dtype="float32"),
            input_ids=paddle.to_tensor([[5, -200, 6]], dtype="int64"),
            position_ids=paddle.to_tensor([[0, 1, 2]], dtype="int64"),
            attention_mask=None,
            labels=paddle.to_tensor([[50, 51, 52]], dtype="int64"),
            loss_mask=paddle.ones([1, 3], dtype="float32"),
            inference_context=context,
            num_image_tiles=None,
        )

        self.assertIs(output, model.language_model.forward_inputs[0])
        self.assertEqual(loss_mask.shape, [1, 3])
        self.assertIn("image_tokens_count", context.key_value_memory_dict)

        cached_output, _ = LLaVAModel.forward(
            model,
            images=paddle.ones([1, 3, 2, 2], dtype="float32"),
            input_ids=paddle.to_tensor([[7, 8]], dtype="int64"),
            position_ids=paddle.to_tensor([[0, 1]], dtype="int64"),
            attention_mask=None,
            labels=paddle.to_tensor([[70, 71]], dtype="int64"),
            loss_mask=paddle.ones([1, 2], dtype="float32"),
            inference_context=context,
        )

        self.assertIs(cached_output, model.language_model.forward_inputs[-1])
        self.assertEqual(len(model.vision_model.inputs), 1)

    def test_forward_no_images_pixel_shuffle_tile_tags_and_cache_count(self):
        model = MinimalLLaVA()
        model.add_encoder = True
        model.add_decoder = True
        model.pre_process = True
        model.post_process = True
        model._drop_vision_class_token = False
        model._pixel_shuffle = True
        model._tile_tags = [[101, 102, 103, 104], [999, 1000, 1001, 1002]]
        model.image_token_index = -200
        model.img_seq_len = 8
        model._language_is_pipeline_parallel = False
        model._language_max_sequence_length = 8
        model.context_parallel_lm = 1
        model.sequence_parallel_lm = False
        model.language_model = RecordingLanguageModel()
        model.vision_model = RecordingVisionModel(paddle.arange(64, dtype="float32").reshape([1, 16, 4]))
        model.vision_projection = RecordingProjection()
        context = type("Context", (), {"key_value_memory_dict": {}})()

        output, loss_mask = LLaVAModel.forward(
            model,
            images=paddle.ones([1, 3, 4, 4], dtype="float32"),
            input_ids=paddle.to_tensor([[5, -200, 6]], dtype="int64"),
            position_ids=paddle.to_tensor([[0, 1, 2]], dtype="int64"),
            attention_mask=None,
            labels=paddle.to_tensor([[50, 51, 52]], dtype="int64"),
            loss_mask=paddle.ones([1, 3], dtype="float32"),
            inference_context=context,
            num_image_tiles=paddle.to_tensor([1], dtype="int64"),
        )

        self.assertIn("image_tokens_count", context.key_value_memory_dict)
        self.assertEqual(len(model.language_model.forward_inputs), 1)
        self.assertIs(output, model.language_model.forward_inputs[0])
        self.assertEqual(loss_mask.shape[0], 1)

        model.add_decoder = False
        empty_output, _ = LLaVAModel.forward(
            model,
            images=paddle.empty([0, 3, 4, 4], dtype="float32"),
            input_ids=paddle.to_tensor([[7, 8]], dtype="int64"),
            position_ids=paddle.to_tensor([[0, 1]], dtype="int64"),
            attention_mask=None,
            labels=None,
            loss_mask=None,
        )
        self.assertEqual(empty_output.shape, [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
