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

REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import paddle

from paddleformers.fleet.models.multimodal import llava_model
from paddleformers.fleet.models.multimodal.llava_model import LLaVAModel, pixel_shuffle


class MinimalLLaVA:
    def _preprocess_data(self, *args, **kwargs):
        return LLaVAModel._preprocess_data(self, *args, **kwargs)

    def _process_embedding_token_parallel(self, *args, **kwargs):
        return LLaVAModel._process_embedding_token_parallel(
            self, *args, **kwargs
        )

    def _apply_tile_tagging(self, *args, **kwargs):
        return LLaVAModel._apply_tile_tagging(self, *args, **kwargs)


class InferenceContext:
    def __init__(self, image_tokens_count=None):
        self.key_value_memory_dict = {}
        if image_tokens_count is not None:
            self.key_value_memory_dict["image_tokens_count"] = (
                image_tokens_count
            )


class RecordingVisionModel:
    def __init__(self, output, class_token_len=0):
        self.output = output
        self.class_token_len = class_token_len
        self.inputs = []
        self._is_fsdp_managed_module = False

    def __call__(self, images):
        self.inputs.append(images)
        return self.output


class RecordingProjection:
    def __init__(self, offset=0.0):
        self.offset = offset
        self.inputs = []

    def __call__(self, embeddings):
        self.inputs.append(embeddings)
        return embeddings + self.offset


class RecordingLanguageModel:
    def __init__(self):
        self.embedding_inputs = []
        self.forward_inputs = []

    def embedding(self, input_ids, position_ids=None):
        self.embedding_inputs.append((input_ids, position_ids))
        values = paddle.cast(input_ids, "float32").unsqueeze(-1)
        values = paddle.concat([values, values + 0.5], axis=-1)
        return values.transpose([1, 0, 2]).contiguous()

    def __call__(self, **kwargs):
        self.forward_inputs.append(kwargs)
        return kwargs


def make_preprocess_model():
    model = MinimalLLaVA()
    model.add_decoder = True
    model.pre_process = True
    model.post_process = True
    model.img_seq_len = 2
    model._language_is_pipeline_parallel = False
    model._language_max_sequence_length = 16
    model.context_parallel_lm = 1
    model.vision_model = MinimalLLaVA()
    return model


class TestLLaVAPreprocessNoMock(unittest.TestCase):
    def test_preprocess_middle_chunk_and_inference_cache_return_early(self):
        model = MinimalLLaVA()
        model.add_decoder = True
        model.pre_process = False
        model.post_process = False
        result = LLaVAModel._preprocess_data(
            model,
            None,
            None,
            paddle.to_tensor([[1, -200, 2]], dtype="int64"),
            None,
            None,
            False,
            None,
            -200,
            paddle.to_tensor([1], dtype="int64"),
        )
        self.assertEqual(result, (None, None, None))

        model.pre_process = True
        model.post_process = True
        language_embeddings = paddle.ones([1, 3, 2], dtype="float32")
        loss_mask = paddle.ones([1, 3], dtype="float32")
        labels = paddle.to_tensor([[10, 11, 12]], dtype="int64")
        result = LLaVAModel._preprocess_data(
            model,
            None,
            language_embeddings,
            paddle.to_tensor([[1, 2, 3]], dtype="int64"),
            loss_mask,
            labels,
            True,
            None,
            -200,
            paddle.to_tensor([], dtype="int64"),
        )
        self.assertIs(result[0], language_embeddings)
        self.assertIs(result[1], loss_mask)
        self.assertIs(result[2], labels)

    def test_preprocess_expands_image_token_and_truncates_outputs(self):
        model = make_preprocess_model()
        model._language_max_sequence_length = 3
        input_ids = paddle.to_tensor([[5, -200, 6]], dtype="int64")
        language_embeddings = paddle.to_tensor(
            [[[5.0, 5.5], [0.0, 0.5], [6.0, 6.5]]], dtype="float32"
        )
        image_embeddings = paddle.to_tensor(
            [[[20.0, 20.5]], [[21.0, 21.5]]], dtype="float32"
        )
        labels = paddle.to_tensor([[50, 51, 52]], dtype="int64")
        loss_mask = paddle.ones([1, 3], dtype="float32")

        final_embedding, final_labels, final_loss_mask = (
            LLaVAModel._preprocess_data(
                model,
                image_embeddings,
                language_embeddings,
                input_ids,
                loss_mask,
                labels,
                False,
                None,
                -200,
                paddle.to_tensor([1], dtype="int64"),
            )
        )

        self.assertEqual(final_embedding.shape, [3, 1, 2])
        self.assertEqual(final_labels.shape, [1, 3])
        self.assertEqual(final_loss_mask.shape, [1, 3])
        self.assertEqual(
            final_embedding[:, 0, :].numpy().tolist(),
            [[5.0, 5.5], [20.0, 20.5], [21.0, 21.5]],
        )
        self.assertEqual(final_loss_mask.numpy().tolist(), [[0.0, 0.0, 0.0]])

    def test_preprocess_pipeline_padding_and_dummy_fsdp_image(self):
        model = make_preprocess_model()
        model._language_is_pipeline_parallel = True
        model._language_max_sequence_length = 5
        model.vision_model._is_fsdp_managed_module = True
        input_ids = paddle.to_tensor([[7, 8]], dtype="int64")
        language_embeddings = paddle.to_tensor(
            [[[7.0, 7.5], [8.0, 8.5]]], dtype="float32"
        )
        image_embeddings = paddle.ones([1, 1, 2], dtype="float32")

        final_embedding, final_labels, final_loss_mask = (
            LLaVAModel._preprocess_data(
                model,
                image_embeddings,
                language_embeddings,
                input_ids,
                paddle.ones([1, 2], dtype="float32"),
                None,
                False,
                None,
                -200,
                paddle.to_tensor([], dtype="int64"),
            )
        )

        self.assertEqual(final_embedding.shape, [5, 1, 2])
        self.assertIsNone(final_labels)
        self.assertIsNone(final_loss_mask)

    def test_preprocess_rejects_mismatched_labels_and_loss_mask(self):
        model = make_preprocess_model()
        with self.assertRaises(AssertionError):
            LLaVAModel._preprocess_data(
                model,
                paddle.ones([2, 1, 2], dtype="float32"),
                paddle.ones([1, 3, 2], dtype="float32"),
                paddle.to_tensor([[5, -200, 6]], dtype="int64"),
                paddle.ones([1, 3], dtype="float32"),
                paddle.ones([1, 2], dtype="int64"),
                False,
                None,
                -200,
                paddle.to_tensor([1], dtype="int64"),
            )


class TestLLaVATokenParallelAndTaggingNoMock(unittest.TestCase):
    def test_process_embedding_token_parallel_middle_and_assertions(self):
        model = MinimalLLaVA()
        model.pre_process = False
        model.post_process = False
        embeddings = paddle.ones([4, 1, 2], dtype="float32")
        labels = paddle.ones([1, 4], dtype="int64")
        loss_mask = paddle.ones([1, 4], dtype="float32")
        packed = {"marker": True}
        result = LLaVAModel._process_embedding_token_parallel(
            model, embeddings, labels, loss_mask, packed
        )
        self.assertIs(result[0], embeddings)
        self.assertIs(result[3], packed)

        model.pre_process = True
        model.post_process = True
        model.context_parallel_lm = 1
        model.sequence_parallel_lm = True
        model.tensor_model_parallel_size_lm = 2
        model.tp_comm_overlap_lm = False
        with self.assertRaises(AssertionError):
            LLaVAModel._process_embedding_token_parallel(
                model, paddle.ones([3, 1, 2], dtype="float32"), None, None, None
            )
        result = LLaVAModel._process_embedding_token_parallel(
            model, paddle.ones([4, 1, 2], dtype="float32"), None, None, None
        )
        self.assertEqual(result[0].shape, [4, 1, 2])

        model.tp_comm_overlap_lm = True
        model._language_max_sequence_length = 6
        with self.assertRaises(AssertionError):
            LLaVAModel._process_embedding_token_parallel(
                model, paddle.ones([4, 1, 2], dtype="float32"), None, None, None
            )

    def test_apply_tile_tagging_prepends_tags_and_rejects_multiple_images(self):
        model = MinimalLLaVA()
        model._tile_tags = [[101, 102], [201, 202], [999, 1000]]
        model.language_model = RecordingLanguageModel()
        image_embeddings = paddle.to_tensor(
            [
                [[1.0, 1.5], [2.0, 2.5]],
                [[3.0, 3.5], [4.0, 4.5]],
                [[5.0, 5.5], [6.0, 6.5]],
            ],
            dtype="float32",
        )

        result = LLaVAModel._apply_tile_tagging(
            model, image_embeddings, paddle.to_tensor([2], dtype="int64")
        )

        self.assertEqual(result.shape, [5, 2, 2])
        self.assertTrue(paddle.allclose(result[2:], image_embeddings))
        with self.assertRaises(AssertionError):
            LLaVAModel._apply_tile_tagging(
                model, image_embeddings, paddle.to_tensor([1, 1], dtype="int64")
            )


class TestLLaVAForwardNoMock(unittest.TestCase):
    def test_forward_encoder_only_projects_image_embeddings(self):
        model = MinimalLLaVA()
        model.add_encoder = True
        model.add_decoder = False
        model._drop_vision_class_token = True
        model._pixel_shuffle = False
        model._tile_tags = None
        model.image_token_index = -200
        vision_output = paddle.to_tensor(
            [[[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]]], dtype="float32"
        )
        model.vision_model = RecordingVisionModel(
            vision_output, class_token_len=1
        )
        model.vision_projection = RecordingProjection(offset=10.0)
        loss_mask = paddle.ones([1, 3], dtype="float32")

        output, returned_loss_mask = LLaVAModel.forward(
            model,
            images=paddle.ones([1, 3, 4, 4], dtype="float32"),
            input_ids=paddle.to_tensor([[5, -200, 6]], dtype="int64"),
            position_ids=paddle.to_tensor([[0, 1, 2]], dtype="int64"),
            attention_mask=None,
            labels=None,
            loss_mask=loss_mask,
        )

        self.assertIs(returned_loss_mask, loss_mask)
        self.assertEqual(output.shape, [2, 1, 2])
        self.assertEqual(
            output[:, 0, :].numpy().tolist(), [[11.0, 11.5], [12.0, 12.5]]
        )

    def test_forward_decoder_uses_encoder_hidden_state_and_language_model(self):
        model = make_preprocess_model()
        model.add_encoder = False
        model.add_decoder = True
        model.sequence_parallel_lm = False
        model.encoder_hidden_state = paddle.to_tensor(
            [[[20.0, 20.5]], [[21.0, 21.5]]], dtype="float32"
        )
        model.image_token_index = -200
        model.language_model = RecordingLanguageModel()

        output, new_loss_mask = LLaVAModel.forward(
            model,
            images=None,
            input_ids=paddle.to_tensor([[5, -200, 6]], dtype="int64"),
            position_ids=paddle.to_tensor([[0, 1, 2]], dtype="int64"),
            attention_mask=None,
            labels=paddle.to_tensor([[50, 51, 52]], dtype="int64"),
            loss_mask=paddle.ones([1, 3], dtype="float32"),
            num_image_tiles=paddle.to_tensor([1], dtype="int64"),
        )

        self.assertEqual(new_loss_mask.shape, [1, 4])
        self.assertEqual(len(model.language_model.forward_inputs), 1)
        self.assertIs(output, model.language_model.forward_inputs[0])
        self.assertEqual(output["decoder_input"].shape, [4, 1, 2])
        self.assertEqual(output["labels"].shape, [4, 1])

    def test_forward_inference_cache_skips_vision_and_processes_language(self):
        model = make_preprocess_model()
        model.add_encoder = True
        model.add_decoder = True
        model.sequence_parallel_lm = False
        model.image_token_index = -200
        model.language_model = RecordingLanguageModel()
        model.vision_model = RecordingVisionModel(
            paddle.ones([1, 2, 2], dtype="float32")
        )
        context = InferenceContext(image_tokens_count=2)

        output, new_loss_mask = LLaVAModel.forward(
            model,
            images=paddle.ones([1, 3, 4, 4], dtype="float32"),
            input_ids=paddle.to_tensor([[5, 6, 7]], dtype="int64"),
            position_ids=paddle.to_tensor([[0, 1, 2]], dtype="int64"),
            attention_mask=None,
            labels=paddle.to_tensor([[50, 51, 52]], dtype="int64"),
            loss_mask=paddle.ones([1, 3], dtype="float32"),
            inference_context=context,
            num_image_tiles=paddle.to_tensor([], dtype="int64"),
        )

        self.assertEqual(model.vision_model.inputs, [])
        self.assertEqual(new_loss_mask.shape, [1, 3])
        self.assertEqual(output["decoder_input"].shape, [1, 3, 2])


class TestLLaVAExtraUtilityNoMock(unittest.TestCase):
    def test_pixel_shuffle_versions_differ_on_larger_grid(self):
        x = paddle.arange(64, dtype="float32").reshape([1, 16, 4])
        version_one = pixel_shuffle(x, scale_factor=0.5, version=1)
        version_two = pixel_shuffle(x, scale_factor=0.5, version=2)
        self.assertEqual(version_one.shape, [1, 4, 16])
        self.assertEqual(version_two.shape, [1, 4, 16])
        self.assertFalse(paddle.allclose(version_one, version_two))
        self.assertEqual(version_one.numel(), x.numel())

    def test_load_state_hooks_preserve_unmatched_keys(self):
        incompatible_type = namedtuple(
            "IncompatibleKeys", ["missing_keys", "unexpected_keys"]
        )
        incompatible = incompatible_type(
            missing_keys=["a.weight", "b.weight", "c.extra_state"],
            unexpected_keys=["unexpected.extra_state", "unexpected.weight"],
        )
        llava_model._load_state_dict_hook_ignore_param_names(
            ["b.weight", "not.present"], None, incompatible
        )
        llava_model._load_state_dict_hook_ignore_extra_state(None, incompatible)
        self.assertEqual(incompatible.missing_keys, ["a.weight"])
        self.assertEqual(incompatible.unexpected_keys, ["unexpected.weight"])


if __name__ == "__main__":
    unittest.main()
