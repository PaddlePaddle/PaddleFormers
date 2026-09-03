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
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import paddle

from paddleformers.datasets.collate import (
    _pack_molmo_multimodal_inputs,
    _pad_and_stack_multimodal_tensors,
    _pad_and_stack_optional_multimodal_tensors,
    mm_collate_fn,
)
from paddleformers.datasets.SFTDataset import Sequence
from paddleformers.transformers import AutoModel, AutoModelForCausalLM
from paddleformers.transformers.molmo import MolmoConfig, MolmoForCausalLM, MolmoModel
from paddleformers.transformers.molmo.modeling import MolmoPretrainedVisionBackbone
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    ids_tensor,
    random_attention_mask,
)


class MolmoModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=7,
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        max_position_embeddings=32,
        num_images=1,
        image_num_patches=4,
        image_patch_pixels=12,
        use_input_mask=False,
        use_labels=False,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.num_images = num_images
        self.image_num_patches = image_num_patches
        self.image_patch_pixels = image_patch_pixels
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()
        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self, with_vision=False) -> MolmoConfig:
        vision_backbone = None
        vit_layers = None
        image_padding_embed = None
        if with_vision:
            vision_backbone = {
                "image_default_input_size": (4, 4),
                "image_patch_size": 2,
                "image_pos_patch_size": 2,
                "image_emb_dim": 8,
                "image_num_heads": 2,
                "image_num_key_value_heads": 2,
                "image_num_layers": 2,
                "image_head_dim": 4,
                "image_mlp_dim": 16,
                "image_mlp_activations": "quick_gelu",
                "image_dropout_rate": 0.0,
                "image_num_pos": 5,
                "image_norm_eps": 1e-5,
                "attention_dropout": 0.0,
                "residual_dropout": 0.0,
                "initializer_range": 0.02,
            }
            vit_layers = (-1,)
            image_padding_embed = "pad_and_partial_pad"

        return MolmoConfig(
            vocab_size=self.vocab_size,
            embedding_size=self.vocab_size,
            additional_vocab_size=0,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            use_cache=False,
            attention_layer_norm=True,
            norm_after=True,
            qkv_bias=False,
            vision_backbone=vision_backbone,
            vit_layers=vit_layers,
            image_padding_embed=image_padding_embed,
            image_pooling_h=1,
            image_pooling_w=1,
            image_pooling_2d="attention-meanq",
            image_projector="mlp",
            vision_attention_type="direct",
            _attn_implementation="eager",
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )

    def get_vision_inputs(self):
        images = paddle.randn(
            [self.batch_size, self.num_images, self.image_num_patches, self.image_patch_pixels],
            dtype=paddle.float32,
        )
        image_masks = paddle.ones(
            [self.batch_size, self.num_images, self.image_num_patches],
            dtype=paddle.float32,
        )
        image_input_idx = paddle.to_tensor(
            [[[1, 2, 3, 4]]] * self.batch_size,
            dtype=paddle.int64,
        )
        return images, image_masks, image_input_idx

    def create_and_check_model(
        self, config: MolmoConfig, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = MolmoModel(config)
        model.eval()

        result = model(input_ids)

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_model_attention_mask(
        self, config: MolmoConfig, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = MolmoModel(config)
        model.eval()

        attn_mask_2d = random_attention_mask([self.batch_size, self.seq_length])
        result_2d = model(input_ids, attention_mask=attn_mask_2d)[0]

        batch, seq_length = input_ids.shape
        causal_mask = paddle.tril(paddle.ones((batch, seq_length, seq_length), dtype=attn_mask_2d.dtype))
        attn_mask_3d = causal_mask & attn_mask_2d.unsqueeze(-1)
        result_3d = model(input_ids, attention_mask=attn_mask_3d)[0]

        attn_mask_4d = attn_mask_3d.unsqueeze(1)
        result_4d = model(input_ids, attention_mask=attn_mask_4d)[0]
        result_no_attention_mask = model(input_ids, attention_mask=None)[0]

        self.parent.assertTrue((result_2d[attn_mask_2d] == result_3d[attn_mask_2d]).all())
        self.parent.assertTrue((result_2d[attn_mask_2d] == result_4d[attn_mask_2d]).all())
        self.parent.assertTrue((result_2d[attn_mask_2d] == result_no_attention_mask[attn_mask_2d]).all())

    def check_model_position_ids(self, config, input_ids, input_mask, *args):
        model = MolmoForCausalLM(config)
        model.eval()

        result_no_position_id = model(
            input_ids,
            labels=input_ids if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
        )
        batch_size, seq_len = input_ids.shape
        position_ids = paddle.arange(seq_len).expand((batch_size, seq_len))
        result_position_id = model(
            input_ids,
            position_ids=position_ids,
            labels=input_ids if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
        )

        if self.parent.use_labels:
            self.parent.assertTrue((result_position_id[1] == result_no_position_id[1]).all())
        else:
            self.parent.assertTrue((result_position_id[0] == result_no_position_id[0]).all())

    def create_and_check_for_causal_lm(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = MolmoForCausalLM(config)
        model.eval()

        result = model(input_ids, attention_mask=input_mask, labels=token_labels, return_dict=True)

        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])
        if self.use_labels:
            self.parent.assertIsInstance(result.loss.item(), float)

    def create_and_check_lm_head_model(self, config, input_ids, input_mask, *args):
        model = MolmoForCausalLM(config)
        model.eval()

        result = model(input_ids, attention_mask=input_mask, use_cache=True, return_dict=False)

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask, sequence_labels, token_labels, choice_labels = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict

    def create_and_check_vision_backbone(self):
        config = self.get_config(with_vision=True)
        images, image_masks, _ = self.get_vision_inputs()
        model = MolmoPretrainedVisionBackbone(config)
        model.eval()

        image_features, cls_embed = model(images, image_masks)

        self.parent.assertEqual(image_features.shape, [self.batch_size, self.num_images, 4, self.hidden_size])
        self.parent.assertEqual(cls_embed.shape, [self.batch_size, self.num_images, 8])

    def create_and_check_multimodal_causal_lm(self):
        config = self.get_config(with_vision=True)
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)
        images, image_masks, image_input_idx = self.get_vision_inputs()
        model = MolmoForCausalLM(config)
        model.eval()

        multimodal_outputs = model(
            input_ids,
            images=images,
            image_masks=image_masks,
            image_input_idx=image_input_idx,
            return_dict=True,
        )
        text_outputs = model(input_ids, return_dict=True)

        self.parent.assertEqual(multimodal_outputs.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])
        self.parent.assertFalse(paddle.allclose(multimodal_outputs.logits, text_outputs.logits))


class MolmoModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = MolmoModel
    return_dict = False
    use_labels = False
    use_test_model_name_list = False
    test_resize_embeddings = False
    test_model_compatibility_keys = False
    has_attentions = False

    all_model_classes = (MolmoModel, MolmoForCausalLM)

    def setUp(self):
        super().setUp()
        paddle.seed(1234)
        self.model_tester = MolmoModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MolmoConfig, vocab_size=64, hidden_size=16)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_config_save_load(self):
        config = self.model_tester.get_config(with_vision=True)

        with tempfile.TemporaryDirectory() as tmpdirname:
            config.save_pretrained(tmpdirname)
            loaded_config = MolmoConfig.from_pretrained(tmpdirname)
            self.assertTrue((Path(tmpdirname) / "config_molmo.py").is_file())

        self.assertEqual(loaded_config.model_type, "molmo")
        self.assertEqual(loaded_config.auto_map["AutoConfig"], "config_molmo.MolmoConfig")
        self.assertEqual(loaded_config.hidden_size, config.hidden_size)
        self.assertEqual(loaded_config.vocab_size, config.vocab_size)
        self.assertEqual(loaded_config.vision_backbone["image_emb_dim"], config.vision_backbone["image_emb_dim"])
        self.assertEqual(loaded_config.llm_patches_per_crop(), config.llm_patches_per_crop())

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_attention_mask(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_attention_mask(*config_and_inputs)

    def test_model_position_ids(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_model_position_ids(*config_and_inputs)

    def test_molmo_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_for_causal_lm_uses_config_output_hidden_states(self):
        config = self.model_tester.get_config(with_vision=True)
        input_ids = ids_tensor(
            [self.model_tester.batch_size, self.model_tester.seq_length], config.vocab_size, dtype=paddle.int64
        )
        config.output_hidden_states = True
        model = MolmoForCausalLM(config)
        model.eval()

        with paddle.no_grad():
            result = model(input_ids, return_dict=True)

        self.assertEqual(len(result.hidden_states), config.num_hidden_layers + 1)

    def test_auto_model_for_causal_lm_from_config(self):
        config = self.model_tester.get_config()
        config.architectures = ["MolmoForCausalLM"]
        model = AutoModelForCausalLM.from_config(config)
        self.assertIsInstance(model, MolmoForCausalLM)

    def test_auto_model_from_config(self):
        config = self.model_tester.get_config()
        model = AutoModel.from_config(config)
        self.assertIsInstance(model, MolmoModel)

    def test_variable_image_crops_are_padded(self):
        first = paddle.ones([1, 4, 3], dtype="float32")
        second = paddle.ones([2, 4, 3], dtype="float32")
        images = _pad_and_stack_multimodal_tensors([first, second])

        first_idx = paddle.ones([1, 4], dtype="int64")
        second_idx = paddle.ones([2, 4], dtype="int64")
        image_input_idx = _pad_and_stack_multimodal_tensors(
            [first_idx, second_idx],
            padding_value=-1,
        )

        self.assertEqual(images.shape, [2, 2, 4, 3])
        self.assertTrue((images[0, 1] == 0).all())
        self.assertTrue((image_input_idx[0, 1] == -1).all())

    def test_packed_multimodal_inputs_use_token_offsets(self):
        first = Sequence(
            token_ids=[1, 2, 3],
            position_ids=[0, 1, 2],
            labels=[1, 2, 3],
            num_examples=1,
            mm_inputs={
                "images": paddle.ones([1, 2, 3]),
                "image_masks": paddle.ones([1, 2]),
                "image_input_idx": paddle.to_tensor([[1, -100]], dtype="int64"),
            },
        )
        second = Sequence(
            token_ids=[4, 5],
            position_ids=[0, 1],
            labels=[4, 5],
            num_examples=1,
            mm_inputs={
                "images": paddle.ones([1, 2, 3]),
                "image_masks": paddle.ones([1, 2]),
                "image_input_idx": paddle.to_tensor([[0, 1]], dtype="int64"),
            },
        )

        images, image_masks, image_input_idx = _pack_molmo_multimodal_inputs([first, second])

        self.assertEqual(images.shape, [2, 2, 3])
        self.assertEqual(image_masks.shape, [2, 2])
        self.assertTrue(paddle.equal_all(image_input_idx, paddle.to_tensor([[1, -100], [3, 4]])))

    def test_mm_collate_packs_molmo_inputs(self):
        first = Sequence(
            token_ids=[1, 2, 3],
            position_ids=[0, 1, 2],
            labels=[1, 2, 3],
            num_examples=1,
            mm_inputs={
                "images": paddle.ones([1, 2, 3]),
                "image_masks": paddle.ones([1, 2]),
                "image_input_idx": paddle.to_tensor([[1, -100]], dtype="int64"),
            },
        )
        second = Sequence(
            token_ids=[4, 5],
            position_ids=[0, 1],
            labels=[4, 5],
            num_examples=1,
            mm_inputs={
                "images": paddle.ones([1, 2, 3]),
                "image_masks": paddle.ones([1, 2]),
                "image_input_idx": paddle.to_tensor([[0, 1]], dtype="int64"),
            },
        )
        training_args = SimpleNamespace(
            num_nextn_predict_layers=0,
            context_parallel_size=1,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            fp8=False,
        )
        model_args = SimpleNamespace(
            use_attn_mask_startend_row_indices=False,
            use_global_causal_attn=False,
        )
        tokenizer = SimpleNamespace(pad_token_id=0)

        for padding_free, batch in ((False, [[first, second]]), (True, [[first], [second]])):
            with self.subTest(padding_free=padding_free):
                inputs = mm_collate_fn(
                    batch=batch,
                    template=None,
                    processor=None,
                    tokenizer=tokenizer,
                    training_args=training_args,
                    model_args=model_args,
                    max_seq_len=5,
                    padding_free=padding_free,
                    model=None,
                )

                self.assertEqual(inputs["input_ids"].shape, [1, 5])
                self.assertEqual(inputs["images"].shape, [1, 2, 2, 3])
                self.assertTrue(
                    paddle.equal_all(
                        inputs["image_input_idx"],
                        paddle.to_tensor([[[1, -100], [3, 4]]]),
                    )
                )

    def test_packed_multimodal_rows_are_padded(self):
        first = paddle.ones([1, 2, 3])
        second = paddle.ones([2, 2, 3])

        images = _pad_and_stack_optional_multimodal_tensors([first, None, second])
        image_input_idx = _pad_and_stack_optional_multimodal_tensors(
            [paddle.ones([1, 2], dtype="int64"), None, paddle.ones([2, 2], dtype="int64")],
            padding_value=-1,
        )

        self.assertEqual(images.shape, [3, 2, 2, 3])
        self.assertTrue((images[1] == 0).all())
        self.assertTrue((image_input_idx[0, 1] == -1).all())
        self.assertTrue((image_input_idx[1] == -1).all())

    def test_multimodal_generate_with_cache(self):
        config = self.model_tester.get_config(with_vision=True)
        config.eos_token_id = None
        input_ids = ids_tensor(
            [self.model_tester.batch_size, self.model_tester.seq_length], config.vocab_size, dtype=paddle.int64
        )
        images, image_masks, image_input_idx = self.model_tester.get_vision_inputs()
        model = MolmoForCausalLM(config)
        model.eval()

        output = model.generate(
            input_ids=input_ids,
            images=images,
            image_masks=image_masks,
            image_input_idx=image_input_idx,
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
        )

        self.assertEqual(output[0].shape, [self.model_tester.batch_size, 2])

    def test_inverse_aoa_config_contains_vision_weights(self):
        config = self.model_tester.get_config(with_vision=True)
        statements = MolmoForCausalLM._gen_inv_aoa_config(config)["aoa_statements"]
        self.assertIn(
            "model.vision_backbone.image_vit.patch_embedding.weight^T -> "
            "model.vision_backbone.image_vit.patch_embedding.weight",
            statements,
        )

    def test_vision_backbone(self):
        self.model_tester.create_and_check_vision_backbone()

    def test_multimodal_causal_lm(self):
        self.model_tester.create_and_check_multimodal_causal_lm()


if __name__ == "__main__":
    unittest.main()
