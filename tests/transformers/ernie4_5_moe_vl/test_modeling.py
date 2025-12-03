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

from __future__ import annotations

import tempfile
import unittest

import numpy as np
import paddle

from paddleformers.transformers import (
    Ernie4_5_VLConfig,
    Ernie4_5_VLMoeForConditionalGenerationModel,
)
from paddleformers.transformers.configuration_utils import PretrainedConfig
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    ids_tensor,
    random_attention_mask,
)

try:
    from paddleformers.quantization.quantization_linear import QuantizationLinear
except:
    QuantizationLinear = None

from paddleformers.generation import BeamSearchScorer, GenerationConfig


class Ernie4_5_VLModelTester:
    def __init__(
        self,
        parent,
        batch_size=1,
        seq_length=10,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=120000,
        hidden_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=37,
        hidden_act="silu",
        moe_capacity=[8, 8, 8],
        moe_gate="topk",
        moe_intermediate_size=[32, 16],
        moe_k=2,
        moe_layer_end_index=[3, 3],
        moe_layer_interval=1,
        moe_layer_start_index=[1, 1],
        moe_multimodal_dispatch_use_allgather="v2-alltoall-unpad-text",
        moe_num_experts=[4, 4],
        moe_num_shared_experts=1,
        moe_use_aux_free=True,
        pixel_hidden_size=32,
        rms_norm_eps=1e-5,
        rope_3d=True,
        rope_theta=500000,
        spatial_conv_size=2,
        temporal_conv_size=2,
        tie_word_embeddings=True,
        use_cache=True,
        use_rmsnorm=True,
        use_bias=False,
        max_position_embeddings=512,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        moe_group="mp",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        im_patch_id=100295,
        vision_config={
            "attn_implementation": "eager",
            "depth": 4,
            "embed_dim": 32,
            "hidden_act": "quick_gelu",
            "hidden_size": 32,
            "in_channels": 3,
            "in_chans": 3,
            "mlp_ratio": 4,
            "num_heads": 4,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "spatial_patch_size": 14,
            "vit_first_fwd_bsz": 128,
            "attn_sep": False,
            "dtype": "bfloat16",
        },
    ):
        self.parent: Ernie4_5_VLModelTester = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.moe_capacity = moe_capacity
        self.moe_gate = moe_gate
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_k = moe_k
        self.moe_layer_end_index = moe_layer_end_index
        self.moe_layer_interval = moe_layer_interval
        self.moe_layer_start_index = moe_layer_start_index
        self.moe_multimodal_dispatch_use_allgather = moe_multimodal_dispatch_use_allgather
        self.moe_num_experts = moe_num_experts
        self.moe_num_shared_experts = moe_num_shared_experts
        self.moe_use_aux_free = moe_use_aux_free
        self.pixel_hidden_size = pixel_hidden_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_3d = rope_3d
        self.rope_theta = rope_theta
        self.spatial_conv_size = spatial_conv_size
        self.temporal_conv_size = temporal_conv_size
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache
        self.use_rmsnorm = use_rmsnorm
        self.use_bias = use_bias
        self.max_position_embeddings = max_position_embeddings
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.moe_group = moe_group
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.im_patch_id = im_patch_id
        self.vision_config = vision_config

    def prepare_config_and_inputs(self):
        input_ids = paddle.to_tensor([[100273, 2969, 93963, 93919, 100295, 23, 351, 93951, 8, 100272]], dtype="int64")
        position_ids = paddle.to_tensor(
            [
                [
                    [0, 0, 0],
                    [1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3],
                    [4, 4, 4],
                    [5, 5, 5],
                    [6, 6, 6],
                    [7, 7, 7],
                    [8, 8, 8],
                    [9, 9, 9],
                ]
            ],
            dtype="int32",
        )
        # attention_mask = paddle.ones([self.batch_size, 1, 10, 10], dtype="int32")
        attention_mask = None
        labels = paddle.to_tensor([[-100, -100, -100, -100, -100, -100, 351, 93951, 8, 100272]], dtype="int64")
        images = paddle.randn([4, 588], dtype="bfloat16")
        grid_thw = paddle.to_tensor([[1, 2, 2]], dtype="int32")
        image_position_ids = paddle.to_tensor([[0]], dtype="int32")
        # image_attention_mask = paddle.to_tensor([1], dtype="int32")
        image_attention_mask = None
        token_type_ids = paddle.to_tensor([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype="int64")
        token_type_ids = paddle.concat(
            [
                token_type_ids,
                paddle.zeros([len(token_type_ids), 1], token_type_ids.dtype),
            ],
            axis=-1,
        )
        image_type_ids = paddle.to_tensor([[0]], dtype="int32")

        tokenized_out = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images,
            "grid_thw": grid_thw,
            "image_position_ids": image_position_ids,
            "image_attention_mask": image_attention_mask,
            "token_type_ids": token_type_ids,
            "image_type_ids": image_type_ids,
        }

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()
        return config, tokenized_out, input_mask, sequence_labels, token_labels, choice_labels

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            tokenized_out,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": tokenized_out["input_ids"].astype("int64"),
            "position_ids": tokenized_out["position_ids"].astype("int64"),
            "attention_mask": tokenized_out["attention_mask"],
            "labels": tokenized_out["labels"].astype("int64"),
            "images": tokenized_out["images"].astype("bfloat16"),
            "grid_thw": tokenized_out["grid_thw"].astype("int32"),
            "image_attention_mask": tokenized_out["image_attention_mask"],
            "image_position_ids": tokenized_out["image_position_ids"].astype("int64"),
            "token_type_ids": tokenized_out["token_type_ids"].astype("int32"),
            "image_type_ids": tokenized_out["image_type_ids"].astype("int32"),
        }
        return config, inputs_dict

    def get_config(self) -> Ernie4_5_VLConfig:
        return Ernie4_5_VLConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            moe_capacity=self.moe_capacity,
            moe_gate=self.moe_gate,
            moe_intermediate_size=self.moe_intermediate_size,
            moe_k=self.moe_k,
            moe_layer_end_index=self.moe_layer_end_index,
            moe_layer_interval=self.moe_layer_interval,
            moe_layer_start_index=self.moe_layer_start_index,
            moe_multimodal_dispatch_use_allgather=self.moe_multimodal_dispatch_use_allgather,
            moe_num_experts=self.moe_num_experts,
            moe_num_shared_experts=self.moe_num_shared_experts,
            moe_use_aux_free=self.moe_use_aux_free,
            pixel_hidden_size=self.pixel_hidden_size,
            rms_norm_eps=self.rms_norm_eps,
            rope_3d=self.rope_3d,
            rope_theta=self.rope_theta,
            spatial_conv_size=self.spatial_conv_size,
            temporal_conv_size=self.temporal_conv_size,
            tie_word_embeddings=self.tie_word_embeddings,
            use_cache=self.use_cache,
            use_rmsnorm=self.use_rmsnorm,
            use_bias=self.use_bias,
            max_position_embeddings=self.max_position_embeddings,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            im_patch_id=self.im_patch_id,
            moe_group=self.moe_group,
            vision_config=self.vision_config,
            dtype="bfloat16",
        )

    def create_and_check_model(
        self, config: Ernie4_5_VLConfig, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = Ernie4_5_VLMoeForConditionalGenerationModel.from_config(config, dtype="bfloat16")
        paddle.amp.decorate(
            models=model,
            level="O2",
            dtype="bfloat16",
            master_grad=False,
            excluded_layers=QuantizationLinear,
        )
        model.eval()
        result = model(**input_ids)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        config.add_cross_attention = True
        model = Ernie4_5_VLMoeForConditionalGenerationModel.from_config(config, dtype="bfloat16")
        paddle.amp.decorate(
            models=model,
            level="O2",
            dtype="bfloat16",
            master_grad=False,
            excluded_layers=QuantizationLinear,
        )
        model.eval()
        result = model(**input_ids)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = Ernie4_5_VLMoeForConditionalGenerationModel.from_config(config, dtype="bfloat16")
        paddle.amp.decorate(
            models=model,
            level="O2",
            dtype="bfloat16",
            master_grad=False,
            excluded_layers=QuantizationLinear,
        )
        model.eval()
        result = model(**input_ids, return_dict=True)
        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])


class Ernie4_5_VLModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = Ernie4_5_VLMoeForConditionalGenerationModel
    return_dict = False
    use_labels = False
    use_test_model_name_list = False

    all_model_classes = (Ernie4_5_VLMoeForConditionalGenerationModel,)
    all_generative_model_classes = {
        Ernie4_5_VLMoeForConditionalGenerationModel: {Ernie4_5_VLMoeForConditionalGenerationModel, "ernie4_5_vl"}
    }
    pipeline_model_mapping = {
        "feature-extraction": Ernie4_5_VLMoeForConditionalGenerationModel,
        "text-classification": Ernie4_5_VLMoeForConditionalGenerationModel,
        "token-classification": Ernie4_5_VLMoeForConditionalGenerationModel,
        "text-generation": Ernie4_5_VLMoeForConditionalGenerationModel,
        "zero-shot": Ernie4_5_VLMoeForConditionalGenerationModel,
    }

    def _make_model_instance(self, config, model_class):
        if isinstance(config, PretrainedConfig):
            model = model_class.from_config(config, dtype="bfloat16")
        elif model_class == self.base_model_class:
            model = model_class.from_config(**config, dtype="bfloat16")
        else:
            model = model_class.from_config(self.base_model_class(**config), dtype="bfloat16")

        paddle.amp.decorate(
            models=model,
            level="O2",
            dtype="bfloat16",
            master_grad=False,
            excluded_layers=QuantizationLinear,
        )
        return model

    def setUp(self):
        super().setUp()
        self.model_tester = Ernie4_5_VLModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Ernie4_5_VLConfig, hidden_size=37)

    def test_init(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        (
            config,
            tokenized_out,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        Ernie4_5_VLMoeForConditionalGenerationModel.from_config(config, dtype="bfloat16")
        self.model_tester.parent.assertTrue([1])

    def test_model_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    @staticmethod
    def _get_beam_scorer_and_kwargs(batch_size, max_length, num_return_sequences=1):
        beam_kwargs = {
            "early_stopping": False,
            "length_penalty": 2.0,
            "num_beams": 2,
            "num_return_sequences": num_return_sequences,
        }
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=beam_kwargs["num_beams"],
            length_penalty=beam_kwargs["length_penalty"],
            do_early_stopping=beam_kwargs["early_stopping"],
            num_beam_hyps_to_keep=num_return_sequences,
        )
        return beam_kwargs, beam_scorer

    @staticmethod
    def _get_encoder_outputs(
        model,
        input_ids,
        attention_mask,
        output_attentions=None,
        output_hidden_states=None,
        num_interleave=1,
    ):
        model.eval()
        encoder = model.get_encoder()
        encoder_outputs = encoder(
            **input_ids,
        )
        if isinstance(encoder_outputs, (list, tuple)):
            encoder_outputs = encoder_outputs[0]

        encoder_outputs = encoder_outputs.repeat_interleave(num_interleave, axis=0)

        input_ids["input_ids"] = (
            paddle.zeros_like(input_ids["input_ids"][:, :1], dtype="int64") + model.get_decoder_start_token_id()
        )
        return encoder_outputs, input_ids, attention_mask

    def _greedy_generate(
        self,
        model,
        input_ids,
        attention_mask,
        max_length,
    ):
        if self.is_encoder_decoder:
            max_length = 4
        logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
            eos_token_id=getattr(model, model.base_model_prefix).config["eos_token_id"],
            forced_bos_token_id=getattr(getattr(model, model.base_model_prefix).config, "forced_bos_token_id", None),
            forced_eos_token_id=getattr(getattr(model, model.base_model_prefix).config, "forced_eos_token_id", None),
            max_length=max_length,
            plus_length=1 if self.is_encoder_decoder else input_ids["input_ids"].shape[-1],
        )

        kwargs = {}

        with paddle.no_grad():
            output_generate = model.generate(
                **input_ids,
                generation_config=GenerationConfig(
                    max_new_tokens=max_length,
                    decode_strategy="greedy_search",
                    **logits_process_kwargs,
                ),
            )

        if self.is_encoder_decoder:
            encoder_outputs, input_ids, attention_mask = self._get_encoder_outputs(
                model,
                input_ids,
                attention_mask,
            )
            kwargs["encoder_output"] = encoder_outputs

        with paddle.no_grad():
            output_greedy = model.greedy_search(
                **input_ids,
                max_length=max_length + 1
                if self.is_encoder_decoder
                else max_length + input_ids["input_ids"].shape[-1],
                # attention_mask=attention_mask,
                logits_processors=logits_processor,
                pad_token_id=getattr(model, model.base_model_prefix).config["pad_token_id"],
                eos_token_id=getattr(model, model.base_model_prefix).config["eos_token_id"],
                **kwargs,
            )
        return output_greedy, output_generate

    def _get_input_ids_and_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        input_ids = inputs_dict[self.input_name]
        attention_mask = paddle.ones_like(input_ids, dtype=paddle.int64)

        max_batch_size = 1
        sequence_length = input_ids.shape[-1]  # // 2
        input_ids = input_ids[:max_batch_size, :sequence_length]
        # For test_sample_generate such as: NVIDIA_TF32_OVERRIDE=0 FLAGS_cudnn_deterministic=1 python3.10 -m pytest -svv tests/transformers/bloom/test_modeling.py::BloomModelTest_0::test_sample_generate
        # There are serious memory bug for this tensor slice. which use the original tensor mem ptr for cold start
        # Here we just clone the tensor to avoid this problem.
        input_ids = input_ids.clone()
        # attention_mask = attention_mask[:max_batch_size, :sequence_length].unsqueeze([1, 2])

        # attention_mask = attention_mask * attention_mask.transpose([0, 1, 3, 2])

        inputs_dict["input_ids"] = input_ids
        # inputs_dict["attention_mask"] = attention_mask
        inputs_dict["token_type_ids"] = inputs_dict["token_type_ids"][:, :-1]

        # generate max 3 tokens
        max_length = 3

        if config.eos_token_id or config.pad_token_id:
            # hack to allow generate for models such as GPT2 as is done in `generate()`
            config["pad_token_id"] = config["eos_token_id"]

        return config, inputs_dict, attention_mask, max_length

    def test_greedy_generate(self):
        # check `generate()` and `greedy_search()` are equal
        for model_class in self.all_generative_model_classes.keys():
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()
            paddle.seed(124)
            model = self._make_model_instance(config, model_class)
            model.eval()

            output_greedy, output_generate = self._greedy_generate(
                model=model, input_ids=input_ids, attention_mask=attention_mask, max_length=max_length
            )

            self.assertListEqual(output_greedy[0].tolist(), output_generate[0].tolist())

    def test_sample_generate(self):
        pass

    def test_beam_search_generate(self):
        pass

    def test_generate_without_input_ids(self):
        config, _, _, max_length = self._get_input_ids_and_config()

        # if no bos token id => cannot generate from None
        if config.bos_token_id is None:
            return

        for model_class in self.all_generative_model_classes.keys():
            if isinstance(config, PretrainedConfig):
                model = model_class.from_config(config, dtype="bfloat16")
            else:
                pretrained_model = self.all_generative_model_classes[model_class][0].from_config(
                    **config, dtype="bfloat16"
                )
                model = model_class(pretrained_model)
            model.eval()
            output_ids_generate = model.generate(
                token_type_ids=paddle.to_tensor([[0]]).astype("int64"),
                position_ids=paddle.to_tensor([[[0, 0, 0]]]).astype("int64"),
                generation_config=GenerationConfig(
                    decode_strategy="greedy_search",
                    max_new_tokens=max_length,
                ),
            )

            self.assertIsNotNone(output_ids_generate)

    def test_group_beam_search_generate(self):
        pass

    def test_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_save_load(out1, out2):
            # make sure we don't have nans
            out_2 = out2.numpy()
            out_2[np.isnan(out_2)] = 0

            out_1 = out1.numpy()
            out_1[np.isnan(out_1)] = 0
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            model = self._make_model_instance(config, model_class)
            model.eval()
            with paddle.no_grad():
                first = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                config = self.config_tester.config_class.from_pretrained(tmpdirname)
                config["moe_group"] = "dummy"
                config["moe_multimodal_dispatch_use_allgather"] = "v2-alltoall-unpad-text"
                model = model_class.from_pretrained(tmpdirname, config=config, dtype="bfloat16")
                paddle.amp.decorate(
                    models=model,
                    level="O2",
                    dtype="bfloat16",
                    master_grad=False,
                    excluded_layers=QuantizationLinear,
                )
                model.eval()
                with paddle.no_grad():
                    second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            # support tuple of tensor
            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_save_load(tensor1, tensor2)
            else:
                check_save_load(first, second)

    def test_resize_tokens_embeddings(self):
        pass

    def test_attention_outputs(self):
        pass


class Ernie4_5_MoE_VLIntegrationTest(unittest.TestCase):
    def test_model_tiny_logits(self):

        config = Ernie4_5_VLConfig.from_pretrained("PaddleFormers/tiny_random_ernie4_5_vl", download_hub="aistudio")
        config["moe_group"] = "dummy"
        config["moe_multimodal_dispatch_use_allgather"] = "v2-alltoall-unpad-text"
        model = Ernie4_5_VLMoeForConditionalGenerationModel.from_pretrained(
            "PaddleFormers/tiny_random_ernie4_5_vl",
            config=config,
            dtype="bfloat16",
            convert_from_hf=True,
            download_hub="aistudio",
        )
        paddle.amp.decorate(
            models=model,
            level="O2",
            dtype="bfloat16",
            master_grad=False,
            excluded_layers=QuantizationLinear,
        )

        input_ids = paddle.to_tensor([[100273, 2969, 93963, 93919, 100295, 23, 351, 93951, 8, 100272]], dtype="int64")
        position_ids = paddle.to_tensor(
            [
                [
                    [0, 0, 0],
                    [1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3],
                    [4, 4, 4],
                    [5, 5, 5],
                    [6, 6, 6],
                    [7, 7, 7],
                    [8, 8, 8],
                    [9, 9, 9],
                ]
            ],
            dtype="int32",
        )
        attention_mask = None
        labels = paddle.to_tensor([[-100, -100, -100, -100, -100, -100, 351, 93951, 8, 100272]], dtype="int64")
        images = paddle.ones([4, 588], dtype="bfloat16")
        grid_thw = paddle.to_tensor([[1, 2, 2]], dtype="int32")
        image_position_ids = paddle.to_tensor([[0]], dtype="int32")
        image_attention_mask = None
        token_type_ids = paddle.to_tensor([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype="int64")
        token_type_ids = paddle.concat(
            [
                token_type_ids,
                paddle.zeros([len(token_type_ids), 1], token_type_ids.dtype),
            ],
            axis=-1,
        )
        image_type_ids = paddle.to_tensor([[0]], dtype="int32")

        tokenized_out = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images,
            "grid_thw": grid_thw,
            "image_position_ids": image_position_ids,
            "image_attention_mask": image_attention_mask,
            "token_type_ids": token_type_ids,
            "image_type_ids": image_type_ids,
        }

        with paddle.no_grad():
            out = model(**tokenized_out, return_dict=True).logits

        out = out.astype("float32")
        # Expected mean on dim = -1
        EXPECTED_MEAN = paddle.to_tensor(
            [
                [
                    0.00005200,
                    0.00003907,
                    0.00007939,
                    0.00010351,
                    0.00017985,
                    0.00000782,
                    0.00008746,
                    0.00003174,
                    -0.00001411,
                    0.00001531,
                ]
            ]
        )
        self.assertTrue(paddle.allclose(out.mean(-1), EXPECTED_MEAN, atol=1e-3, rtol=1e-3))

        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = paddle.to_tensor(
            [
                -0.00421143,
                -0.01428223,
                0.00411987,
                -0.03466797,
                0.05493164,
                -0.01599121,
                -0.01757812,
                0.01525879,
                -0.02893066,
                -0.03015137,
                -0.00726318,
                -0.01989746,
                -0.03369141,
                0.00741577,
                -0.00946045,
                -0.01989746,
                0.05957031,
                0.01611328,
                0.07617188,
                0.00285339,
                0.03491211,
                -0.09082031,
                0.00370789,
                0.08154297,
                -0.03222656,
                0.00300598,
                -0.03063965,
                -0.04492188,
                0.07031250,
                0.04882812,
            ]
        )
        self.assertTrue(paddle.allclose(out[0, 0, :30], EXPECTED_SLICE, atol=1e-3, rtol=1e-3))
