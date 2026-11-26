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

import tempfile
from unittest.mock import patch

import numpy as np
import paddle
import pytest

from paddleformers.transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForConditionalGeneration,
)
from paddleformers.transformers.fast_vlm import (
    FastVlmConfig,
    FastVlmForConditionalGeneration,
    FastVlmModel,
)


@pytest.fixture(autouse=True)
def deterministic_seed():
    paddle.seed(26)


def config(tied=False, bias=True):
    return FastVlmConfig(
        text_config=dict(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            fuse_rms_norm=False,
            _attn_implementation="eager",
            tie_word_embeddings=tied,
        ),
        vision_config=dict(
            hidden_size=64,
            model_args=dict(inference_mode=True, layers=[1] * 5, embed_dims=[8, 16, 32, 32, 32], mlp_ratios=[2] * 5),
        ),
        image_token_id=7,
        image_seq_length=4,
        tie_word_embeddings=tied,
        multimodal_projector_bias=bias,
    )


def inputs():
    ids = paddle.to_tensor([[1, 7, 7, 7, 7, 2]])
    return dict(input_ids=ids, attention_mask=paddle.ones_like(ids), pixel_values=paddle.randn([1, 3, 128, 128]))


def test_native_config_and_auto_registration():
    c = config()
    assert c.model_type == "fast_vlm"
    assert c.image_token_id == c.image_token_index == 7
    with tempfile.TemporaryDirectory() as directory:
        c.save_pretrained(directory)
        loaded = AutoConfig.from_pretrained(directory)
    assert isinstance(loaded, FastVlmConfig)
    assert loaded.text_config.hidden_size == 32
    assert isinstance(AutoModel.from_config(loaded), FastVlmModel)
    assert isinstance(AutoModelForConditionalGeneration.from_config(loaded), FastVlmForConditionalGeneration)


@pytest.mark.parametrize("tied,bias", [(False, True), (True, False)])
def test_native_hf_format_roundtrip(tied, bias):
    model = FastVlmForConditionalGeneration(config(tied, bias)).eval()
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory, save_checkpoint_format="flex_checkpoint")
        loaded = FastVlmForConditionalGeneration.from_pretrained(directory, load_checkpoint_format="flex_checkpoint")
    assert set(loaded.state_dict()) == set(model.state_dict())
    for key, value in model.state_dict().items():
        np.testing.assert_array_equal(loaded.state_dict()[key].numpy(), value.numpy())
    if tied:
        assert loaded.lm_head.weight is loaded.get_input_embeddings().weight


def test_native_base_model_hf_format_roundtrip():
    from paddleformers.transformers.fastvlm.modeling import FastVLMModel
    from paddleformers.transformers.qwen2.modeling import Qwen2Model

    assert FastVLMModel.base_model_class is FastVLMModel
    assert Qwen2Model.base_model_class is Qwen2Model
    model = FastVlmModel(config()).eval()
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory, save_checkpoint_format="flex_checkpoint")
        loaded = FastVlmModel.from_pretrained(directory, load_checkpoint_format="flex_checkpoint")
    for key, value in model.state_dict().items():
        np.testing.assert_array_equal(loaded.state_dict()[key].numpy(), value.numpy())


def test_cached_generation_reuses_vision_and_matches_no_cache():
    model = FastVlmForConditionalGeneration(config()).eval()
    batch = inputs()
    # This is an untied synthetic LM. Avoid newly generated image placeholders
    # while testing cache reuse; placeholder count validation is tested below.
    with paddle.no_grad():
        model.lm_head.weight[7] = paddle.zeros_like(model.lm_head.weight[7])
    options = dict(max_new_tokens=4, min_new_tokens=4, eos_token_id=127, decode_strategy="greedy_search")
    with patch.object(model.model.vision_tower, "forward", wraps=model.model.vision_tower.forward) as vision:
        cached = model.generate(**batch, use_cache=True, **options)[0]
        assert vision.call_count == 1
        assert not bool((cached == 7).any())
    with patch.object(model.model.vision_tower, "forward", wraps=model.model.vision_tower.forward) as vision:
        uncached = model.generate(**batch, use_cache=False, **options)[0]
        assert vision.call_count == 4
    np.testing.assert_array_equal(cached.numpy(), uncached.numpy())


def test_cached_continuation_and_inputs_embeds():
    model = FastVlmForConditionalGeneration(config()).eval()
    batch = inputs()
    with paddle.no_grad():
        initial = model(**batch, use_cache=True)
        continuation = paddle.to_tensor([[8, 9]])
        full_ids = paddle.concat([batch["input_ids"], continuation], axis=1)
        mask = paddle.ones_like(full_ids)
        expected = model(
            input_ids=full_ids, attention_mask=mask, pixel_values=batch["pixel_values"], use_cache=False
        ).logits[:, -2:]
        actual = model(
            input_ids=continuation, attention_mask=mask, past_key_values=initial.past_key_values, use_cache=True
        ).logits
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-5, rtol=1e-5)
        embeds = model.get_input_embeddings()(batch.pop("input_ids"))
        actual = model(inputs_embeds=embeds, **batch, use_cache=False).logits
        np.testing.assert_allclose(actual.numpy(), initial.logits.numpy(), atol=1e-6, rtol=1e-6)


def test_image_count_validation_and_vision_gradient():
    model = FastVlmForConditionalGeneration(config()).eval()
    batch = inputs()
    with pytest.raises(ValueError, match="Image features and image tokens"):
        model(input_ids=paddle.to_tensor([[1, 7, 2]]), pixel_values=batch["pixel_values"], use_cache=False)
    labels = batch["input_ids"].clone()
    labels = paddle.where(labels == 7, paddle.full_like(labels, -100), labels)
    loss = model(**batch, labels=labels, use_cache=False).loss
    loss.backward()
    for weight in (
        model.model.vision_tower.timm_model.patch_embed[0].reparam_conv.weight,
        model.model.multi_modal_projector.linear_1.weight,
    ):
        assert weight.grad is not None
        assert bool(paddle.isfinite(weight.grad).all())
        assert float(paddle.abs(weight.grad).sum()) > 0


def test_text_only_and_logits_to_keep():
    model = FastVlmForConditionalGeneration(config()).eval()
    ids = paddle.to_tensor([[1, 2, 3]])
    with paddle.no_grad():
        full = model(input_ids=ids, use_cache=False)
        last = model(input_ids=ids, use_cache=False, logits_to_keep=1)
    assert full.image_hidden_states is None
    # Slicing before LMHead changes the GEMM row count, not the mathematical result.
    np.testing.assert_allclose(last.logits.numpy(), full.logits[:, -1:].numpy(), atol=1e-5, rtol=1e-5)


def test_native_image_feature_output_contract():
    model = FastVlmForConditionalGeneration(config()).eval()
    batch = inputs()
    with paddle.no_grad():
        features = model.get_image_features(batch["pixel_values"])
        output = model(**batch, use_cache=False)
    # Paddle ModelOutput normalizes Python lists to tuples.
    assert isinstance(features.pooler_output, (list, tuple))
    assert list(features.pooler_output[0].shape) == [4, 32]
    assert list(output.image_hidden_states.shape) == [4, 32]


def test_lora_backward_through_native_text_and_projector():
    from paddleformers.peft import LoRAConfig, LoRAModel

    model = FastVlmForConditionalGeneration(config())
    model = LoRAModel(
        model,
        LoRAConfig(
            r=2,
            lora_alpha=4,
            target_modules=[
                "model.language_model.layers.*.self_attn.qkv_proj",
                "model.multi_modal_projector.linear_1",
            ],
        ),
    )
    model.mark_only_lora_as_trainable()
    batch = inputs()
    labels = paddle.to_tensor([[7, 7, 7, 7, 2, -100]])
    labels = paddle.where(labels == 7, paddle.full_like(labels, -100), labels)
    model.train()
    model(**batch, labels=labels, use_cache=False).loss.backward()
    gradients = {name: param.grad for name, param in model.named_parameters() if "lora_B" in name}
    assert len(gradients) == 3
    for value in gradients.values():
        assert value is not None and bool(paddle.isfinite(value).all())
        assert float(value.abs().sum()) > 0
