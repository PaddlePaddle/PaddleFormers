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
from paddleformers.transformers.deepseek_ocr2 import (
    DeepseekOcr2Config,
    DeepseekOcr2ForConditionalGeneration,
    DeepseekOcr2Model,
)
from paddleformers.transformers.deepseek_ocr2.modeling import DeepseekOcr2TextTopkRouter


def native_config(tied=False, bias=False, grouped=False):
    return DeepseekOcr2Config(
        image_token_id=7,
        tie_word_embeddings=tied,
        text_config=dict(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            mlp_layer_types=["dense", "sparse"],
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            n_group=2,
            topk_group=1,
            topk_method="group_limited_greedy" if grouped else "greedy",
            fuse_rms_norm=False,
            attention_bias=bias,
            mlp_bias=bias,
            _attn_implementation="eager",
        ),
        vision_config=dict(
            sam_config=dict(
                hidden_size=16,
                output_channels=8,
                num_hidden_layers=1,
                num_attention_heads=4,
                image_size=32,
                patch_size=4,
                mlp_dim=32,
                downsample_channels=[16, 32],
                window_size=4,
                global_attn_indexes=[],
                _attn_implementation="eager",
            ),
            encoder_config=dict(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                fuse_rms_norm=False,
                _attn_implementation="eager",
            ),
        ),
    )


def inputs(local=False):
    ids = paddle.to_tensor([[1] + [7] * (289 if local else 145) + [2, 3]])
    result = dict(input_ids=ids, attention_mask=paddle.ones_like(ids), pixel_values=paddle.randn([1, 3, 32, 32]))
    if local:
        result.update(pixel_values_local=paddle.randn([1, 3, 24, 24]), num_local_patches=paddle.to_tensor([1]))
    return result


@pytest.fixture(autouse=True)
def seed():
    paddle.seed(26)


def test_native_auto_config_and_legacy_are_distinct():
    from paddleformers.transformers import DeepseekOCR2Config

    with tempfile.TemporaryDirectory() as directory:
        native_config().save_pretrained(directory)
        config = AutoConfig.from_pretrained(directory)
        assert isinstance(config, DeepseekOcr2Config)
        assert isinstance(AutoModel.from_config(config), DeepseekOcr2Model)
        assert isinstance(AutoModelForConditionalGeneration.from_config(config), DeepseekOcr2ForConditionalGeneration)
        DeepseekOCR2Config(hidden_size=32).save_pretrained(directory)
        assert isinstance(AutoConfig.from_pretrained(directory), DeepseekOCR2Config)


@pytest.mark.parametrize("tied,bias", [(False, True), (True, False)])
def test_native_hf_checkpoint_roundtrip(tied, bias):
    model = DeepseekOcr2ForConditionalGeneration(native_config(tied, bias)).eval()
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory, save_checkpoint_format="flex_checkpoint")
        loaded = AutoModelForConditionalGeneration.from_pretrained(directory, load_checkpoint_format="flex_checkpoint")
    for name, value in model.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), loaded.state_dict()[name].numpy())
    if tied:
        assert loaded.lm_head.weight is loaded.get_input_embeddings().weight


def test_base_model_roundtrip():
    model = DeepseekOcr2Model(native_config()).eval()
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory, save_checkpoint_format="flex_checkpoint")
        loaded = DeepseekOcr2Model.from_pretrained(directory, load_checkpoint_format="flex_checkpoint")
    for name, value in model.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), loaded.state_dict()[name].numpy())


def test_multimodal_gradients_are_not_disabled():
    model = DeepseekOcr2ForConditionalGeneration(native_config()).train()
    batch = inputs(local=True)
    labels = batch["input_ids"].clone()
    labels[:, 1:-2] = -100
    result = model(**batch, labels=labels, use_cache=False)
    assert bool(paddle.isfinite(result.loss))
    result.loss.backward()
    for parameter in (
        model.model.vision_tower.sam_encoder.patch_embed.proj.weight,
        model.model.vision_tower.vision_encoder.layers[0].self_attn.qkv_proj.weight,
        model.model.multi_modal_projector.weight,
        model.model.language_model.layers[1].mlp.experts.gate_up_proj,
    ):
        assert parameter.grad is not None
        assert bool(paddle.isfinite(parameter.grad).all())
        assert float(parameter.grad.abs().sum()) > 0


def test_multitoken_cache_and_inputs_embeds():
    model = DeepseekOcr2ForConditionalGeneration(native_config()).eval()
    batch = inputs(local=True)
    with paddle.no_grad():
        initial = model(**batch, use_cache=True)
        appended = paddle.to_tensor([[9, 10]])
        full = dict(batch, input_ids=paddle.concat([batch["input_ids"], appended], axis=1))
        full["attention_mask"] = paddle.ones_like(full["input_ids"])
        expected = model(**full, use_cache=False).logits[:, -2:]
        actual = model(
            input_ids=appended,
            attention_mask=full["attention_mask"],
            past_key_values=initial.past_key_values,
            use_cache=True,
        ).logits
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-5, rtol=1e-5)
        embeddings = model.get_input_embeddings()(batch.pop("input_ids"))
        actual = model(**batch, inputs_embeds=embeddings, use_cache=False).logits
        np.testing.assert_allclose(actual.numpy(), initial.logits.numpy(), atol=1e-5, rtol=1e-5)


def test_cached_generate_uses_global_and_local_vision_once_each():
    model = DeepseekOcr2ForConditionalGeneration(native_config()).eval()
    with paddle.no_grad():
        model.lm_head.weight[7] = paddle.zeros_like(model.lm_head.weight[7])
    batch = inputs(local=True)
    options = dict(max_new_tokens=3, min_new_tokens=3, eos_token_id=127, decode_strategy="greedy_search")
    with patch.object(model.model.vision_tower, "forward", wraps=model.model.vision_tower.forward) as vision:
        cached = model.generate(**batch, use_cache=True, **options)[0]
        assert vision.call_count == 2
        assert not bool((cached == 7).any())
    with patch.object(model.model.vision_tower, "forward", wraps=model.model.vision_tower.forward) as vision:
        uncached = model.generate(**batch, use_cache=False, **options)[0]
        assert vision.call_count == 6
    np.testing.assert_array_equal(cached.numpy(), uncached.numpy())


def test_validation_and_text_only():
    model = DeepseekOcr2ForConditionalGeneration(native_config()).eval()
    with pytest.raises(ValueError, match="do not match"):
        model(input_ids=paddle.to_tensor([[1, 7, 2]]), pixel_values=paddle.randn([1, 3, 32, 32]))
    with pytest.raises(ValueError, match="crop counts"):
        model(**dict(inputs(local=True), num_local_patches=[2]))
    output = model(input_ids=paddle.to_tensor([[1, 2, 3]]), use_cache=False, logits_to_keep=1)
    assert output.logits.shape == [1, 1, 128]


def test_grouped_router_does_not_renormalize_topk():
    config = native_config(grouped=True).text_config
    router = DeepseekOcr2TextTopkRouter(config)
    with paddle.no_grad():
        router.weight.set_value(paddle.arange(128, dtype="float32").reshape([4, 32]) / 1000)
    hidden = paddle.ones([2, 32])
    weights, indices = router(hidden)
    scores = paddle.nn.functional.softmax(paddle.matmul(hidden, router.weight, transpose_y=True), axis=-1)
    assert bool((indices // 2 == 1).all())
    np.testing.assert_allclose(paddle.sort(weights).numpy(), paddle.sort(scores[:, 2:]).numpy(), atol=1e-6)
    assert bool((weights.sum(-1) < 1).all())


def test_lora_gradients_and_image_feature_outputs():
    from paddleformers.peft import LoRAConfig, LoRAModel

    model = DeepseekOcr2ForConditionalGeneration(native_config())
    batch = inputs(local=True)
    with paddle.no_grad():
        features = model.get_image_features(
            batch["pixel_values"], pixel_values_local=batch["pixel_values_local"], num_local_patches=[1]
        )
        expected = model.model.vision_tower(batch["pixel_values"])
        np.testing.assert_array_equal(features.last_hidden_state.numpy(), expected.numpy())
        assert features.local_last_hidden_state.shape == [1, 144, 32]
        assert features.pooler_output[0].shape == [289, 32]
    model = LoRAModel(
        model,
        LoRAConfig(
            r=2,
            lora_alpha=4,
            target_modules=["model.language_model.layers.*.self_attn.q_proj", "model.multi_modal_projector"],
        ),
    )
    model.mark_only_lora_as_trainable()
    labels = batch["input_ids"].clone()
    labels[:, 1:-2] = -100
    model.train()
    model(**batch, labels=labels, use_cache=False).loss.backward()
    gradients = [parameter.grad for name, parameter in model.named_parameters() if "lora_B" in name]
    assert len(gradients) == 3
    for gradient in gradients:
        assert gradient is not None and bool(paddle.isfinite(gradient).all())
        assert float(gradient.abs().sum()) > 0
