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

import numpy as np
import paddle
import pytest

from paddleformers.nn.mlp import MLP
from paddleformers.transformers.lfm2.modeling import (
    Lfm2MLP,
    convert_lfm2_legacy_state_dict,
)
from paddleformers.transformers.lfm2_vl.modeling import (
    Lfm2VlForConditionalGeneration,
    Siglip2Attention,
    Siglip2EncoderLayer,
    Siglip2MLP,
)
from paddleformers.transformers.siglip_vision_model.modeling import (
    SiglipAttention,
    SiglipEncoderLayer,
    SiglipMLP,
)
from tests.cinn.checkpoint_utils import strict_set_state_dict
from tests.transformers.lfm2_vl.test_modeling import get_config, hf_reference_weights


def test_common_forward_methods_are_reused_without_parameter_aliases():
    assert Lfm2MLP.forward is MLP.forward
    assert Siglip2Attention.forward is SiglipAttention.forward
    assert Siglip2MLP.forward is SiglipMLP.forward
    assert Siglip2EncoderLayer.forward is SiglipEncoderLayer.forward
    model = Lfm2VlForConditionalGeneration(get_config())
    keys = list(model.state_dict())
    assert not any(".feed_forward.w1." in name for name in keys)
    assert len([key for key in keys if ".feed_forward.gate_proj.weight" in key]) == 3


def test_siglip_padding_mask_is_preserved_with_common_attention():
    config = get_config().vision_config
    config._attn_implementation = "flashmask"
    layer = Siglip2Attention(config).eval()
    assert layer.config._attn_implementation == "sdpa"
    assert config._attn_implementation == "flashmask"
    hidden = paddle.randn([1, 4, config.hidden_size])
    mask = paddle.to_tensor([[[[0.0, 0.0, -1e9, -1e9]]]])
    expected = layer(hidden, mask)[0]
    changed = hidden.clone()
    changed[:, 2:] += 100
    actual = layer(changed, mask)[0]
    np.testing.assert_allclose(actual[:, :2].numpy(), expected[:, :2].numpy(), atol=1e-5, rtol=1e-5)


def test_reused_siglip_layer_to_static_preserves_padding():
    config = get_config().vision_config
    config._attn_implementation = "eager"
    layer = Siglip2EncoderLayer(config).eval()
    hidden = paddle.randn([1, 4, config.hidden_size])
    mask = paddle.to_tensor([[[[0.0, 0.0, -1e9, -1e9]]]])
    expected = layer(hidden, mask)[0]
    compiled = paddle.jit.to_static(layer, full_graph=True)
    actual = compiled(hidden, mask)[0]
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-5, rtol=1e-5)


def test_legacy_cinn_checkpoint_migration_preserves_every_value():
    config = get_config()
    config.text_config.conv_bias = True
    reference = Lfm2VlForConditionalGeneration(config).eval()
    hf = hf_reference_weights(reference, tied=False)
    legacy = {}
    for name, array in hf.items():
        is_linear = name.endswith(".weight") and name.split(".")[-2] in (
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "in_proj",
            "w1",
            "w2",
            "w3",
            "fc1",
            "fc2",
            "linear_1",
            "linear_2",
            "patch_embedding",
        )
        legacy[name] = paddle.to_tensor(array.T.copy() if is_linear else array)
    converted = convert_lfm2_legacy_state_dict(legacy)
    assert set(converted) == set(reference.state_dict())
    actual = Lfm2VlForConditionalGeneration(config).eval()
    strict_set_state_dict(actual, converted)
    for key, value in reference.state_dict().items():
        np.testing.assert_array_equal(actual.state_dict()[key].numpy(), value.numpy(), err_msg=key)
    np.testing.assert_array_equal(
        legacy["model.language_model.layers.0.conv.in_proj.weight"].numpy(),
        paddle.concat(
            [converted[f"model.language_model.layers.0.conv.{gate}_proj.weight"] for gate in ("b", "c", "x")], axis=1
        ).numpy(),
    )
    conflicting = dict(
        legacy,
        **{
            "model.language_model.layers.0.feed_forward.gate_proj.weight": converted[
                "model.language_model.layers.0.feed_forward.gate_proj.weight"
            ]
        },
    )
    with pytest.raises(ValueError, match="conflicting"):
        convert_lfm2_legacy_state_dict(conflicting)


def test_lora_reaches_common_mlp_and_shortconv_projections():
    from paddleformers.peft import LoRAConfig, LoRAModel

    model = LoRAModel(
        Lfm2VlForConditionalGeneration(get_config()),
        LoRAConfig(
            r=2,
            lora_alpha=4,
            target_modules=[
                "model.language_model.layers.*.feed_forward.down_proj",
                "model.language_model.layers.*.conv.b_proj",
            ],
        ),
    )
    model.mark_only_lora_as_trainable()
    model.train()
    ids = paddle.to_tensor([[1, 2, 3, 4]])
    model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
    gradients = [parameter.grad for name, parameter in model.named_parameters() if "lora_B" in name]
    assert len(gradients) == 5
    for gradient in gradients:
        assert gradient is not None and bool(paddle.isfinite(gradient).all())
        assert float(gradient.abs().sum()) > 0
