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

from unittest.mock import patch

import numpy as np
import paddle
import pytest

from paddleformers.transformers.lfm2.modeling import (
    Lfm2ForCausalLM,
    Lfm2HybridConvCache,
)
from paddleformers.transformers.lfm2_vl.modeling import Lfm2VlForConditionalGeneration

from .test_modeling import get_config


@pytest.mark.parametrize("layer_types", [["conv", "full_attention", "conv"], ["conv"] * 3, ["full_attention"] * 3])
@pytest.mark.parametrize("kernel_size,bias,prefix_length", [(1, False, 1), (3, True, 1), (3, False, 4)])
def test_cached_continuation_matches_full_forward(layer_types, kernel_size, bias, prefix_length):
    config = get_config().text_config
    config.layer_types = layer_types
    config.conv_L_cache = kernel_size
    config.conv_bias = bias
    config._attn_implementation = "eager"
    model = Lfm2ForCausalLM(config).eval()
    ids = paddle.to_tensor([[0, 4, 5, 6, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17]])
    mask = (ids != 0).astype("int64")
    with paddle.no_grad():
        expected = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        prefill = model(input_ids=ids[:, :prefix_length], attention_mask=mask[:, :prefix_length], use_cache=True)
        cache = prefill.past_key_values
        assert isinstance(cache, Lfm2HybridConvCache)
        assert cache.get_seq_length() == prefix_length
        parts = [prefill.logits]
        # Include both single-token decoding and multi-token continuation.
        start = prefix_length
        for chunk in (1, 2, 3, 7):
            if start == ids.shape[1]:
                break
            end = min(start + chunk, ids.shape[1])
            output = model(
                input_ids=ids[:, start:end], attention_mask=mask[:, :end], past_key_values=cache, use_cache=True
            )
            assert output.past_key_values is cache
            assert cache.get_seq_length() == end
            parts.append(output.logits)
            start = end
        actual = paddle.concat(parts, axis=1)
        # Fully masked padding queries have no defined attention distribution;
        # compare all real tokens, including the other unpadded batch member.
        valid = mask.numpy().astype(bool)
        np.testing.assert_allclose(actual.numpy()[valid], expected.numpy()[valid], atol=3e-6, rtol=3e-6)
        for index, kind in enumerate(layer_types):
            if kind == "conv":
                assert list(cache.conv_states[index].shape) == [2, config.hidden_size, kernel_size - 1]
            else:
                assert cache.layers[index].keys.shape[-2] == ids.shape[1]
        order = paddle.to_tensor([1, 0, 1])
        before_conv = [None if x is None else x.clone() for x in cache.conv_states]
        before_keys = [None if x.keys is None else x.keys.clone() for x in cache.layers]
        cache.reorder_cache(order)
        for index, value in enumerate(before_conv):
            if value is not None:
                np.testing.assert_array_equal(cache.conv_states[index].numpy(), value.numpy()[[1, 0, 1]])
        for index, value in enumerate(before_keys):
            if value is not None:
                np.testing.assert_array_equal(cache.layers[index].keys.numpy(), value.numpy()[[1, 0, 1]])
        cache.reset()
        assert cache.get_seq_length() == 0
        assert all(value is None for value in cache.conv_states)


def test_multimodal_generate_reuses_vision_and_returns_same_tokens():
    config = get_config()
    config.text_config._attn_implementation = "eager"
    model = Lfm2VlForConditionalGeneration(config).eval()
    ids = paddle.to_tensor([[1, 7, 2, 3]])
    inputs = dict(
        input_ids=ids,
        attention_mask=paddle.ones_like(ids),
        pixel_values=paddle.randn([1, 4, 12]),
        pixel_attention_mask=paddle.ones([1, 4], dtype="int64"),
        spatial_shapes=paddle.to_tensor([[2, 2]]),
    )
    options = dict(decode_strategy="greedy_search", max_new_tokens=4, min_new_tokens=4, eos_token_id=127)
    with patch.object(model.model.vision_tower, "forward", wraps=model.model.vision_tower.forward) as vision:
        cached = model.generate(**inputs, use_cache=True, **options)[0]
        assert vision.call_count == 1
    with patch.object(model.model.vision_tower, "forward", wraps=model.model.vision_tower.forward) as vision:
        uncached = model.generate(**inputs, use_cache=False, **options)[0]
        assert vision.call_count == 4
    np.testing.assert_array_equal(cached.numpy(), uncached.numpy())


def test_cache_rejects_unsupported_packed_decode_instead_of_leaking_documents():
    model = Lfm2ForCausalLM(get_config().text_config).eval()
    with pytest.raises(ValueError, match="packed training"):
        model(
            input_ids=paddle.to_tensor([[1, 2]]),
            use_cache=True,
            attn_mask_startend_row_indices=paddle.to_tensor([[[1, 2]]], dtype="int32"),
        )
