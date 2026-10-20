# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Small independent checks of masks, window gathers, fused FFN, and loading rules."""
import json
import math

import numpy as np
import paddle
import pytest
from safetensors.numpy import save_file

from paddleformers.transformers.falcon_perception import (
    FalconPerceptionConfig,
    FalconPerceptionForSegmentation,
)
from paddleformers.transformers.falcon_perception.attention import (
    AttentionMask,
    WindowMask,
    attention_with_lse,
    window_attention,
)
from paddleformers.transformers.falcon_perception.modeling import (
    KVCache,
    squared_relu_gate,
)
from paddleformers.transformers.falcon_perception.rope import (
    apply_rotary_emb,
    precompute_freqs_cis,
)


@pytest.fixture(autouse=True)
def cpu_device():
    old = paddle.get_device()
    paddle.set_device("cpu")
    yield
    paddle.set_device(old)


def test_hybrid_mask_and_decode_offset():
    tokens = [0, 244, 245, 227, 227, 230, 42, 11, 43, 0]
    mask = AttentionMask(paddle.to_tensor([tokens]), 0, 11, 244, 230)
    actual = mask.materialize(10, 10).numpy()[0, 0]
    documents = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    visual = set(range(1, 5))
    expected = np.array(
        [
            [(q >= k and documents[q] == documents[k] and k > 0) or (q in visual and k in visual) for k in range(10)]
            for q in range(10)
        ]
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(mask.with_offset(8).materialize(1, 9).numpy()[0, 0], expected[8:9, :9])


def test_sink_attention_matches_extra_zero_value():
    rng = np.random.default_rng(4)
    q = rng.normal(size=[1, 2, 5, 8]).astype("float32")
    k = rng.normal(size=[1, 2, 5, 8]).astype("float32")
    v = rng.normal(size=[1, 2, 5, 6]).astype("float32")
    sink = np.array([0.3, -0.2], dtype="float32")
    mask = AttentionMask(paddle.to_tensor([[244, 227, 230, 42, 43]]), 0, 11, 244, 230)
    output, lse = attention_with_lse(paddle.to_tensor(q), paddle.to_tensor(k), paddle.to_tensor(v), mask)
    actual = (output * paddle.nn.functional.sigmoid(lse - paddle.to_tensor(sink)[None, :, None]).unsqueeze(-1)).numpy()
    scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(8)
    scores = np.where(mask.materialize(5, 5).numpy(), scores, -np.inf)
    combined = np.concatenate([scores, np.broadcast_to(sink[None, :, None, None], [1, 2, 5, 1])], axis=-1)
    probs = np.exp(combined - combined.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(actual, probs[..., :-1] @ v, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("high,low,ratio", [(7, 3, 0.1), (9, 4, 0.25)])
def test_window_attention_matches_dense_with_different_value_width(high, low, ratio):
    rng = np.random.default_rng(7)
    q = rng.normal(size=[1, 2, high * high, 4]).astype("float32")
    k = rng.normal(size=[1, 2, low * low, 4]).astype("float32")
    v = rng.normal(size=[1, 2, low * low, 6]).astype("float32")
    mask = WindowMask(high, high, low, low, ratio=ratio, chunk_size=11)
    actual = window_attention(paddle.to_tensor(q), paddle.to_tensor(k), paddle.to_tensor(v), mask).numpy()
    expected = np.zeros_like(actual)
    for index in range(high * high):
        r = np.float32((index // high + 0.5) / high)
        c = np.float32((index % high + 0.5) / high)
        r0 = int(np.floor(np.clip(np.float32(r - ratio), 0, 1) * low))
        r1 = int(np.ceil(np.clip(np.float32(r + ratio), 0, 1) * low))
        c0 = int(np.floor(np.clip(np.float32(c - ratio), 0, 1) * low))
        c1 = int(np.ceil(np.clip(np.float32(c + ratio), 0, 1) * low))
        selected = [i * low + j for i in range(r0, r1) for j in range(c0, c1)]
        scores = (q[:, :, index, None, :] * k[:, :, selected, :]).sum(-1) / 2
        probs = np.exp(scores - scores.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        expected[:, :, index] = (probs[..., None] * v[:, :, selected, :]).sum(-2)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_temporal_rope_adjacent_pairs():
    x = np.arange(16, dtype="float32").reshape([1, 2, 1, 8]) / 10
    freq = precompute_freqs_cis(8, 2)
    actual, _ = apply_rotary_emb(paddle.to_tensor(x), paddle.to_tensor(x), freq[None])
    f = 1 / (10000 ** (np.arange(0, 8, 2, dtype="float32") / 8))
    angles = np.arange(2)[:, None] * f[None]
    z = x.reshape(1, 2, 1, 4, 2)
    a, b = z[..., 0], z[..., 1]
    expected = np.stack(
        [
            a * np.cos(angles)[None, :, None] - b * np.sin(angles)[None, :, None],
            a * np.sin(angles)[None, :, None] + b * np.cos(angles)[None, :, None],
        ],
        axis=-1,
    ).reshape(x.shape)
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-6)


def test_interleaved_squared_gate():
    x = np.array([[2, 3, -4, 5, 1.5, -2]], dtype="float32")
    np.testing.assert_array_equal(squared_relu_gate(paddle.to_tensor(x), 3).numpy(), [[12, 0, -4.5]])


def tiny_config():
    return FalconPerceptionConfig(
        dim=16,
        n_layers=1,
        n_heads=2,
        head_dim=8,
        n_kv_heads=1,
        ffn_dim=24,
        vocab_size=300,
        max_seq_len=256,
        coord_enc_dim=8,
        coord_dec_dim=16,
        coord_out_dim=16,
        size_enc_dim=8,
        size_dec_dim=16,
        size_out_dim=16,
        do_segmentation=False,
    )


def create_checkpoint(directory):
    config = tiny_config()
    model = FalconPerceptionForSegmentation(config)
    rng = np.random.default_rng(123)
    linear = {f"{name}.weight" for name, layer in model.named_sublayers() if isinstance(layer, paddle.nn.Linear)}
    values = {}
    expected = {}
    for key, tensor in model.state_dict().items():
        shape = list(tensor.shape)
        expected[key] = rng.normal(size=shape).astype("float32")
        values[key] = expected[key].T.copy() if key in linear else expected[key]
    (directory / "config.json").write_text(json.dumps(config.to_dict()))
    save_file(values, str(directory / "model.safetensors"))
    return values, expected


def test_strict_loading_and_no_blanket_2d_transpose(tmp_path):
    _, expected = create_checkpoint(tmp_path)
    model = FalconPerceptionForSegmentation.from_pretrained(tmp_path)
    assert model.weight_loading_info["num_loaded"] == len(expected)
    for key, tensor in model.state_dict().items():
        np.testing.assert_array_equal(tensor.numpy(), expected[key])


def test_missing_and_unexpected_weights_raise(tmp_path):
    values, _ = create_checkpoint(tmp_path)
    values.pop("norm.weight")
    save_file(values, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="Missing checkpoint tensors"):
        FalconPerceptionForSegmentation.from_pretrained(tmp_path)
    values["unexpected.weight"] = np.ones([2, 2], dtype="float32")
    save_file(values, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="Unexpected checkpoint tensor"):
        FalconPerceptionForSegmentation.from_pretrained(tmp_path)


def test_kv_cache_layers_advance_once_and_reset_overwrites():
    cache = KVCache(1, 8, 2, 4, 2)
    cache.set_pos_t(paddle.to_tensor([[5]]))
    first = paddle.arange(24, dtype="float32").reshape([1, 2, 3, 4])
    for layer in range(2):
        keys, values = cache.insert_kv(layer, first + layer, -first)
        np.testing.assert_array_equal(keys.numpy(), (first + layer).numpy())
        assert cache.get_pos() == (0 if layer == 0 else 3)
    np.testing.assert_array_equal(cache.increment_and_get_pos_t().numpy(), [[6]])
    tail = paddle.ones([1, 2, 1, 4]) * 99
    for layer in range(2):
        keys, values = cache.insert_kv(layer, tail, -tail)
        np.testing.assert_array_equal(keys[:, :, -1:].numpy(), tail.numpy())
        assert keys.shape[2] == 4
    cache.reset()
    assert cache.get_pos() == 0 and cache.pos_t is None
    cache.set_pos_t(paddle.to_tensor([[0]]))
    for layer in range(2):
        keys, values = cache.insert_kv(layer, tail, -tail)
        np.testing.assert_array_equal(keys.numpy(), tail.numpy())
        np.testing.assert_array_equal(values.numpy(), -tail.numpy())


def test_checkpoint_shape_mismatch_raises(tmp_path):
    values, _ = create_checkpoint(tmp_path)
    values["output.weight"] = values["output.weight"][:, :-1]
    save_file(values, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="Shape mismatch for output.weight"):
        FalconPerceptionForSegmentation.from_pretrained(tmp_path)


def test_mask_nms_and_rle_roundtrip():
    from pycocotools import mask as mask_utils

    first = np.zeros([6, 7], dtype=bool)
    first[1:4, 1:4] = True
    third = np.zeros_like(first)
    third[4:, 5:] = True
    masks = [paddle.to_tensor(first), paddle.to_tensor(first), paddle.to_tensor(third)]
    assert FalconPerceptionForSegmentation._mask_nms(masks) == [0, 2]
    rles = FalconPerceptionForSegmentation._mask_to_coco_rle(paddle.stack([masks[0], masks[2]]))
    for rle, expected in zip(rles, [first, third]):
        np.testing.assert_array_equal(
            mask_utils.decode(dict(size=rle["size"], counts=rle["counts"].encode())), expected
        )
