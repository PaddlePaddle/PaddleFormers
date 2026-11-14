# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Exercise the public Auto entry points with real local checkpoints."""
import json
import shutil

import numpy as np
import paddle
import pytest
from safetensors.numpy import save_file

from paddleformers.transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForConditionalGeneration,
    FalconPerceptionConfig,
    FalconPerceptionForSegmentation,
)

AUTO_CLASSES = (AutoModel, AutoModelForCausalLM, AutoModelForConditionalGeneration)


def tiny_config(**kwargs):
    return FalconPerceptionConfig(
        dim=16,
        n_layers=1,
        n_heads=2,
        head_dim=8,
        n_kv_heads=1,
        ffn_dim=24,
        vocab_size=300,
        max_seq_len=32,
        coord_enc_dim=8,
        coord_dec_dim=16,
        coord_out_dim=16,
        size_enc_dim=8,
        size_dec_dim=16,
        size_out_dim=16,
        do_segmentation=False,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def device():
    previous = paddle.get_device()
    paddle.set_device("cpu")
    yield
    paddle.set_device(previous)


@pytest.fixture
def hf_checkpoint(tmp_path):
    """Use HF architecture and Linear layout, including non-square weights."""
    config = tiny_config(architectures=["FalconPerceptionForSegmentation"])
    model = FalconPerceptionForSegmentation(config)
    rng = np.random.default_rng(123)
    linear = {f"{name}.weight" for name, layer in model.named_sublayers() if isinstance(layer, paddle.nn.Linear)}
    expected = {
        key: rng.normal(size=list(tensor.shape)).astype("float32") for key, tensor in model.state_dict().items()
    }
    values = {key: value.T.copy() if key in linear else value for key, value in expected.items()}
    config.save_pretrained(str(tmp_path))
    save_file(values, str(tmp_path / "model.safetensors"))
    return tmp_path, expected


def test_auto_config_and_model_type_fallback(tmp_path):
    # No architecture and no model-name hint in the path: resolve model_type.
    tiny_config().save_pretrained(str(tmp_path))
    config = AutoConfig.from_pretrained(str(tmp_path))
    assert type(config) is FalconPerceptionConfig
    for auto_class in AUTO_CLASSES:
        assert type(auto_class.from_config(config)) is FalconPerceptionForSegmentation


@pytest.mark.parametrize("auto_class", AUTO_CLASSES)
def test_auto_loads_hf_weights(auto_class, hf_checkpoint):
    directory, expected = hf_checkpoint
    model = auto_class.from_pretrained(str(directory), dtype="float32", local_files_only=True)
    assert type(model) is FalconPerceptionForSegmentation
    assert not model.training
    assert model.weight_loading_info["num_transposed"] > 0
    for key, value in model.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), expected[key])


@pytest.mark.parametrize("auto_class", AUTO_CLASSES)
def test_auto_loads_hf_subfolder(auto_class, hf_checkpoint, tmp_path):
    directory, expected = hf_checkpoint
    nested = tmp_path / "nested"
    nested.mkdir()
    for name in ("config.json", "model.safetensors"):
        shutil.copy2(directory / name, nested / name)
    config = AutoConfig.from_pretrained(str(nested))
    model = auto_class.from_pretrained(
        str(tmp_path), subfolder="nested", config=config, cache_dir=str(tmp_path / "cache"), local_files_only=True
    )
    assert model.weight_loading_info["source"] == str(nested)
    for key, value in model.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), expected[key])


def test_auto_ignores_original_remote_code(hf_checkpoint):
    directory, _ = hf_checkpoint
    path = directory / "config.json"
    values = json.loads(path.read_text())
    values["auto_map"] = {
        "AutoConfig": "configuration_broken.BrokenConfig",
        "AutoModelForCausalLM": "modeling_broken.BrokenModel",
    }
    path.write_text(json.dumps(values))
    for name in ("configuration_broken.py", "modeling_broken.py"):
        (directory / name).write_text('raise RuntimeError("Original HF code must not run")\n')
    assert type(AutoConfig.from_pretrained(str(directory))) is FalconPerceptionConfig
    assert type(AutoModelForCausalLM.from_pretrained(str(directory))) is FalconPerceptionForSegmentation


def test_auto_from_config_dtype_restores_default():
    previous = paddle.get_default_dtype()
    try:
        paddle.set_default_dtype("float64")
        model = AutoModel.from_config(tiny_config(), dtype="float32")
        assert model.dtype == paddle.float32
        assert paddle.get_default_dtype() == "float64"
        with pytest.raises(ValueError, match="Supported dtypes"):
            AutoModel.from_config(tiny_config(), dtype="float16")
        assert paddle.get_default_dtype() == "float64"
    finally:
        paddle.set_default_dtype(previous)


def test_auto_cached_inference():
    if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() == 0:
        pytest.skip("The installed RMSNorm kernel requires a GPU.")
    from paddleformers.transformers.falcon_perception.modeling import KVCache

    paddle.set_device("gpu:0")
    model = AutoModelForCausalLM.from_config(tiny_config()).eval()
    ids = paddle.to_tensor([[12, 13, 14, 15]], dtype="int64")
    mask = model.get_attention_mask(paddle.to_tensor([[12, 13, 14, 15, 16]]))
    cache = KVCache(1, 32, model.config.n_heads, model.config.head_dim, model.config.n_layers)
    with paddle.no_grad():
        logits, hidden = model(
            tokens=ids,
            attention_mask=mask,
            kv_cache=cache,
            rope_pos_t=paddle.arange(4, dtype="int64")[None],
            rope_pos_hw=paddle.full([1, 4, 2], float("nan")),
        )
        assert logits.shape == [1, 4, 300]
        assert hidden.shape == [1, 4, 16]
        assert cache.get_pos() == 4
        assert bool(paddle.isfinite(logits).all())
        logits, hidden = model(tokens=paddle.to_tensor([[16]]), attention_mask=mask, kv_cache=cache)
        assert logits.shape == [1, 1, 300]
        assert hidden.shape == [1, 1, 16]
        assert cache.get_pos() == 5
        assert bool(paddle.isfinite(logits).all())
