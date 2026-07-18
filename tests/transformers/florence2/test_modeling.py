# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

import os
from tempfile import TemporaryDirectory

import numpy as np
import paddle
import pytest

from paddleformers.peft.lora import LoRAConfig, LoRAModel
from paddleformers.transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from paddleformers.transformers.florence2 import (
    Florence2Config,
    Florence2ForConditionalGeneration,
    Florence2Processor,
)
from paddleformers.transformers.florence2.modeling import (
    BaseModelOutput,
    shift_tokens_right,
)


def _small_config():
    return Florence2Config(
        vision_config={
            "dim_embed": [16, 32, 64, 64],
            "num_heads": [2, 4, 4, 4],
            "num_groups": [2, 4, 4, 4],
            "depths": [1, 1, 1, 1],
            "patch_size": [3, 3, 3, 3],
            "patch_stride": [2, 2, 2, 2],
            "patch_padding": [1, 1, 1, 1],
            "patch_prenorm": [False, True, True, True],
            "window_size": 2,
            "projection_dim": 16,
        },
        text_config={
            "vocab_size": 32,
            "d_model": 16,
            "encoder_attention_heads": 2,
            "decoder_attention_heads": 2,
            "encoder_ffn_dim": 32,
            "decoder_ffn_dim": 32,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "max_position_embeddings": 32,
        },
        vocab_size=32,
    )


def _small_model():
    model = Florence2ForConditionalGeneration(_small_config())
    model.eval()
    return model


def test_florence2_small_forward_and_loss():
    model = _small_model()
    output = model(
        input_ids=paddle.to_tensor([[0, 5, 2]]),
        pixel_values=paddle.randn([1, 3, 32, 48]),
        labels=paddle.to_tensor([[4, 5, 2]]),
        use_cache=False,
    )
    assert output.logits.shape == [1, 3, 32]
    assert paddle.isfinite(output.loss)


def test_florence2_return_dict_false_keeps_seq2seq_outputs():
    model = _small_model()
    inputs = {
        "input_ids": paddle.to_tensor([[0, 5, 2]]),
        "pixel_values": paddle.randn([1, 3, 32, 48]),
        "decoder_input_ids": paddle.to_tensor([[2, 4, 5]]),
        "use_cache": True,
    }
    dict_output = model(**inputs)
    tuple_output = model(**inputs, return_dict=False)
    assert isinstance(tuple_output, tuple)
    assert paddle.allclose(tuple_output[0], dict_output.logits)
    assert len(tuple_output) == len(dict_output.to_tuple())
    assert tuple_output[1] is not None

    labels_output = model(**inputs, labels=paddle.to_tensor([[4, 5, 2]]), return_dict=False)
    assert isinstance(labels_output, tuple)
    assert labels_output[0].ndim == 0
    assert paddle.allclose(labels_output[1], model(**inputs, labels=paddle.to_tensor([[4, 5, 2]])).logits)


def test_florence2_processor_rejects_mismatched_batch_sizes():
    class ImageProcessor:
        def __call__(self, images, return_tensors, **kwargs):
            return {"pixel_values": np.zeros([len(images), 3, 2, 2], dtype="float32")}

    class Tokenizer:
        def __call__(self, texts, return_tensors, **kwargs):
            return {
                "input_ids": np.zeros([len(texts), 2], dtype="int64"),
                "attention_mask": np.ones([len(texts), 2], dtype="int64"),
            }

    processor = Florence2Processor.__new__(Florence2Processor)
    processor.image_processor = ImageProcessor()
    processor.tokenizer = Tokenizer()
    processor.image_seq_length = 2
    processor.task_prompts_without_inputs = {}
    processor.task_prompts_with_input = {}

    with pytest.raises(ValueError, match="Each prompt must be associated with an image"):
        processor(text=["<CAPTION>"], images=[np.zeros([2, 2, 3]), np.zeros([2, 2, 3])])
    with pytest.raises(ValueError, match="Each prompt must be associated with an image"):
        processor(text=["<CAPTION>", "<OCR>"], images=[np.zeros([2, 2, 3])])
    inputs = processor(text=None, images=[np.zeros([2, 2, 3]), np.zeros([2, 2, 3])])
    assert list(inputs["input_ids"].shape)[0] == 2


def test_florence2_lora_train_step_and_merge():
    model = _small_model()
    lora_model = LoRAModel(
        model,
        LoRAConfig(target_modules=[".*q_proj$", ".*v_proj$"], r=2, lora_alpha=4),
    )
    input_ids = paddle.to_tensor([[0, 5, 2]])
    pixel_values = paddle.randn([1, 3, 32, 48])
    labels = paddle.to_tensor([[4, 5, 2]])
    lora_model.train()
    output = lora_model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
    output.loss.backward()
    trainable_parameters = [parameter for parameter in lora_model.parameters() if not parameter.stop_gradient]
    assert any(parameter.grad is not None for parameter in trainable_parameters)
    optimizer = paddle.optimizer.AdamW(learning_rate=1e-3, parameters=trainable_parameters)
    optimizer.step()
    optimizer.clear_grad()
    lora_model.eval()
    before_merge = lora_model(input_ids=input_ids, pixel_values=pixel_values, labels=labels).logits
    lora_model.merge()
    after_merge = lora_model(input_ids=input_ids, pixel_values=pixel_values, labels=labels).logits
    assert paddle.allclose(before_merge, after_merge, atol=1e-5)


def test_florence2_greedy_generation():
    model = _small_model()
    generated_ids, _ = model.generate(
        input_ids=paddle.to_tensor([[0, 5, 2]]),
        pixel_values=paddle.randn([1, 3, 32, 48]),
        max_new_tokens=3,
        decode_strategy="greedy_search",
    )
    assert generated_ids.shape[0] == 1
    assert 1 <= generated_ids.shape[1] <= 3


def test_florence2_lora_save_and_load():
    input_ids = paddle.to_tensor([[0, 5, 2]])
    pixel_values = paddle.randn([1, 3, 32, 48])
    labels = paddle.to_tensor([[4, 5, 2]])
    lora_config = LoRAConfig(target_modules=[".*q_proj$", ".*v_proj$"], r=2, lora_alpha=4)
    base_model = _small_model()
    base_state = base_model.state_dict()
    lora_model = LoRAModel(base_model, lora_config)
    lora_model.eval()
    expected_logits = lora_model(input_ids=input_ids, pixel_values=pixel_values, labels=labels).logits
    with TemporaryDirectory() as tempdir:
        lora_model.save_pretrained(tempdir)
        loaded_base_model = _small_model()
        loaded_base_model.set_dict(base_state)
        loaded_lora_model = LoRAModel.from_pretrained(loaded_base_model, tempdir)
        loaded_lora_model.eval()
        actual_logits = loaded_lora_model(input_ids=input_ids, pixel_values=pixel_values, labels=labels).logits
    assert paddle.allclose(expected_logits, actual_logits, atol=1e-5)


def test_florence2_shift_labels_and_default_decoder_inputs():
    model = _small_model()
    labels = paddle.to_tensor([[4, -100, 2]])
    expected = paddle.to_tensor([[2, 4, 1]])
    assert paddle.equal(shift_tokens_right(labels, 1, 2), expected).all()
    assert paddle.equal(model.prepare_decoder_input_ids_from_labels(labels), expected).all()

    output = model(input_ids=paddle.to_tensor([[0, 5, 2]]), labels=labels)
    assert output.logits.shape == [1, 3, 32]
    assert output.past_key_values is None

    language_output = model.language_model.model(input_ids=paddle.to_tensor([[0, 5, 2]]), use_cache=False)
    assert language_output.last_hidden_state.shape == [1, 3, 16]


def test_florence2_prefill_cache_and_direct_last_logits():
    model = _small_model()
    input_ids = paddle.to_tensor([[0, 5, 2]])
    pixel_values = paddle.randn([1, 3, 32, 48])
    decoder_input_ids = paddle.to_tensor([[2, 4]])
    text_attention_mask = paddle.ones(input_ids.shape, dtype="int64")
    decoder_attention_mask = paddle.ones(decoder_input_ids.shape, dtype="int64")
    prefill = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=text_attention_mask,
        decoder_attention_mask=decoder_attention_mask,
        decoder_input_ids=decoder_input_ids,
        use_cache=True,
    )
    encoder_outputs = BaseModelOutput(last_hidden_state=prefill.encoder_last_hidden_state)
    encoder_attention_mask = paddle.ones(prefill.encoder_last_hidden_state.shape[:2], dtype="int64")
    full_decoder_ids = paddle.to_tensor([[2, 4, 5]])
    prepared = model.prepare_inputs_for_generation(
        full_decoder_ids,
        past_key_values=prefill.past_key_values,
        encoder_outputs=encoder_outputs,
        attention_mask=encoder_attention_mask,
        pixel_values=None,
        decoder_attention_mask=paddle.ones(full_decoder_ids.shape, dtype="int64"),
        use_cache=True,
    )
    assert prepared["decoder_input_ids"].shape == [1, 1]
    assert prepared["pixel_values"] is None
    assert prepared["attention_mask"] is encoder_attention_mask
    assert prepared["decoder_attention_mask"].shape == [1, 3]
    assert prepared["use_cache"] is True

    cached = model(**prepared)
    direct = model(
        encoder_outputs=encoder_outputs,
        attention_mask=encoder_attention_mask,
        decoder_input_ids=full_decoder_ids,
        decoder_attention_mask=paddle.ones(full_decoder_ids.shape, dtype="int64"),
        use_cache=False,
    )
    assert paddle.allclose(cached.logits[:, -1], direct.logits[:, -1], atol=1e-5)


def test_florence2_local_config_auto_classes():
    checkpoint = os.environ.get("FLORENCE2_CHECKPOINT_DIR")
    if not checkpoint:
        pytest.skip("FLORENCE2_CHECKPOINT_DIR is not set")

    config = AutoConfig.from_pretrained(checkpoint)
    assert isinstance(config, Florence2Config)

    model = AutoModelForCausalLM.from_pretrained(checkpoint, load_checkpoint_format="flex_checkpoint", dtype="float32")
    assert isinstance(model, Florence2ForConditionalGeneration)


def test_florence2_processor_from_pretrained():
    checkpoint = os.environ.get("FLORENCE2_CHECKPOINT_DIR")
    if not checkpoint:
        pytest.skip("FLORENCE2_CHECKPOINT_DIR is not set")

    processor = AutoProcessor.from_pretrained(checkpoint)
    inputs = processor(text="<CAPTION>", images=np.zeros([768, 768, 3], dtype="uint8"))
    assert processor.image_seq_length == 577
    assert list(inputs["input_ids"].shape) == [1, 8]
    assert list(inputs["attention_mask"].shape) == [1, 8]
    assert list(inputs["pixel_values"].shape) == [1, 3, 768, 768]
    assert inputs["input_ids"].dtype == paddle.int64
    assert inputs["pixel_values"].dtype == paddle.float32


def test_florence2_real_lora_train_step_and_merge():
    checkpoint = os.environ.get("FLORENCE2_CHECKPOINT_DIR")
    reference_path = os.environ.get("FLORENCE2_TORCH_REFERENCE")
    if not checkpoint or not reference_path:
        pytest.skip("FLORENCE2_CHECKPOINT_DIR and FLORENCE2_TORCH_REFERENCE must both be set")

    reference = np.load(reference_path)
    lora_model = LoRAModel(
        Florence2ForConditionalGeneration.from_pretrained(
            checkpoint, load_checkpoint_format="flex_checkpoint", dtype="float32"
        ),
        LoRAConfig(target_modules=[".*q_proj$", ".*v_proj$"], r=2, lora_alpha=4),
    )
    input_ids = paddle.to_tensor(reference["input_ids"], dtype="int64")
    attention_mask = paddle.to_tensor(reference["attention_mask"], dtype="int64")
    pixel_values = paddle.to_tensor(reference["pixel_values"], dtype="float32")
    labels = paddle.to_tensor(reference["labels"], dtype="int64")
    trainable_parameters = [parameter for parameter in lora_model.parameters() if not parameter.stop_gradient]
    lora_model.train()
    output = lora_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels,
    )
    output.loss.backward()
    assert paddle.isfinite(output.loss)
    assert any(
        parameter.grad is not None and paddle.isfinite(parameter.grad).all() for parameter in trainable_parameters
    )
    optimizer = paddle.optimizer.AdamW(learning_rate=1e-5, parameters=trainable_parameters)
    optimizer.step()
    optimizer.clear_grad()
    lora_model.eval()
    before_merge = lora_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels,
    ).logits
    lora_model.merge()
    after_merge = lora_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels,
    ).logits
    print(f"FLORENCE2_REAL_LORA_MERGE max_abs={float(paddle.max(paddle.abs(before_merge - after_merge))):.8g}")
    assert paddle.allclose(before_merge, after_merge, atol=1e-4)


def test_florence2_real_checkpoint_reference_and_cache():
    checkpoint = os.environ.get("FLORENCE2_CHECKPOINT_DIR")
    reference_path = os.environ.get("FLORENCE2_TORCH_REFERENCE")
    if not checkpoint or not reference_path:
        pytest.skip("FLORENCE2_CHECKPOINT_DIR and FLORENCE2_TORCH_REFERENCE must both be set")

    reference = np.load(reference_path)
    input_ids = paddle.to_tensor(reference["input_ids"], dtype="int64")
    attention_mask = paddle.to_tensor(reference["attention_mask"], dtype="int64")
    pixel_values = paddle.to_tensor(reference["pixel_values"], dtype="float32")
    labels = paddle.to_tensor(reference["labels"], dtype="int64")
    model = Florence2ForConditionalGeneration.from_pretrained(
        checkpoint, load_checkpoint_format="flex_checkpoint", dtype="float32"
    )
    model.eval()

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels,
        use_cache=True,
    )
    logits_diff = np.abs(output.logits.numpy() - reference["logits"])
    intermediates_path = os.environ.get("FLORENCE2_TORCH_INTERMEDIATES")
    if intermediates_path:
        intermediates = np.load(intermediates_path)
        vision_features = model.vision_tower.forward_features_unpool(pixel_values).numpy()
        image_features = model._encode_image(pixel_values).numpy()
        encoder_hidden_states = output.encoder_last_hidden_state.numpy()
        print(
            "FLORENCE2_INTERMEDIATE_ALIGNMENT "
            f"vision_max_abs={np.abs(vision_features - intermediates['vision_features']).max():.8g} "
            f"image_max_abs={np.abs(image_features - intermediates['image_features']).max():.8g} "
            f"encoder_max_abs={np.abs(encoder_hidden_states - intermediates['encoder_last_hidden_state']).max():.8g}"
        )
    valid_labels = reference["labels"] != -100
    reference_token_logits = reference["logits"][valid_labels].astype("float64")
    reference_labels = reference["labels"][valid_labels]
    reference_loss = np.mean(
        np.logaddexp.reduce(reference_token_logits, axis=-1)
        - reference_token_logits[np.arange(reference_labels.size), reference_labels]
    )
    loss_diff = abs(float(output.loss) - reference_loss)
    print(
        "FLORENCE2_LOGITS_ALIGNMENT "
        f"max_abs={logits_diff.max():.8g} mean_abs={logits_diff.mean():.8g} "
        f"paddle_loss={float(output.loss):.8g} torch_loss={reference_loss:.8g} loss_abs={loss_diff:.8g} "
        f"argmax_equal={np.array_equal(output.logits.numpy().argmax(-1), reference['logits'].argmax(-1))}"
    )
    assert list(output.logits.shape) == list(reference["logits"].shape)
    assert paddle.isfinite(output.logits).all()
    assert paddle.isfinite(output.loss)
    assert loss_diff < 1e-3
    assert logits_diff.max() < 1e-3
    assert logits_diff.mean() < 1e-4
    assert np.array_equal(output.logits.numpy().argmax(-1), reference["logits"].argmax(-1))

    decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels)
    prefill = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        decoder_input_ids=decoder_input_ids[:, :-1],
        decoder_attention_mask=paddle.ones(decoder_input_ids[:, :-1].shape, dtype="int64"),
        use_cache=True,
    )
    encoder_outputs = BaseModelOutput(last_hidden_state=prefill.encoder_last_hidden_state)
    encoder_attention_mask = paddle.ones(prefill.encoder_last_hidden_state.shape[:2], dtype="int64")
    full_decoder_attention_mask = paddle.ones(decoder_input_ids.shape, dtype="int64")
    prepared = model.prepare_inputs_for_generation(
        decoder_input_ids,
        past_key_values=prefill.past_key_values,
        encoder_outputs=encoder_outputs,
        attention_mask=encoder_attention_mask,
        pixel_values=None,
        decoder_attention_mask=full_decoder_attention_mask,
        use_cache=True,
    )
    cached = model(**prepared)
    direct = model(
        encoder_outputs=encoder_outputs,
        attention_mask=encoder_attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=full_decoder_attention_mask,
        use_cache=False,
    )
    assert paddle.allclose(cached.logits[:, -1], direct.logits[:, -1], atol=1e-5)

    reference_prefill = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        decoder_input_ids=decoder_input_ids[:, :1],
        decoder_attention_mask=paddle.ones([1, 1], dtype="int64"),
        use_cache=True,
    )
    reference_encoder_outputs = BaseModelOutput(last_hidden_state=reference_prefill.encoder_last_hidden_state)
    reference_encoder_mask = paddle.ones(reference_prefill.encoder_last_hidden_state.shape[:2], dtype="int64")
    reference_decoder_ids = decoder_input_ids[:, :2]
    reference_prepared = model.prepare_inputs_for_generation(
        reference_decoder_ids,
        past_key_values=reference_prefill.past_key_values,
        encoder_outputs=reference_encoder_outputs,
        attention_mask=reference_encoder_mask,
        decoder_attention_mask=paddle.ones(reference_decoder_ids.shape, dtype="int64"),
        use_cache=True,
    )
    reference_cached = model(**reference_prepared)
    reference_direct = model(
        encoder_outputs=reference_encoder_outputs,
        attention_mask=reference_encoder_mask,
        decoder_input_ids=reference_decoder_ids,
        decoder_attention_mask=paddle.ones(reference_decoder_ids.shape, dtype="int64"),
        use_cache=False,
    )
    first_diff = np.abs(reference_prefill.logits.numpy() - reference["first_logits"])
    cached_diff = np.abs(reference_cached.logits.numpy() - reference["cached_logits"])
    direct_diff = np.abs(reference_direct.logits.numpy() - reference["direct_logits"])
    print(
        "FLORENCE2_CACHE_ALIGNMENT "
        f"first_max_abs={first_diff.max():.8g} cached_max_abs={cached_diff.max():.8g} "
        f"direct_max_abs={direct_diff.max():.8g}"
    )
    assert first_diff.max() < 1e-3
    assert cached_diff.max() < 1e-3
    assert direct_diff.max() < 1e-3
    assert np.array_equal(reference_cached.logits.numpy().argmax(-1), reference["cached_logits"].argmax(-1))

    generated_ids, _ = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        max_new_tokens=reference["generated_ids"].shape[1] - 1,
        decode_strategy="greedy_search",
    )
    expected_generated_ids = reference["generated_ids"][:, 1:]
    print(
        "FLORENCE2_GENERATION_ALIGNMENT "
        f"shape={list(generated_ids.shape)} exact_match={np.array_equal(generated_ids.numpy(), expected_generated_ids)}"
    )
    assert np.array_equal(generated_ids.numpy(), expected_generated_ids)


def test_florence2_reorder_cache_and_tied_embeddings():
    model = _small_model()
    assert model.language_model.model.encoder.embed_tokens is model.language_model.model.shared
    assert model.language_model.model.decoder.embed_tokens is model.language_model.model.shared
    assert model.language_model.lm_head.embed_tokens is model.language_model.model.shared

    self_key = paddle.arange(16, dtype="float32").reshape([2, 1, 2, 4])
    self_value = self_key + 100
    cross_key = self_key + 200
    cross_value = self_key + 300
    reordered = model._reorder_cache(((self_key, self_value, cross_key, cross_value),), paddle.to_tensor([1, 0]))
    assert paddle.equal(reordered[0][0], self_key[[1, 0]]).all()
    assert paddle.equal(reordered[0][1], self_value[[1, 0]]).all()
    assert paddle.equal(reordered[0][2], cross_key).all()
    assert paddle.equal(reordered[0][3], cross_value).all()
