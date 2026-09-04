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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import paddle
import pytest

from paddleformers.cli.train.sft.workflow import (
    ModelReproObservationCallback,
    apply_glm_moe_dsa_training_contract,
    load_tokenizer_and_processor,
    save_final_hf_model_if_requested,
    validate_pretokenized_offline_dataset,
)
from paddleformers.datasets.collate import collate_fn
from paddleformers.datasets.SFTDataset import TextSequence


def fixed_sequence():
    return TextSequence(
        token_ids=[154820, 42, 42, 17, 99, 42, 8],
        labels=[42, 42, 17, 99, 42, 8, 3],
        position_ids=[0, 1, 2, 3, 4, 5, 6],
        num_examples=1,
    )


def test_load_tokenizer_uses_independent_source():
    tokenizer = SimpleNamespace()
    processor = SimpleNamespace()
    model_args = SimpleNamespace(
        tokenizer_name_or_path="/tokenizer-only",
        model_name_or_path="/weights-only",
        stage="PT",
    )
    data_args = SimpleNamespace(processor_use_fast=None)

    with patch(
        "paddleformers.cli.train.sft.workflow.AutoTokenizer.from_pretrained",
        return_value=tokenizer,
    ) as load_tokenizer, patch(
        "paddleformers.cli.train.sft.workflow.AutoProcessor.from_pretrained",
        return_value=processor,
    ) as load_processor:
        actual_tokenizer, actual_processor = load_tokenizer_and_processor(model_args, data_args)

    load_tokenizer.assert_called_once_with("/tokenizer-only")
    load_processor.assert_called_once_with("/weights-only", use_fast=None)
    assert actual_tokenizer is tokenizer
    assert actual_processor is processor


def test_final_hf_export_always_writes_last_fc_to_hf():
    trainer = MagicMock()
    args = SimpleNamespace(save_to_hf=False, tensor_model_parallel_size=1)
    assert save_final_hf_model_if_requested(trainer, args) is True
    trainer.save_model.assert_called_once_with(
        merge_tensor_parallel=False,
        last_fc_to_hf=True,
    )

    trainer.reset_mock()
    args.save_to_hf = True
    args.tensor_model_parallel_size = 2
    assert save_final_hf_model_if_requested(trainer, args) is True
    trainer.save_model.assert_called_once_with(
        merge_tensor_parallel=True,
        last_fc_to_hf=True,
    )


def test_validate_pretokenized_offline_dataset_rejects_wrong_length():
    sequence = fixed_sequence()
    sequence.position_ids = sequence.position_ids[:-1]

    with pytest.raises(ValueError, match="position_ids length 6 != 7"):
        validate_pretokenized_offline_dataset([[sequence]], expected_length=7)


def test_glm_moe_dsa_training_contract_propagates_frozen_provider_fields():
    model_config = SimpleNamespace(model_type="glm_moe_dsa")
    training_args = SimpleNamespace(
        num_nextn_predict_layers=1,
        mtp_num_layers=1,
        fp32_residual_connection=False,
        moe_token_dispatcher_type="alltoall",
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        context_parallel_size=1,
        expert_model_parallel_size=2,
        expert_tensor_model_parallel_size=1,
        sequence_parallel=True,
    )
    model_args = SimpleNamespace(mtp_attention_flexible=True, persist_layer_norm=False)
    data_args = SimpleNamespace(pretokenized_dataset=True)

    apply_glm_moe_dsa_training_contract(model_config, training_args, model_args, data_args)

    assert not hasattr(model_config, "mtp_num_layers")
    assert training_args.mtp_num_layers == 0
    assert model_config.num_nextn_predict_layers == 1
    assert model_config.mtp_enabled is True
    assert model_config.fp32_residual_connection is False
    assert model_config.moe_token_dispatcher_type == "alltoall"
    assert model_config.tensor_model_parallel_size == 2
    assert model_config.pipeline_model_parallel_size == 4
    assert model_config.context_parallel_size == 1
    assert model_config.expert_model_parallel_size == 2
    assert model_config.expert_tensor_parallel_size == 1
    assert model_config.sequence_parallel is True
    assert model_config.persist_layer_norm is False


def test_glm_moe_dsa_training_contract_rejects_invalid_expert_tensor_parallel_size():
    model_config = SimpleNamespace(model_type="glm_moe_dsa")
    training_args = SimpleNamespace(
        num_nextn_predict_layers=1,
        mtp_num_layers=1,
        fp32_residual_connection=False,
        moe_token_dispatcher_type="alltoall",
        expert_tensor_model_parallel_size=0,
    )
    model_args = SimpleNamespace(mtp_attention_flexible=True, persist_layer_norm=False)
    data_args = SimpleNamespace(pretokenized_dataset=True)

    with pytest.raises(ValueError, match="expert_tensor_model_parallel_size must be -1 or at least 1"):
        apply_glm_moe_dsa_training_contract(model_config, training_args, model_args, data_args)


def test_glm_moe_dsa_pretokenized_mtp_requires_flexible_mask():
    model_config = SimpleNamespace(model_type="glm_moe_dsa")
    training_args = SimpleNamespace(
        num_nextn_predict_layers=1,
        mtp_num_layers=1,
        fp32_residual_connection=False,
        moe_token_dispatcher_type="alltoall",
    )
    model_args = SimpleNamespace(mtp_attention_flexible=False, persist_layer_norm=False)
    data_args = SimpleNamespace(pretokenized_dataset=True)

    with pytest.raises(ValueError, match="mtp_attention_flexible=true"):
        apply_glm_moe_dsa_training_contract(model_config, training_args, model_args, data_args)


def test_pretokenized_mtp_padding_uses_explicit_zero_sentinel():
    tokenizer = SimpleNamespace(pad_token_id=154820)
    training_args = SimpleNamespace(
        num_nextn_predict_layers=1,
        context_parallel_size=1,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        fp8=False,
        max_seq_len=7,
    )
    model_args = SimpleNamespace(
        mtp_attention_flexible=False,
        use_attn_mask_startend_row_indices=False,
        use_global_causal_attn=False,
    )

    batch = collate_fn(
        [[fixed_sequence()]],
        tokenizer=tokenizer,
        training_args=training_args,
        model_args=model_args,
        max_seq_len=None,
        padding_free=False,
        input_pad_token_id=0,
    )

    assert batch["input_ids"].tolist() == [[154820, 42, 42, 17, 99, 42, 8, 0]]
    assert batch["labels"].tolist() == [[42, 42, 17, 99, 42, 8, 3, -100]]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3, 4, 5, 6, 0]]


def test_pretokenized_mtp_flexible_mask_matches_main_stream_length():
    tokenizer = SimpleNamespace(pad_token_id=154820)
    training_args = SimpleNamespace(
        num_nextn_predict_layers=1,
        context_parallel_size=1,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        fp8=False,
        max_seq_len=7,
    )
    model_args = SimpleNamespace(
        mtp_attention_flexible=True,
        use_attn_mask_startend_row_indices=False,
        use_global_causal_attn=False,
    )

    batch = collate_fn(
        [[fixed_sequence()]],
        tokenizer=tokenizer,
        training_args=training_args,
        model_args=model_args,
        max_seq_len=None,
        padding_free=False,
        input_pad_token_id=0,
    )

    assert list(batch["input_ids"].shape) == [1, 8]
    assert list(batch["attention_mask"].shape) == [1, 1, 7, 7]


class FakeTensor:
    def __init__(self, values):
        self._values = values
        self.shape = [1, len(values)]
        self.dtype = "int64"

    def detach(self):
        return self

    def cast(self, dtype):
        return self

    def reshape(self, shape):
        return self

    def numpy(self):
        class Array:
            def __init__(self, values):
                self._values = values

            def tolist(self):
                return list(self._values)

        return Array(self._values)


def test_model_repro_observation_callback_projects_paddle_57_44(tmp_path, monkeypatch):
    callback = ModelReproObservationCallback(
        raw_loss_path=str(tmp_path / "raw_loss.jsonl"),
        input_receipt_path=str(tmp_path / "input_receipt.json"),
    )
    monkeypatch.setattr(
        "paddleformers.cli.train.sft.workflow.paddle.distributed.get_rank",
        lambda: 0,
    )
    state = SimpleNamespace(global_step=0, is_world_process_zero=True)
    args = SimpleNamespace(num_nextn_predict_layers=1)
    input_ids = list(range(57)) + [154820] * 4
    dataset_labels = [-100] * 13 + list(range(44))
    model_labels = dataset_labels[1:] + [-100] + [-100] * 4
    position_ids = list(range(57)) + [0] * 4
    callback.on_load_data_end(
        args,
        state,
        SimpleNamespace(),
        inputs={
            "input_ids": FakeTensor(input_ids),
            "labels": FakeTensor(model_labels),
            "position_ids": FakeTensor(position_ids),
        },
    )
    receipt = __import__("json").loads((tmp_path / "input_receipt.json").read_text())
    assert receipt["semantic"]["input_token_count"] == 57
    assert receipt["semantic"]["supervised_target_count"] == 44
    assert receipt["semantic"]["projection"] == "dataset_row_before_paddle_padding_and_label_roll"
    assert receipt["carrier_padding"]["count"] == 4
    assert receipt["mtp_sentinel"]["present"] is True


def test_model_repro_parameter_record_preserves_orientation_and_signed_zero():
    tensor = paddle.to_tensor([[0.0, -0.0], [1.0, 2.0]], dtype="float32")
    record = ModelReproObservationCallback._parameter_record(tensor)
    assert record["shape"] == [2, 2]
    assert record["dtype"] == "paddle.float32"
    assert record["positive_zero_count"] == 1
    assert record["negative_zero_count"] == 1
    assert record["sha256"] != record["transpose_sha256"]


def test_layer0_fine_forward_specs_are_explicit_and_fail_closed():
    assert ModelReproObservationCallback._forward_contract_specs("coarse") is None
    specs = ModelReproObservationCallback._forward_contract_specs("layer0_fine")
    assert len(specs) == 13
    assert specs["1.input_layernorm"] == "layer0_input_rmsnorm_output"
    assert specs["1.self_attn.q_a_proj"] == "layer0_q_down_projection_output"
    assert specs["1"] == "base_transformer_layer_0_output"
    with pytest.raises(ValueError, match="unsupported MODEL_REPRO_FORWARD_BOUNDARY_SET"):
        ModelReproObservationCallback._forward_contract_specs("unknown")


def test_first_tensor_prefers_first_tensor_in_nested_module_output():
    first = paddle.to_tensor([1.0])
    second = paddle.to_tensor([2.0])
    assert ModelReproObservationCallback._first_tensor((None, {"output": first}, second)) is first


def test_model_repro_observation_callback_writes_raw_loss(tmp_path):
    callback = ModelReproObservationCallback(raw_loss_path=str(tmp_path / "raw_loss.jsonl"))
    state = SimpleNamespace(global_step=1, is_world_process_zero=True)
    callback.on_log(SimpleNamespace(), state, SimpleNamespace(), logs={"mtp_0 loss": 2.5}, raw_loss=1.234567891)
    event = __import__("json").loads((tmp_path / "raw_loss.jsonl").read_text())
    assert event == {"step": 1, "loss": 1.234567891, "mtp_0_loss": 2.5}
