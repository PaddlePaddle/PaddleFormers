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
from unittest.mock import patch

from paddleformers.cli.train.sft.workflow import (
    ModelReproObservationCallback,
    apply_glm_moe_dsa_training_contract,
    load_tokenizer_and_processor,
)


def test_load_tokenizer_uses_independent_source():
    tokenizer = SimpleNamespace()
    model_args = SimpleNamespace(
        tokenizer_name_or_path="/tokenizer-only",
        model_name_or_path="/weights-only",
        stage="PT",
    )
    data_args = SimpleNamespace(processor_use_fast=None)

    with patch(
        "paddleformers.cli.train.sft.workflow.AutoTokenizer.from_pretrained",
        return_value=tokenizer,
    ) as load_tokenizer:
        actual_tokenizer, processor = load_tokenizer_and_processor(model_args, data_args)

    load_tokenizer.assert_called_once_with("/tokenizer-only")
    assert actual_tokenizer is tokenizer
    assert processor is tokenizer


def _base_training_args(**overrides):
    args = SimpleNamespace(
        num_nextn_predict_layers=1,
        mtp_num_layers=1,
        fp32_residual_connection=False,
        moe_token_dispatcher_type="alltoall",
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_model_parallel_size=1,
        sequence_parallel=True,
        moe_router_bias_update_rate=0.0,
        moe_expert_fusion=True,
        mtp_loss_scaling_factor=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_glm_moe_dsa_training_contract_applies_bias_activation_fusion_after_set_llm_config(monkeypatch, capsys):
    model_config = SimpleNamespace(model_type="glm_moe_dsa", bias_activation_fusion=True)
    training_args = _base_training_args()
    model_args = SimpleNamespace(mtp_attention_flexible=True, persist_layer_norm=False)
    data_args = SimpleNamespace()
    monkeypatch.setenv("MODEL_REPRO_BIAS_ACTIVATION_FUSION", "0")

    apply_glm_moe_dsa_training_contract(model_config, training_args, model_args, data_args)

    assert model_config.bias_activation_fusion is False
    captured = capsys.readouterr()
    assert "[BIAS-ACT-FUSION] model_config.bias_activation_fusion=False" in captured.out


def test_glm_moe_dsa_training_contract_does_not_require_pretokenized_dataset_field():
    model_config = SimpleNamespace(model_type="glm_moe_dsa")
    training_args = _base_training_args()
    model_args = SimpleNamespace(mtp_attention_flexible=True, persist_layer_norm=False)
    data_args = SimpleNamespace()

    apply_glm_moe_dsa_training_contract(model_config, training_args, model_args, data_args)

    assert model_config.mtp_num_layers == 1
    assert model_config.moe_expert_fusion is True
    assert model_config.moe_token_dispatcher_type == "alltoall"


def test_glm_moe_dsa_training_contract_applies_pp_p2p_needles_from_env(monkeypatch, capsys):
    model_config = SimpleNamespace(
        model_type="glm_moe_dsa",
        overlap_p2p_comm=True,
        batch_p2p_comm=None,
        variable_seq_lengths=False,
    )
    training_args = _base_training_args()
    model_args = SimpleNamespace(mtp_attention_flexible=True, persist_layer_norm=False)
    data_args = SimpleNamespace()
    monkeypatch.setenv("MODEL_REPRO_OVERLAP_P2P_COMM", "0")
    monkeypatch.setenv("MODEL_REPRO_BATCH_P2P_COMM", "1")
    monkeypatch.setenv("MODEL_REPRO_VARIABLE_SEQ_LENGTHS", "1")

    apply_glm_moe_dsa_training_contract(model_config, training_args, model_args, data_args)

    assert model_config.overlap_p2p_comm is False
    assert model_config.batch_p2p_comm is True
    assert model_config.variable_seq_lengths is True
    captured = capsys.readouterr()
    assert "[PP-P2P] model_config.overlap_p2p_comm=False" in captured.out
    assert "batch_p2p_comm=True" in captured.out
    assert "variable_seq_lengths=True" in captured.out


def test_glm_moe_dsa_training_contract_keeps_registered_mtp_loss_weight_when_cli_is_silent():
    model_config = SimpleNamespace(model_type="glm_moe_dsa", mtp_loss_scaling_factor=0.1)
    training_args = _base_training_args()
    model_args = SimpleNamespace(mtp_attention_flexible=True, persist_layer_norm=False)
    data_args = SimpleNamespace()

    apply_glm_moe_dsa_training_contract(model_config, training_args, model_args, data_args)

    assert model_config.mtp_loss_scaling_factor == 0.1


def test_machine_loss_payload_exposes_unrounded_losses_gate_field():
    events = [
        {"step": 1, "loss": 11.810652732849121},
        {"step": 2, "loss": 12.74130916595459},
    ]
    payload = ModelReproObservationCallback._machine_loss_payload(events)

    assert payload["losses"] == [11.810652732849121, 12.74130916595459]
    assert payload["steps"] == [1, 2]
    assert payload["framework"] == "paddle"
    assert payload["raw"] is True


def test_model_repro_observation_callback_writes_raw_loss(tmp_path):
    callback = ModelReproObservationCallback(raw_loss_path=str(tmp_path / "raw_loss.jsonl"))
    state = SimpleNamespace(global_step=1, is_world_process_zero=True)
    callback.on_log(SimpleNamespace(), state, SimpleNamespace(), logs={"mtp_0 loss": 2.5}, raw_loss=1.234567891)
    event = __import__("json").loads((tmp_path / "raw_loss.jsonl").read_text())
    assert event == {"step": 1, "loss": 1.234567891, "mtp_0_loss": 2.5}


def test_model_repro_observation_callback_writes_env_and_loss_json(tmp_path, monkeypatch):
    env_path = tmp_path / "env.json"
    loss_path = tmp_path / "loss.json"
    raw_path = tmp_path / "raw_loss.jsonl"
    monkeypatch.setenv("MODEL_REPRO_ENV_PATH", str(env_path))
    monkeypatch.setenv("MODEL_REPRO_LOSS_PATH", str(loss_path))
    monkeypatch.setenv("MODEL_REPRO_MODEL_ID", "zai-org/GLM-5.2")
    monkeypatch.setenv("MODEL_REPRO_MODEL_REVISION", "b4734de4facf877f85769a911abafc5283eab3d9")
    callback = ModelReproObservationCallback(raw_loss_path=str(raw_path))
    args = SimpleNamespace(bf16=True)
    state = SimpleNamespace(global_step=1, is_world_process_zero=True)
    control = SimpleNamespace()
    callback.on_train_begin(args, state, control)
    callback.on_log(args, state, control, logs={}, raw_loss=11.810652732849121)
    callback.on_train_end(args, state, control)
    env = __import__("json").loads(env_path.read_text())
    loss = __import__("json").loads(loss_path.read_text())
    assert env["framework"] == "paddle"
    assert env["device"] in {"cuda", "cpu"}
    assert env["dtype"] == "bfloat16"
    assert env["model_id"] == "zai-org/GLM-5.2"
    assert env["revision"] == "b4734de4facf877f85769a911abafc5283eab3d9"
    assert env["weights_loaded"] is True
    assert loss["losses"] == [11.810652732849121]
