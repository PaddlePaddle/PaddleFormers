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

from paddleformers.cli.train.sft import workflow as sft_workflow
from paddleformers.cli.train.sft.workflow import (
    apply_glm_moe_dsa_training_contract,
    load_tokenizer_and_processor,
)
from paddleformers.transformers.glm4_moe.modeling import Glm4MoePreTrainedModel


def test_load_tokenizer_uses_independent_source():
    tokenizer = SimpleNamespace()
    processor = SimpleNamespace()
    model_args = SimpleNamespace(
        tokenizer_name_or_path="/tokenizer-only",
        model_name_or_path="/weights-only",
        stage="PT",
    )
    data_args = SimpleNamespace(processor_use_fast=None)

    with patch.object(
        sft_workflow.AutoTokenizer,
        "from_pretrained",
        return_value=tokenizer,
    ) as load_tokenizer, patch.object(
        sft_workflow.AutoProcessor,
        "from_pretrained",
        return_value=processor,
    ) as load_processor:
        actual_tokenizer, actual_processor = load_tokenizer_and_processor(model_args, data_args)

    load_tokenizer.assert_called_once_with("/tokenizer-only")
    load_processor.assert_called_once_with("/weights-only", use_fast=None)
    assert actual_tokenizer is tokenizer
    assert actual_processor is processor


def test_load_processor_uses_autoprocessor_on_text_sft():
    tokenizer = SimpleNamespace()
    processor = SimpleNamespace()
    model_args = SimpleNamespace(
        tokenizer_name_or_path=None,
        model_name_or_path="/glm4-weights",
        stage="SFT",
    )
    data_args = SimpleNamespace(processor_use_fast=True)

    with patch.object(sft_workflow.AutoTokenizer, "from_pretrained", return_value=tokenizer,), patch.object(
        sft_workflow.AutoProcessor,
        "from_pretrained",
        return_value=processor,
    ) as load_processor:
        actual_tokenizer, actual_processor = load_tokenizer_and_processor(model_args, data_args)

    load_processor.assert_called_once_with("/glm4-weights", use_fast=True)
    assert actual_tokenizer is tokenizer
    assert actual_processor is processor
    assert actual_processor is not tokenizer


def test_load_processor_falls_back_to_tokenizer_without_processor_files():
    tokenizer = SimpleNamespace()
    model_args = SimpleNamespace(
        tokenizer_name_or_path="/tokenizer-only",
        model_name_or_path="/extracted-GLM-5.2-weights",
        stage="SFT",
    )
    data_args = SimpleNamespace(processor_use_fast=None)

    with patch.object(sft_workflow.AutoTokenizer, "from_pretrained", return_value=tokenizer,), patch.object(
        sft_workflow.AutoProcessor,
        "from_pretrained",
        side_effect=ValueError("Unrecognized processing class"),
    ):
        actual_tokenizer, actual_processor = load_tokenizer_and_processor(model_args, data_args)

    assert actual_tokenizer is tokenizer
    assert actual_processor is tokenizer


def test_load_processor_reraises_missing_processor_on_glm4_checkpoint():
    tokenizer = SimpleNamespace()
    model_args = SimpleNamespace(
        tokenizer_name_or_path=None,
        model_name_or_path="/glm4-weights",
        stage="SFT",
    )
    data_args = SimpleNamespace(processor_use_fast=None)

    with patch.object(sft_workflow.AutoTokenizer, "from_pretrained", return_value=tokenizer,), patch.object(
        sft_workflow.AutoProcessor,
        "from_pretrained",
        side_effect=ValueError("Unrecognized processing class"),
    ):
        try:
            load_tokenizer_and_processor(model_args, data_args)
        except ValueError as exc:
            assert "Unrecognized processing class" in str(exc)
        else:
            raise AssertionError("GLM-4 AutoProcessor failure must not fall back to tokenizer")


def test_glm4_moe_aoa_keeps_gate_weight_float32_under_uac():
    config = SimpleNamespace(
        using_sonic_moe=False,
        n_routed_experts=4,
        num_hidden_layers=2,
        first_k_dense_replace=1,
        mtp_num_layers=0,
        num_nextn_predict_layers=0,
        num_attention_heads=8,
        num_key_value_heads=8,
        tie_word_embeddings=False,
        use_qk_norm=False,
        attention_bias=False,
        use_accuracy_compatible=True,
        moe_expert_fusion=False,
    )
    config.get = lambda key, default=False: default
    statements = Glm4MoePreTrainedModel._gen_aoa_config(config)["aoa_statements"]
    joined = "\n".join(statements)
    assert "mlp.gate.weight, dtype='float32'" in joined
    assert "mlp.gate.weight, dtype='bfloat16'" not in joined


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


def test_glm_moe_dsa_training_contract_does_not_require_pretokenized_dataset_field():
    model_config = SimpleNamespace(model_type="glm_moe_dsa")
    training_args = _base_training_args()
    model_args = SimpleNamespace(mtp_attention_flexible=True, persist_layer_norm=False)
    data_args = SimpleNamespace()

    apply_glm_moe_dsa_training_contract(model_config, training_args, model_args, data_args)

    assert model_config.num_nextn_predict_layers == 1
    assert not hasattr(model_config, "mtp_num_layers")
    assert training_args.mtp_num_layers == 0
    assert model_config.moe_expert_fusion is True
    assert model_config.moe_token_dispatcher_type == "alltoall"


def test_glm_moe_dsa_training_contract_copies_pp_p2p_from_training_args():
    model_config = SimpleNamespace(
        model_type="glm_moe_dsa",
        overlap_p2p_comm=True,
        batch_p2p_comm=None,
        variable_seq_lengths=False,
    )
    training_args = _base_training_args(overlap_p2p_comm=False, batch_p2p_comm=True)
    model_args = SimpleNamespace(mtp_attention_flexible=True, persist_layer_norm=False)
    data_args = SimpleNamespace()

    apply_glm_moe_dsa_training_contract(model_config, training_args, model_args, data_args)

    assert model_config.overlap_p2p_comm is False
    assert model_config.batch_p2p_comm is True
    assert model_config.variable_seq_lengths is False


def test_glm_moe_dsa_training_contract_copies_variable_seq_lengths_from_training_args():
    model_config = SimpleNamespace(
        model_type="glm_moe_dsa",
        overlap_p2p_comm=True,
        batch_p2p_comm=None,
        variable_seq_lengths=False,
    )
    training_args = _base_training_args(variable_seq_lengths=True)
    model_args = SimpleNamespace(mtp_attention_flexible=True, persist_layer_norm=False)
    data_args = SimpleNamespace()

    apply_glm_moe_dsa_training_contract(model_config, training_args, model_args, data_args)

    assert model_config.variable_seq_lengths is True


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


def test_glm_moe_dsa_training_contract_applies_bias_activation_fusion_env(monkeypatch, capsys):
    model_config = SimpleNamespace(model_type="glm_moe_dsa", bias_activation_fusion=True)
    training_args = _base_training_args()
    model_args = SimpleNamespace(mtp_attention_flexible=True, persist_layer_norm=False)
    data_args = SimpleNamespace()

    monkeypatch.setenv("MODEL_REPRO_BIAS_ACTIVATION_FUSION", "0")
    apply_glm_moe_dsa_training_contract(model_config, training_args, model_args, data_args)

    assert model_config.bias_activation_fusion is False
    captured = capsys.readouterr()
    assert "[BIAS-ACT-FUSION] model_config.bias_activation_fusion=False" in captured.out
