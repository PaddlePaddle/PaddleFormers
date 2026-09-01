# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Training Ernie Model."""

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from functools import partial
from pathlib import Path

import numpy as np
import paddle

from paddleformers.cli.utils.process import add_new_special_tokens
from paddleformers.data.causal_dataset import (
    build_train_valid_test_datasets,
    check_data_split,
)
from paddleformers.data.indexed_dataset import SFTMMapIndexedDatasetBuilder
from paddleformers.datasets.collate import collate_fn, mm_collate_fn
from paddleformers.datasets.data_utils import estimate_training
from paddleformers.datasets.loader import create_dataset as create_dataset_sft
from paddleformers.datasets.loader import create_indexed_dataset
from paddleformers.datasets.SFTDataset import TextSequence
from paddleformers.datasets.template.template import get_template_and_fix_tokenizer
from paddleformers.nn.attention import AttentionInterface
from paddleformers.peft import LoRAConfig, LoRAModel
from paddleformers.trainer import (
    FP8QuantWeightCallback,
    IntervalStrategy,
    MoECorrectionBiasAdjustCallback,
    MoeExpertsGradScaleCallback,
    MoEGateSpGradSyncCallBack,
    MoEQuantileBalancingCallback,
    RuntimeTimer,
    TrainerCallback,
    get_last_checkpoint,
    set_random_seed,
    set_seed,
)
from paddleformers.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForCausalLMPipe,
    AutoModelForConditionalGeneration,
    AutoModelForConditionalGenerationPipe,
    AutoProcessor,
    AutoTokenizer,
    Llama3Tokenizer,
    LlamaTokenizer,
)
from paddleformers.transformers.configuration_utils import (
    LlmMetaConfig,
    QuantizationConfig,
)
from paddleformers.utils.log import logger

from .make_data_utils import DataGenerator
from .sft_trainer import SFTTrainer

# Fine-tune Environment Variables to support sharding stage1 overlap optimization.
os.environ["USE_CASUAL_MASK"] = "False"

from paddleformers.cli.hparams import (
    DataArguments,
    FinetuningArguments,
    GeneratingArguments,
    ModelArguments,
)

# Imported from the submodule directly: hparams/__init__ lazily exposes only the
# dataclasses listed in its import_structure, which does not include this one.
from paddleformers.cli.hparams.preprocess_args import End2EndProcessorArguments
from paddleformers.cli.utils import (
    freeze_model_parameters,
    get_lora_target_modules,
    get_multimodel_lora_target_modules,
)


class ModelReproObservationCallback(TrainerCallback):
    """Rank-zero loss.json / env.json writer for formal model-reproduction runs.

    Dump hooks, parameter inventories, and forward/backward receipts stay in
    Explore trees. This callback only writes the acceptance-oracle artifacts.
    """

    def __init__(self, raw_loss_path=None):
        self.raw_loss_path = raw_loss_path
        self.env_path = os.environ.get("MODEL_REPRO_ENV_PATH")
        self.loss_path = os.environ.get("MODEL_REPRO_LOSS_PATH")
        self._loss_events = []

    @staticmethod
    def _sha256_file(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def _write_json(path, payload):
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _normalized_device():
        """Return the device class the benchmark checker expects, not the GPU model name."""
        return "cuda" if paddle.device.is_compiled_with_cuda() else "cpu"

    @staticmethod
    def _normalized_dtype(args):
        return "bfloat16" if getattr(args, "bf16", False) else "float32"

    @staticmethod
    def _machine_loss_payload(events, raw_path=None, source_sha256=None):
        """Return the machine loss artifact.

        ``losses`` is the benchmark gate field: an unrounded main-loss series with
        one entry per recorded step. ``events`` keeps per-step diagnostic detail.
        """
        return {
            "schema": "glm52-machine-loss/v1",
            "framework": "paddle",
            "raw": True,
            "owning_cli_exit_code": 0,
            "losses": [event["loss"] for event in events if "loss" in event],
            "event_count": len(events),
            "steps": [event["step"] for event in events],
            "events": events,
            "source": str(raw_path) if raw_path else None,
            "source_sha256": source_sha256,
        }

    @classmethod
    def _environment_payload(cls, args):
        config_path = os.environ.get("MODEL_REPRO_MODEL_CONFIG_PATH")
        try:
            nccl_package = importlib.metadata.version("nvidia-nccl-cu12")
        except importlib.metadata.PackageNotFoundError:
            nccl_package = None
        return {
            "schema": "glm52-environment/v1",
            "framework": "paddle",
            "framework_version": paddle.__version__,
            "python_version": platform.python_version(),
            "device": cls._normalized_device(),
            "device_name": paddle.device.cuda.get_device_name(0) if paddle.device.is_compiled_with_cuda() else None,
            "dtype": cls._normalized_dtype(args),
            "cuda": paddle.version.cuda(),
            "cudnn": paddle.version.cudnn(),
            "nccl_package": nccl_package,
            "model_id": os.environ.get("MODEL_REPRO_MODEL_ID"),
            "revision": os.environ.get("MODEL_REPRO_MODEL_REVISION"),
            "model_config_sha256": cls._sha256_file(config_path) if config_path else None,
            "weights_loaded": True,
            "world_size": paddle.distributed.get_world_size() if paddle.distributed.is_initialized() else 1,
        }

    @staticmethod
    def _is_writer(state):
        return bool(getattr(state, "is_world_process_zero", False))

    def on_train_begin(self, args, state, control, **kwargs):
        if not self._is_writer(state):
            return
        if self.raw_loss_path:
            raw_path = Path(self.raw_loss_path).expanduser().resolve()
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.unlink(missing_ok=True)
        if self.env_path:
            self._write_json(self.env_path, self._environment_payload(args))

    def on_train_end(self, args, state, control, **kwargs):
        if not self.loss_path or not self._is_writer(state):
            return
        raw_path = Path(self.raw_loss_path).expanduser().resolve() if self.raw_loss_path else None
        payload = self._machine_loss_payload(
            self._loss_events,
            raw_path=raw_path,
            source_sha256=self._sha256_file(raw_path) if raw_path and raw_path.is_file() else None,
        )
        self._write_json(self.loss_path, payload)

    def on_log(self, args, state, control, logs=None, raw_loss=None, **kwargs):
        if not self.raw_loss_path or raw_loss is None or not self._is_writer(state):
            return
        event = {"step": int(state.global_step), "loss": float(raw_loss)}
        for key, value in (logs or {}).items():
            normalized_key = key.replace(" ", "_")
            if normalized_key.startswith("mtp_") and normalized_key.endswith("_loss"):
                event[normalized_key] = float(value)
        path = os.path.abspath(os.path.expanduser(self.raw_loss_path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self._loss_events.append(event)


def load_tokenizer_and_processor(model_args, data_args):
    tokenizer_path = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    if "VL" in model_args.stage:
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, use_fast=data_args.processor_use_fast)
    else:
        processor = tokenizer
    return tokenizer, processor


def apply_glm_moe_dsa_training_contract(model_config, training_args, model_args, data_args):
    """Propagate CLI training semantics into the Fleet provider used by GLM MoE DSA."""
    if getattr(model_config, "model_type", None) != "glm_moe_dsa":
        return

    requested_mtp = int(getattr(training_args, "num_nextn_predict_layers", 0) or 0)
    explicit_mtp = int(getattr(training_args, "mtp_num_layers", 0) or 0)
    if requested_mtp and explicit_mtp and requested_mtp != explicit_mtp:
        raise ValueError(
            f"GLM MoE DSA MTP depth mismatch: num_nextn_predict_layers={requested_mtp}, "
            f"mtp_num_layers={explicit_mtp}"
        )
    mtp_depth = explicit_mtp or requested_mtp
    model_config.num_nextn_predict_layers = mtp_depth
    model_config.mtp_num_layers = mtp_depth
    training_args.num_nextn_predict_layers = mtp_depth
    training_args.mtp_num_layers = mtp_depth
    model_config.mtp_enabled = mtp_depth > 0

    requested_mtp_loss_scaling_factor = getattr(training_args, "mtp_loss_scaling_factor", None)
    if requested_mtp_loss_scaling_factor is not None:
        model_config.mtp_loss_scaling_factor = float(requested_mtp_loss_scaling_factor)
    logger.info(
        "GLM MoE DSA MTP loss weight: mtp_loss_scaling_factor="
        f"{getattr(model_config, 'mtp_loss_scaling_factor', None)} "
        f"(cli={requested_mtp_loss_scaling_factor!r}, mtp_depth={mtp_depth})"
    )

    if getattr(data_args, "pretokenized_dataset", False) and mtp_depth > 0 and not model_args.mtp_attention_flexible:
        raise ValueError("pretokenized GLM MoE DSA MTP requires mtp_attention_flexible=true")

    model_config.fp32_residual_connection = training_args.fp32_residual_connection
    model_config.moe_token_dispatcher_type = getattr(training_args, "moe_token_dispatcher_type", "alltoall")
    model_config.moe_router_bias_update_rate = float(getattr(training_args, "moe_router_bias_update_rate", 0.001))
    _moe_expert_fusion = getattr(training_args, "moe_expert_fusion", None)
    _moe_expert_fusion_env = os.environ.get("MODEL_REPRO_MOE_FUSION", None)
    if _moe_expert_fusion_env is not None:
        model_config.moe_expert_fusion = _moe_expert_fusion_env == "1"
    elif _moe_expert_fusion is not None:
        model_config.moe_expert_fusion = bool(_moe_expert_fusion)
    # CLI dataclass has no bias_activation_fusion field; set_llm_config() would
    # leave the GLMMoEModelProvider default True. Honour the env after that call.
    _bias_activation_fusion_env = os.environ.get("MODEL_REPRO_BIAS_ACTIVATION_FUSION", None)
    if _bias_activation_fusion_env is not None:
        model_config.bias_activation_fusion = _bias_activation_fusion_env == "1"
        print(
            "[BIAS-ACT-FUSION] model_config.bias_activation_fusion=" f"{model_config.bias_activation_fusion}",
            flush=True,
        )
    # YAML overlap_p2p_comm / batch_p2p_comm land on TrainingArguments
    # (pipeline runtime). Copy them onto GLMMoEModelProvider after
    # set_llm_config. Do not copy variable_seq_lengths: that YAML flag
    # drives pipeline enable_dynamic_shape, while the provider default
    # stays False.

    def _env_bool(name):
        value = os.environ.get(name)
        if value is None:
            return None
        return value == "1"

    _overlap_p2p_comm = _env_bool("MODEL_REPRO_OVERLAP_P2P_COMM")
    if _overlap_p2p_comm is None:
        _overlap_p2p_comm = getattr(training_args, "overlap_p2p_comm", None)
    if _overlap_p2p_comm is not None:
        model_config.overlap_p2p_comm = bool(_overlap_p2p_comm)
    _batch_p2p_comm = _env_bool("MODEL_REPRO_BATCH_P2P_COMM")
    if _batch_p2p_comm is None:
        _batch_p2p_comm = getattr(training_args, "batch_p2p_comm", None)
    if _batch_p2p_comm is not None:
        model_config.batch_p2p_comm = bool(_batch_p2p_comm)
    if any(field is not None for field in (_overlap_p2p_comm, _batch_p2p_comm)):
        print(
            "[PP-P2P] model_config.overlap_p2p_comm="
            f"{getattr(model_config, 'overlap_p2p_comm', None)} "
            f"batch_p2p_comm={getattr(model_config, 'batch_p2p_comm', None)}",
            flush=True,
        )
    for parallel_field in (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
    ):
        configured_size = int(getattr(training_args, parallel_field, -1))
        setattr(model_config, parallel_field, max(configured_size, 1))
    model_config.sequence_parallel = bool(getattr(training_args, "sequence_parallel", False))
    configured_expert_tensor_parallel_size = int(getattr(training_args, "expert_tensor_model_parallel_size", -1))
    expert_tensor_parallel_size = (
        1 if configured_expert_tensor_parallel_size == -1 else configured_expert_tensor_parallel_size
    )
    if expert_tensor_parallel_size < 1:
        raise ValueError(
            "GLM MoE DSA expert_tensor_model_parallel_size must be -1 or at least 1, "
            f"got {configured_expert_tensor_parallel_size}"
        )
    model_config.expert_tensor_parallel_size = expert_tensor_parallel_size
    if model_args.persist_layer_norm is not None:
        model_config.persist_layer_norm = model_args.persist_layer_norm


def freeze_param_except_mtp(model, config):
    logger.info("freeze_param_except_mtp.")

    def extract_layer_idx(text):
        match = re.search(r"model.layers.(-?\d+\.?\d*)", text)
        if match:
            num_str = match.group(1)
            if "." in num_str:
                return float(num_str)
            else:
                return int(num_str)
        return None

    # not sure can work on all model
    jackpot = set(range(config.num_hidden_layers, config.num_hidden_layers + config.mtp_num_layers))
    for name, param in model.state_dict().items():
        layer_idx = extract_layer_idx(name)
        is_mtp = layer_idx in jackpot
        if not is_mtp:
            param.stop_gradient = True
        else:
            param.stop_gradient = False


def create_pretrained_dataset(training_args, data_args, model_args):
    assert data_args.input_dir is not None and len(data_args.input_dir.split()) > 1

    check_data_split(
        data_args.split,
        training_args.do_train,
        training_args.do_eval,
        training_args.do_predict,
    )

    if training_args.max_steps < 0:
        raise ValueError(
            f"max_steps mush be larger than 0 when using pretrain offline dataset, but get {training_args.max_steps}."
        )

    train_val_test_num_samples = [
        training_args.per_device_train_batch_size
        * training_args.dataset_world_size
        * training_args.max_steps
        * training_args.gradient_accumulation_steps,
        training_args.per_device_eval_batch_size
        * training_args.dataset_world_size
        * training_args.eval_iters
        * (training_args.max_steps // training_args.eval_steps + 1),
        training_args.per_device_eval_batch_size * training_args.dataset_world_size * training_args.test_iters,
    ]

    train_dataset, valid_dataset, test_dataset = build_train_valid_test_datasets(
        data_prefix=data_args.input_dir.split(),
        data_impl="mmap",
        splits_string=data_args.split,
        train_val_test_num_samples=train_val_test_num_samples,
        seq_length=data_args.max_seq_len + training_args.num_nextn_predict_layers,
        seed=training_args.seed,
        skip_warmup=True,
        data_cache_path=None,
    )

    from paddleformers.data import Stack

    def _collate_data(batch, stack_fn=Stack()):
        input_keys = ["input_ids", "labels", "position_ids", "attn_mask_startend_row_indices"]
        return_list = []
        for batch_sequence in batch:
            # tokens
            padded_token_ids = np.array([batch_sequence["text"][:-1]])
            # labels
            padded_labels = np.array([batch_sequence["text"][1:]])
            # position_ids
            padded_position_ids = np.array([sum(batch_sequence["position_ids"], [])[:-1]])
            return_list.append(
                [
                    padded_token_ids,
                    padded_labels,
                    padded_position_ids,
                ]
            )
            # attn mask
            oral_position_ids = batch_sequence["position_ids"]
            from paddleformers.datasets.collate import (
                gen_attn_mask_startend_row_indices,
            )

            return_list[-1].append(
                gen_attn_mask_startend_row_indices(
                    oral_position_ids,
                    data_args.max_seq_len + training_args.num_nextn_predict_layers,
                    model_args.use_global_causal_attn,
                )[:, :, :-1, :]
            )

        return_list = [np.concatenate(tensor_list) for tensor_list in zip(*return_list)]
        input_dict = dict(zip(input_keys, return_list))
        return input_dict

    return train_dataset, valid_dataset, test_dataset, _collate_data


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    generating_args: "GeneratingArguments",
    finetuning_args: "FinetuningArguments",
    preprocess_args: "End2EndProcessorArguments" = None,
):
    """_summary_

    Args:
        model_args (ModelArguments): _description_
        data_args (DataArguments): _description_
        generating_args (GeneratingArguments): _description_
        finetuning_args (FinetuningArguments): _description_
        preprocess_args (End2EndProcessorArguments): carries the multimodal
            resolution bounds (``max_pixels`` / ``min_pixels``). Optional so that
            callers that do not parse it keep working.
        callbacks (Optional[list[&quot;TrainerCallback&quot;]], optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_
    """

    training_args = finetuning_args
    training_args.max_seq_len = data_args.max_seq_len
    training_args.model_name_or_path = model_args.model_name_or_path
    training_args.download_hub = model_args.download_hub
    training_args.copy_custom_file_list = model_args.copy_custom_file_list

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    training_args.print_config(training_args, "Train")

    if training_args.pre_alloc_memory > 0:
        memory_size = int(training_args.pre_alloc_memory * 1024 * 1024 * 1024)
        x = paddle.empty([memory_size], dtype=paddle.uint8)
        logger.info(f"pre_alloc_memory size {x.shape}")
        del x

    # Setup GPU & distributed training
    paddle.set_device(training_args.device)
    set_random_seed(seed_=training_args.seed)
    set_seed(seed=training_args.seed)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Load model
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        elif training_args.bf16:
            dtype = "bfloat16"
        else:
            raise ValueError("Please specific dtype: --fp16 or --bf16")
    else:
        dtype = "float32"

    if finetuning_args.weight_quantize_algo is not None:
        quantization_config = dict(
            weight_quantize_algo=finetuning_args.weight_quantize_algo,
            ignore_modules=[".*out_linear.*"],
        )
    else:
        quantization_config = dict(weight_quantize_algo=finetuning_args.weight_quantize_algo)
    quantization_config = QuantizationConfig.from_dict(quantization_config)

    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        dtype=dtype,
        quantization_config=quantization_config,
    )
    if getattr(training_args, "pad_token_id", None) is not None:
        model_config.pad_token_id = training_args.pad_token_id

    if (
        model_config.tie_word_embeddings
        and model_config.quantization_config.is_weight_quantize()
        and training_args.pipeline_model_parallel_size > 1
    ):
        raise ValueError(
            "Tie-weight model is not supported quantization in pipeline parallel mode. But got pipeline_model_parallel_size: {}".format(
                training_args.pipeline_model_parallel_size
            )
        )

    architectures_to_check = {"Qwen2Moe", "DeepseekV2", "DeepseekV3"}
    if (
        any(architecture in str(model_config.architectures) for architecture in architectures_to_check)
        and training_args.data_parallel_size > 1
        and not training_args.use_expert_parallel
    ):
        raise ValueError("Please set use_expert_parallel to true in expert parallel mode.")

    # (Liuting) Not support acc calculation now due to MTP.
    if "DeepseekV3" in str(model_config.architectures):
        training_args.prediction_loss_only = True

    LlmMetaConfig.set_llm_config(model_config, training_args)
    apply_glm_moe_dsa_training_contract(model_config, training_args, model_args, data_args)
    model_config.use_fast_layer_norm = model_args.use_fast_layer_norm

    # autoregressive mtp training
    if model_config.mtp_num_layers > 1:
        tmp = model_config.mtp_num_layers
        model_config.mtp_num_layers = model_config.num_nextn_predict_layers
        model_config.num_nextn_predict_layers = tmp

        tmp = training_args.mtp_num_layers
        training_args.mtp_num_layers = training_args.num_nextn_predict_layers
        training_args.num_nextn_predict_layers = tmp

        logger.info(
            f"MTP args changing for autoregressive mtp training, mtp_num_layers: {model_config.mtp_num_layers}, num_nextn_predict_layers: {model_config.num_nextn_predict_layers}!!"
        )

    # Config for model using dropout, such as GPT.
    if hasattr(model_config, "hidden_dropout_prob"):
        model_config.hidden_dropout_prob = finetuning_args.hidden_dropout_prob
    if hasattr(model_config, "attention_probs_dropout_prob"):
        model_config.attention_probs_dropout_prob = finetuning_args.attention_probs_dropout_prob
    if hasattr(model_config, "ignore_index"):
        model_config.ignore_index = -100

    avaible_attn_impl = AttentionInterface._global_mapping.keys()
    if model_args._attn_implementation not in avaible_attn_impl:
        raise ValueError(
            f"Invalid _attn_implementation: {model_args._attn_implementation}, available _attn_implementation: {avaible_attn_impl}"
        )

    model_config.pp_seg_method = model_args.pp_seg_method
    model_config.seq_length = data_args.max_seq_len
    model_config.max_sequence_length = data_args.max_seq_len
    model_config._attn_implementation = model_args._attn_implementation
    model_config.is_lora = model_args.lora
    model_config.moe_logging = model_args.moe_logging

    # Sync arguments to MLLM sub_config
    if getattr(model_config, "text_config", None) is not None:
        LlmMetaConfig.set_llm_config(model_config.text_config, training_args)
        model_config.text_config._attn_implementation = model_args._attn_implementation
        model_config.text_config.max_sequence_length = data_args.max_seq_len
        if hasattr(model_config.text_config, "mtp_num_hidden_layers"):
            model_config.text_config.mtp_num_hidden_layers = getattr(training_args, "num_nextn_predict_layers", 0)
    if getattr(model_config, "vision_config", None) is not None:
        model_config.vision_config._attn_implementation = model_args._attn_implementation
        model_config.vision_config.recompute_granularity = model_config.recompute_granularity
        model_config.vision_config.recompute_method = model_config.recompute_method
        model_config.vision_config.recompute_num_layers = model_config.recompute_num_layers
        # recompute_granularity="selective" requires recompute_modules to be set,
        # otherwise the vision TransformerConfig.__post_init__ assertion fails.
        model_config.vision_config.recompute_modules = getattr(model_config, "recompute_modules", None)

    # Sync freeze_config to model_config so that Fleet model providers can read it
    freeze_config = getattr(training_args, "freeze_config", "")
    if freeze_config:
        model_config.freeze_vision_model = "freeze_vision" in freeze_config
        model_config.freeze_language_model = "freeze_llm" in freeze_config
        model_config.freeze_vision_projection = "freeze_aligner" in freeze_config

    # Sync enable_auto_parallel to model_config for Fleet to access
    model_config.enable_auto_parallel = training_args.enable_auto_parallel

    logger.info(f"Final model config: {model_config}")
    logger.info("Creating model")

    if data_args.make_offline_data:
        logger.info("Making offline data..., model is not loaded!")
        logger.info(f"Training data: {data_args.train_dataset_path}")
    else:
        logger.info(f"Loading model weights from {model_args.model_name_or_path}")
        if "VL" in model_args.stage:
            model_class = AutoModelForConditionalGeneration
            if training_args.pipeline_model_parallel_size > 1:
                if data_args.eval_with_do_generation and training_args.do_eval:
                    raise ValueError("Please set eval_with_do_generation to false in pipeline parallel mode.")
                model_class = AutoModelForConditionalGenerationPipe
        else:
            model_class = AutoModelForCausalLM
            if training_args.pipeline_model_parallel_size > 1:
                if data_args.eval_with_do_generation and training_args.do_eval:
                    raise ValueError("Please set eval_with_do_generation to false in pipeline parallel mode.")
                model_class = AutoModelForCausalLMPipe

        if model_args.continue_training and not training_args.autotuner_benchmark:
            model = model_class.from_pretrained(
                model_args.model_name_or_path,
                config=model_config,
                convert_from_hf=training_args.convert_from_hf,
                load_via_cpu=training_args.load_via_cpu,
                load_checkpoint_format=training_args.load_checkpoint_format,
            )
        else:
            model = model_class.from_config(model_config, dtype=dtype)

        if training_args.do_train and model_args.neftune:
            # Inspired by https://github.com/neelsjain/NEFTune
            if hasattr(model, "get_input_embeddings"):

                def neft_post_hook(module, input, output):
                    if module.training:
                        mag_norm = model_args.neftune_noise_alpha / paddle.sqrt(
                            paddle.to_tensor(output.shape[0] * output.shape[1], dtype="float32")
                        )
                        output = output + paddle.uniform(
                            shape=output.shape, dtype=output.dtype, min=-mag_norm, max=mag_norm
                        )
                    return output

                neft_post_hook_handle = model.get_input_embeddings().register_forward_post_hook(neft_post_hook)
            else:
                raise NotImplementedError("Only support neftune for model with get_input_embeddings")

    runtime_timer = RuntimeTimer("Creating SFT MapDataset")

    # Load tokenizer & processor & dataset
    tokenizer, processor = load_tokenizer_and_processor(model_args, data_args)
    add_new_special_tokens(tokenizer, data_args.new_special_tokens_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # if using chat_template, data_args.eval_with_do_generation must be false
    if tokenizer.chat_template is not None:
        data_args.eval_with_do_generation = False

    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, Llama3Tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Text GLM SFT keeps processor=tokenizer (load_tokenizer_and_processor).
    # VL stages still need the develop max/min pixel wiring on AutoProcessor.
    if preprocess_args is not None and processor is not tokenizer:
        if preprocess_args.max_pixels is not None:
            processor.image_max_pixels = preprocess_args.max_pixels
        if preprocess_args.min_pixels is not None:
            processor.image_min_pixels = preprocess_args.min_pixels

    type_map = {"bf16": "bfloat16", "fp16": "float16"}
    compute_type = type_map.get(training_args.compute_type, "float32")
    dataset_config = {
        "tokenizer": tokenizer,
        "processor": processor,
        "max_seq_len": data_args.max_seq_len,
        "random_seed": training_args.seed,
        "num_replicas": training_args.dataset_world_size,
        "rank": training_args.dataset_rank,
        "num_samples_each_epoch": data_args.num_samples_each_epoch,
        "random_shuffle": data_args.random_shuffle,
        "greedy_intokens": data_args.greedy_intokens,
        "packing": data_args.packing,
        "mix_strategy": data_args.mix_strategy,
        "encode_one_turn": data_args.encode_one_turn,
        "use_template": data_args.use_template,
        "is_pretraining": True if "pt" in model_args.stage.lower() else False,
        "truncate_packing": data_args.truncate_packing,
        "stage": model_args.stage,
        "template_backend": data_args.template_backend,
        "split_multi_turn": data_args.split_multi_turn,
        "dataset_type": data_args.dataset_type,
        "dtype": compute_type,
        "dataset_num_proc": finetuning_args.dataset_num_proc,
        "binpacking": data_args.binpacking,
        "packing_interval": data_args.packing_interval,
        "packed_idx_cache_dir": data_args.packed_idx_cache_dir,
        "dataloader_num_workers": training_args.dataloader_num_workers,
        "template": data_args.template,
        # YAML enable_thinking lands on GeneratingArguments. glm5_2 is a
        # ReasoningTemplate; without this copy it stays register_template default True.
        "enable_thinking": getattr(generating_args, "enable_thinking", None),
        "tool_format": None,
        "default_system": None,
        "truncation_strategy": data_args.truncation_strategy,
        "skip_warmup": data_args.skip_warmup,
    }

    if dataset_config["template_backend"] == "custom":
        template_instance = get_template_and_fix_tokenizer(dataset_config)
    else:
        template_instance = None
    dataset_config.update(
        {
            "template_instance": template_instance,
        }
    )
    # make offline dataset
    if data_args.make_offline_data:
        import time

        if tokenizer.vocab_size < 2**16 - 1:
            save_dtype = np.uint16
        else:
            save_dtype = np.int32
        dataclass = TextSequence

        global_batch_size = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * max(training_args.data_parallel_size, 1)
            * max(training_args.sharding_parallel_size, 1)
        )

        logger.info(f"training_args.per_device_train_batch_size: {training_args.per_device_train_batch_size}")
        logger.info(f"training_args.gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
        logger.info(f"training_args.data_parallel_size: {training_args.data_parallel_size}")
        logger.info(f"training_args.sharding_parallel_size: {training_args.sharding_parallel_size}")
        logger.info(f"global_batch_size: {global_batch_size}")

        def fetch_and_serialize(generator, dtype):
            sample = next(generator)
            result = []
            for sequence in sample:
                serialized = []
                for key in train_builder._data_file_dict.keys():
                    tensor = np.array(getattr(sequence, key), dtype=dtype)
                    serialized.append((key, tensor.tobytes(order="C"), tensor.size))
                result.append(serialized)
            return result

        if (
            training_args.do_train
            and data_args.train_dataset_path
            and training_args.should_load_dataset
            and paddle.distributed.get_rank() == 0
        ):
            runtime_timer.start("Create SFT Train MapDataset")
            os.makedirs(os.path.join(data_args.dataset_output_dir, "train"), exist_ok=True)

            train_output_idx_files = os.path.join(data_args.dataset_output_dir, "train", "index.idx")
            train_dataset = create_dataset_sft(
                task_group=data_args.train_dataset_path,
                task_group_prob=data_args.train_dataset_prob,
                sub_dataset_type=data_args.train_dataset_type,
                **dataset_config,
            )
            output_file_dict = {}
            train_dir = os.path.join(data_args.dataset_output_dir, "train")
            index_file = os.path.join(data_args.dataset_output_dir, "train", "index.idx")
            for field in fields(dataclass):
                output_path = os.path.join(train_dir, f"{field.name}.bin")
                output_file_dict[field.name] = output_path
            train_builder = SFTMMapIndexedDatasetBuilder(output_file_dict, save_dtype, index_file=index_file)
            train_sample_generator = DataGenerator(train_dataset)
            count = 0
            start_time = time.time()

            with ThreadPoolExecutor(max_workers=2) as executor:
                future = executor.submit(fetch_and_serialize, train_sample_generator, save_dtype)
                while not train_dataset.iter_all_examples:
                    serialized_sequences = future.result()
                    future = executor.submit(fetch_and_serialize, train_sample_generator, save_dtype)
                    if train_dataset.iter_all_examples:
                        break
                    for serialized in serialized_sequences:
                        train_builder.add_item_bytes(serialized)
                    train_builder.end_document()
                    count += 1
                    if count % 1000 == 0:
                        logger.info(
                            f"Processed {count} samples in {time.time() - start_time:.2f} seconds, average speed: {count / (time.time() - start_time):.2f} samples/second"
                        )
            train_builder.finalize(train_output_idx_files)
            logger.info(f"{runtime_timer.log()}")

        if (
            training_args.do_eval
            and data_args.eval_dataset_path
            and training_args.should_load_dataset
            and paddle.distributed.get_rank() == 0
        ):
            runtime_timer.start("Create SFT Eval MapDataset")
            os.makedirs(os.path.join(data_args.dataset_output_dir, "eval"), exist_ok=True)

            eval_output_idx_files = os.path.join(data_args.dataset_output_dir, "eval", "index.idx")
            eval_dataset = create_dataset_sft(
                task_group=data_args.eval_dataset_path,
                task_group_prob=data_args.eval_dataset_prob,
                sub_dataset_type=data_args.eval_dataset_type,
                is_valid=True,
                **dataset_config,
            )
            output_file_dict = {}
            eval_dir = os.path.join(data_args.dataset_output_dir, "eval")
            index_file = os.path.join(data_args.dataset_output_dir, "eval", "index.idx")
            for field in fields(dataclass):
                output_path = os.path.join(eval_dir, f"{field.name}.bin")
                output_file_dict[field.name] = output_path
            eval_builder = SFTMMapIndexedDatasetBuilder(output_file_dict, save_dtype, index_file=index_file)
            for sequences in eval_dataset:
                for sequence in sequences:
                    eval_builder.add_item(sequence)
                eval_builder.end_document()
            eval_builder.finalize(eval_output_idx_files)
            logger.info(f"{runtime_timer.log()}")
        logger.info("Make SFT Offline DataSet Done.")
        return

    if data_args.dataset_type == "pretrain":
        training_args.test_iters = training_args.eval_iters * 10
        train_dataset, eval_dataset, test_dataset, data_collator = create_pretrained_dataset(
            training_args, data_args, model_args
        )
    elif data_args.dataset_type == "offline":
        train_file_path = os.path.join(data_args.input_dir, "train")
        train_dataset = create_indexed_dataset(
            data_file_prefix=train_file_path,
            skip_warmup=data_args.skip_warmup,
            warmup_only_rank0=data_args.warmup_only_rank0,
        )
        if training_args.do_eval:
            eval_file_path = os.path.join(data_args.input_dir, "eval")
            eval_dataset = create_indexed_dataset(
                data_file_prefix=eval_file_path,
                skip_warmup=data_args.skip_warmup,
                warmup_only_rank0=data_args.warmup_only_rank0,
            )
    else:
        if training_args.should_load_dataset:
            train_dataset = create_dataset_sft(
                task_group=data_args.train_dataset_path,
                task_group_prob=data_args.train_dataset_prob,
                sub_dataset_type=data_args.train_dataset_type,
                **dataset_config,
            )
        if training_args.do_eval and training_args.should_load_dataset:
            eval_dataset = create_dataset_sft(
                task_group=data_args.eval_dataset_path,
                task_group_prob=data_args.eval_dataset_prob,
                sub_dataset_type=data_args.eval_dataset_type,
                is_valid=True,
                **dataset_config,
            )

    # Freeze model based on training args (Supports for MLLM Full training)
    if not model_args.lora and getattr(training_args, "freeze_config", ""):
        freeze_model_parameters(model, training_args.freeze_config)

    model = create_peft_model(model_args, training_args, dtype, model)
    # Create trainer

    # padding to the maximum seq length in batch data when max_seq_len is None
    if getattr(model, "is_fleet", False) and not model_args.lora:
        if training_args.per_device_train_batch_size > 1:
            max_seq_len = data_args.max_seq_len
            logger.warning(f"Setting max_seq_len to {max_seq_len} for mbs > 1 using PaddleFleet model.")
        else:
            max_seq_len = None
            logger.warning("Setting max_seq_len to None for mbs = 1 using PaddleFleet Model.")
    else:
        max_seq_len = (
            data_args.max_seq_len
            if (data_args.packing or training_args.sequence_parallel or training_args.context_parallel_size > 1)
            else None
        )
        logger.info(f"Setting max_seq_len to {max_seq_len} using PaddleFormers Model.")
    if data_args.dataset_type != "pretrain":
        if "VL" in model_args.stage:
            data_collator = partial(
                mm_collate_fn,
                template=template_instance,
                processor=processor,
                tokenizer=tokenizer,
                training_args=training_args,
                model_args=model_args,
                max_seq_len=max_seq_len,
                padding_free=data_args.padding_free,
                model=model,
            )
        else:
            data_collator = partial(
                collate_fn,
                tokenizer=tokenizer,
                training_args=training_args,
                model_args=model_args,
                max_seq_len=max_seq_len,
                padding_free=data_args.padding_free,
            )

    if training_args.max_steps == -1:
        if data_args.mix_strategy == "random":
            raise ValueError(
                "When using 'random' mix_strategy, max_steps must be explicitly set (cannot be -1). "
                "Random mixing requires a fixed number of training steps to properly sample data."
            )
        if training_args.should_load_dataset and paddle.distributed.get_rank() == 0:
            if data_args.dataset_type not in {"pretrain", "offline", "map"}:
                training_args.max_steps = estimate_training(train_dataset, data_args, training_args, model_args)
                del train_dataset
                gc.collect()
                train_dataset = create_dataset_sft(
                    task_group=data_args.train_dataset_path,
                    task_group_prob=data_args.train_dataset_prob,
                    sub_dataset_type=data_args.train_dataset_type,
                    **dataset_config,
                )
            else:
                training_args.max_steps = math.ceil(len(train_dataset) / training_args.global_batch_size)
                training_args.max_steps *= training_args.num_train_epochs
                logger.info(
                    f"len(train_dataset): {len(train_dataset)}, global_batch_size: {training_args.global_batch_size}, \
                    training_args.num_train_epochs: {training_args.num_train_epochs}, training_args.max_steps: {training_args.max_steps}"
                )

        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.barrier()
            max_steps = paddle.to_tensor([training_args.max_steps])
            paddle.distributed.broadcast(max_steps, src=0)
            training_args.max_steps = int(max_steps.item())
        if training_args.max_steps <= 0:
            raise ValueError(f"Invalid max_steps: {training_args.max_steps}. Please check your dataset")

        logger.info(f"Re-setting training_args.max_steps to {training_args.max_steps}.")
    # Create the learning_rate sheduler and optimizer
    if training_args.decay_steps is None:
        training_args.decay_steps = training_args.max_steps

    if training_args.save_strategy == IntervalStrategy.EPOCH:
        training_args.save_strategy = IntervalStrategy.STEPS
        training_args.save_steps = int(training_args.max_steps / training_args.num_train_epochs)
    if training_args.evaluation_strategy == IntervalStrategy.EPOCH:
        training_args.evaluation_strategy = IntervalStrategy.STEPS
        training_args.eval_steps = int(training_args.max_steps / training_args.num_train_epochs)
    if training_args.logging_strategy == IntervalStrategy.EPOCH:
        training_args.logging_strategy = IntervalStrategy.STEPS
        training_args.logging_steps = int(training_args.max_steps / training_args.num_train_epochs)

    callbacks = []
    raw_loss_path = os.environ.get("MODEL_REPRO_RAW_LOSS_PATH")
    env_path = os.environ.get("MODEL_REPRO_ENV_PATH")
    loss_path = os.environ.get("MODEL_REPRO_LOSS_PATH")
    if raw_loss_path or env_path or loss_path:
        callbacks.append(ModelReproObservationCallback(raw_loss_path))
    if getattr(model_config.get_text_config(), "topk_method", None) == "noaux_tc":
        callbacks += [MoECorrectionBiasAdjustCallback(lr=training_args.moe_router_bias_update_rate)]
    elif getattr(model_config.get_text_config(), "topk_method", None) == "quantile_balancing":
        callbacks += [MoEQuantileBalancingCallback()]

    if training_args.use_expert_parallel:
        callbacks += [MoeExpertsGradScaleCallback(training_args)]

    if training_args.sequence_parallel and not model_args.lora:
        callbacks += [MoEGateSpGradSyncCallBack()]

    if not model_args.lora:
        callbacks += [FP8QuantWeightCallback()]

    print("callbacks:", callbacks, flush=True)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=(train_dataset if training_args.do_train and training_args.should_load_dataset else None),
        eval_dataset=(eval_dataset if training_args.do_eval and training_args.should_load_dataset else None),
        tokenizer=tokenizer,
        processing_class=processor,
        data_collator=data_collator,
        do_generation=data_args.eval_with_do_generation,
        data_args=data_args,
        callbacks=callbacks,
    )

    if training_args.train_mtp_only:
        # activate autoregressive mtp training
        freeze_param_except_mtp(model, model_config)

    trainable_parameters = [
        p for p in model.parameters() if not p.stop_gradient or ("quantization_linear" in p.name and "w_1" in p.name)
    ]
    trainer.set_optimizer_grouped_parameters(trainable_parameters)

    # Train
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        if model_args.neftune:
            neft_post_hook_handle.remove()
        total_tokens = (
            data_args.max_seq_len
            * training_args.per_device_train_batch_size
            * training_args.dataset_world_size
            * training_args.gradient_accumulation_steps
            * training_args.max_steps
        )
        total_tokens_per_second_per_gpu = (
            total_tokens / train_result.metrics["train_runtime"] / training_args.world_size
        )
        logger.info(f"Total_Tokens_per_second_per_gpu: {total_tokens_per_second_per_gpu} ")
        if not training_args.autotuner_benchmark:
            trainer.save_model(merge_tensor_parallel=training_args.tensor_model_parallel_size > 1, last_fc_to_hf=True)
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()


def create_peft_model(model_args, training_args, dtype, model):
    if model_args.lora:
        if training_args.sharding_parallel_size > 1:
            assert (
                not training_args.stage1_overlap
            ), "Currently not support enabling sharding_stage1_overlap in lora mode."
        if model_args.lora_path is None:
            target_modules = get_lora_target_modules(model)

            # Freeze model based on training args (Supports for MLLM LoRA training)
            if getattr(training_args, "freeze_config", ""):
                target_modules = get_multimodel_lora_target_modules(model, target_modules, training_args.freeze_config)

            lora_config = LoRAConfig(
                target_modules=target_modules,
                r=model_args.lora_rank,
                lora_alpha=2 * model_args.lora_rank if not model_args.rslora else 4,
                rslora=model_args.rslora,
                lora_plus_scale=model_args.lora_plus_scale,
                merge_weights=False,
                tensor_model_parallel_size=training_args.tensor_model_parallel_size,
                dtype=dtype,
                base_model_name_or_path=model_args.model_name_or_path,
            )
            model = LoRAModel(model, lora_config)
        else:
            model = LoRAModel.from_pretrained(
                model=model,
                lora_path=model_args.lora_path,
                load_checkpoint_format=training_args.load_checkpoint_format,
            )
        if hasattr(model, "_set_pipeline_name_mapping"):
            model._set_pipeline_name_mapping()
        model.print_trainable_parameters()

    return model
