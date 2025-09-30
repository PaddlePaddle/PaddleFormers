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

"""Training DPO"""

import gc
import importlib.util
import os
import time
from functools import partial

if importlib.util.find_spec("triton") is not None:
    try:
        import use_triton_in_paddle

        use_triton_in_paddle.make_triton_compatible_with_paddle()
    except Exception as _:
        raise RuntimeError(
            "Triton is installed, but not yet compatible with Paddle. "
            "Please run 'python -m pip install use-triton-in-paddle' to enable Triton support in Paddle."
        )

import paddle
from paddleformers.trainer import (
    IntervalStrategy,
    get_last_checkpoint,
    set_seed,
)
from paddleformers.transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForCausalLMPipe,
)
from paddleformers.trainer.trainer_utils import ShardingOption
from paddleformers.utils.log import logger
from paddleformers import __version__ as paddleformers_version

from ernie.callbacks import LayerwiseDropoutCallback
from ernie.configuration import Ernie4_5_MoeConfig
from ernie.dataset.dpo import collate_fn, create_dataset
from ernie.modeling_moe import Ernie4_5_MoeForCausalLM
from ernie.modeling_moe_pp import Ernie4_5_MoeForCausalLMPipe
from ernie.tokenizer import Ernie4_5_Tokenizer
from ernie.utils.common_utils import check_refined_recompute
from ernie.utils.download_utils import check_download_repo
from ernie.utils.load_utils import resolve_weight_source

# isort: off
from .dpo_estimate_training import dpo_estimate_training
from .trainer import ErnieMoEDPOTrainer
from .dpo_utils import calculate_effective_tokens, DPOConfig
from ...hparams import (
    DataArguments,
    FinetuningArguments,
    GeneratingArguments,
    ModelArguments,
)


def run_dpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    generating_args: "GeneratingArguments",
    finetuning_args: "FinetuningArguments",
):
    """
    DPO Training
    """
    if finetuning_args.dpo_benchmark:
        finetuning_args.do_train = True
        finetuning_args.do_export = False
        finetuning_args.do_predict = False
        finetuning_args.do_eval = False
        finetuning_args.overwrite_output_dir = True
        finetuning_args.load_best_model_at_end = False
        finetuning_args.save_strategy = IntervalStrategy.NO
        finetuning_args.evaluation_strategy = IntervalStrategy.NO
        if not finetuning_args.disable_tqdm:
            finetuning_args.logging_steps = 1
            finetuning_args.logging_strategy = IntervalStrategy.STEPS

    if not model_args.use_sparse_head_and_loss_fn:
        model_args.use_sparse_head_and_loss_fn = True
        logger.warning(
            "Dpo training requires use_sparse_head_and_loss_fn=True. Set use_sparse_head_and_loss_fn to True"
        )

    if data_args.max_seq_len < 16:
        data_args.max_seq_len = 16
        logger.warning(
            f"max_seq_len must be greater than 16, set max_seq_len to {data_args.max_seq_len}."
        )
    if data_args.max_seq_len < data_args.max_prompt_len + 10:
        data_args.max_prompt_len = data_args.max_seq_len - 10
        logger.warning(
            "max_seq_len must be greater than max_prompt_len + 10, "
            "set max_prompt_len to {data_args.max_prompt_len}."
        )
    if finetuning_args.loss_type == "orpo":
        finetuning_args.reference_free = True
        finetuning_args.sft_loss_ratio = 1.0
        finetuning_args.loss_type = "or"
        logger.info("orpo loss_type is equal to sft_loss + pref_loss_ratio * or_loss.")
    if (
        finetuning_args.loss_type in ["or", "simpo"]
        and not finetuning_args.reference_free
    ):
        finetuning_args.reference_free = True
        logger.warning(
            f"{finetuning_args.loss_type} loss_type only supports reference_free. "
            "Set reference_free to True."
        )
    if model_args.lora:
        assert model_args.continue_training, "Continue training is required for LoRA."
    if finetuning_args.pipeline_parallel_degree > 1:
        assert (
            hasattr(finetuning_args, "pipeline_parallel_config")
            and "enable_clear_every_step_cache"
            in finetuning_args.pipeline_parallel_config
        ), "Should set '--pipeline_parallel_config enable_clear_every_step_cache' in bash script for pp."
    if finetuning_args.sequence_parallel:
        if finetuning_args.pipeline_parallel_degree > 1:
            assert (
                hasattr(finetuning_args, "pipeline_parallel_config")
                and "disable_partial_send_recv"
                in finetuning_args.pipeline_parallel_config
            ), "Should set '--pipeline_parallel_config disable_partial_send_recv' in bash script for pp with sp."
        if finetuning_args.tensor_parallel_degree <= 1:
            finetuning_args.sequence_parallel = False
            logger.info("Tensor_parallel_degree = 1. Set sequence_parallel to False.")

    if model_args.lora and model_args.fuse_linear:
        model_args.fuse_linear = False
        logger.info("LoRA does not support fuse_linear. Set fuse_linear to False.")
    if model_args.lora:
        finetuning_args.ref_model_update_steps = -1
        logger.warning(
            "LoRA does not support ref_model_update_steps. Set ref_model_update_steps to -1."
        )

    if finetuning_args.sharding_parallel_degree > 1:
        if (
            ShardingOption.SHARD_GRAD_OP in finetuning_args.sharding
            or ShardingOption.FULL_SHARD in finetuning_args.sharding
        ):
            if finetuning_args.release_grads is True:
                finetuning_args.release_grads = False

    finetuning_args.print_config(model_args, "Model")
    finetuning_args.print_config(data_args, "Data")
    finetuning_args.print_config(finetuning_args, "DPOConfig")

    paddle.set_device(finetuning_args.device)

    set_seed(finetuning_args.seed)

    logger.warning(
        f"Process rank: {finetuning_args.local_rank}, device: {finetuning_args.device}, world_size: "
        f"{finetuning_args.world_size}, distributed training: {bool(finetuning_args.local_rank != -1)}, "
        f"16-bits training: {finetuning_args.fp16 or finetuning_args.bf16}"
    )

    last_checkpoint = None
    if (
        os.path.isdir(finetuning_args.output_dir)
        and finetuning_args.do_train
        and not finetuning_args.overwrite_output_dir
    ):
        uc_async_save = (
            finetuning_args.unified_checkpoint
            and "async_save" in finetuning_args.unified_checkpoint_config
        )
        last_checkpoint = get_last_checkpoint(
            finetuning_args.output_dir,
            signal_folder=finetuning_args.output_signal_dir,
            uc_async_save=uc_async_save,
        )

        if (
            last_checkpoint is not None
            and finetuning_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set the dtype for loading model
    dtype = paddle.get_default_dtype()
    if finetuning_args.fp16_opt_level == "O2":
        if finetuning_args.fp16:
            dtype = "float16"
        if finetuning_args.bf16:
            dtype = "bfloat16"

    logger.info("Start to load model ...")

    model_args.model_name_or_path = check_download_repo(
        model_args.model_name_or_path,
        download_hub=model_args.download_hub,
    )

    # fuse_softmax_mask only support for rocm.
    if not paddle.is_compiled_with_rocm():
        if model_args.fuse_softmax_mask:
            logger.warning(
                "The fuse_softmax_mask flag is only available when using the ROCM version of paddlepaddle. "
            )
            model_args.fuse_softmax_mask = False

    check_refined_recompute(
        finetuning_args.refined_recompute,
        finetuning_args.sequence_parallel,
        lora=model_args.lora,
    )

    if finetuning_args.weight_quantize_algo is not None:
        if finetuning_args.weight_quantize_algo == "weight_only_mix":
            quantization_config = dict(
                weight_quantize_algo={
                    "weight_only_int4": [".*mlp.experts.*"],
                    "weight_only_int8": [
                        ".*self_attn.qkv_proj.*",
                        ".*self_attn.o_proj.*",
                        ".*mlp.up_gate_proj.*",
                        ".*mlp.down_proj.*",
                    ],
                },
                ignore_modules=[".*out_linear.*"],
            )
        else:
            quantization_config = dict(
                weight_quantize_algo=finetuning_args.weight_quantize_algo,
                ignore_modules=[".*out_linear.*"],
            )
    else:
        quantization_config = dict(
            weight_quantize_algo=finetuning_args.weight_quantize_algo
        )

    model_kwargs = dict(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        dtype=dtype,
        tensor_parallel_degree=finetuning_args.tensor_parallel_degree,
        tensor_parallel_rank=finetuning_args.tensor_parallel_rank,
        virtual_pp_degree=model_args.virtual_pp_degree,
        pp_seg_method=model_args.pp_seg_method,
        recompute=finetuning_args.recompute,
        recompute_granularity=model_args.recompute_granularity,
        use_flash_attention=model_args.use_flash_attention,
        tensor_parallel_output=model_args.tensor_parallel_output,
        fuse_linear=model_args.fuse_linear,
        fuse_softmax_mask=model_args.fuse_softmax_mask,
        fuse_rms_norm=model_args.fuse_rms_norm,
        fuse_swiglu=model_args.fuse_swiglu,
        fuse_gate_detach_matmul=model_args.fuse_gate_detach_matmul,
        dpo_config=finetuning_args,
        sequence_parallel=finetuning_args.sequence_parallel,
        max_sequence_length=data_args.max_seq_len,
        use_sparse_head_and_loss_fn=model_args.use_sparse_head_and_loss_fn,
        no_recompute_layers=model_args.no_recompute_layers,
        quantization_config=quantization_config,
        use_fused_head_and_loss_fn=model_args.use_fused_head_and_loss_fn,
        recompute_use_reentrant=model_args.recompute_use_reentrant,
        use_sparse_flash_attn=model_args.use_sparse_flash_attn,
        refined_recompute=finetuning_args.refined_recompute,
        fuse_rope=model_args.fuse_rope,
        moe_group=model_args.moe_group,
        hidden_dropout_prob=finetuning_args.hidden_dropout_prob,
        attention_probs_dropout_prob=finetuning_args.attention_probs_dropout_prob,
        moe_multimodal_dispatch_use_allgather=model_args.moe_multimodal_dispatch_use_allgather,
        moe_group_experts=model_args.moe_group_experts,
        moe_aux_loss_lambda=model_args.moe_aux_loss_lambda,
        moe_orthogonal_loss_lambda=model_args.moe_orthogonal_loss_lambda,
        moe_z_loss_lambda=model_args.moe_z_loss_lambda,
        moe_use_hard_gate=model_args.moe_use_hard_gate,
        moe_use_aux_free=model_args.moe_use_aux_free,
        num_acc_steps=finetuning_args.gradient_accumulation_steps,
        add_tail_layers=model_args.add_tail_layers,
        num_nextn_predict_layers=0,
        download_hub=model_args.download_hub,
    )

    try:
        from paddleformers.utils.download import (
            DownloadSource,
        )  # test if paddleformers is the newest
    except Exception:
        DownloadSource = None

    download_source_kwargs = {}
    if DownloadSource is None:
        if model_args.download_hub == "huggingface":
            download_source_kwargs["from_hf_hub"] = True
        elif model_args.download_hub == "aistudio":
            download_source_kwargs["from_aistudio"] = True
        elif model_args.download_hub == "modelscope":
            download_source_kwargs["from_modelscope"] = True
    else:
        download_source_kwargs["download_hub"] = model_args.download_hub

    weight_source = resolve_weight_source(
        model_args.model_name_or_path,
        download_source_kwargs=download_source_kwargs,
    )
    if weight_source["convert_from_hf"]:
        finetuning_args.use_huggingface_model = True
        if finetuning_args.weight_quantize_algo is not None:
            quantization_config["weight_quantize_algo"] = {
                "weight_only_int4": [".*mlp.experts.*"],
                "weight_only_int8": [
                    ".*self_attn.q_proj.*",
                    ".*self_attn.k_proj.*",
                    ".*self_attn.v_proj.*",
                    ".*self_attn.o_proj.*",
                    ".*mlp.up_proj.*",
                    ".*mlp.gate_proj.*",
                    ".*mlp.down_proj.*",
                ],
            }
        finetuning_args.layerwise_lr_decay_bound = 1.0
        model_args.pp_seg_method = "layer:DecoderLayer|EmptyLayer"
        logger.info("loading model from HuggingFace")

    convert_from_kwargs = {
        (
            "convert_from_hf"
            if paddleformers_version >= "0.3"
            else "convert_from_torch"
        ): weight_source["convert_from_hf"]
    }
    if paddleformers_version >= "0.3":
        finetuning_args.save_to_hf = (
            weight_source["save_to_hf"] and finetuning_args.use_huggingface_model
        )
    if model_args.moe_use_aux_free is False:
        model_kwargs.update({"moe_use_aux_free": model_args.moe_use_aux_free})

    if finetuning_args.use_huggingface_model:
        if (
            model_args.use_attn_mask_start_row_indices
            and model_args.use_sparse_flash_attn
        ):
            _attn_implementation = "flashmask"
        else:
            _attn_implementation = "sdpa"
        config = AutoConfig.from_pretrained(
            _attn_implementation=_attn_implementation,
            use_filtered_label_loss=model_args.use_sparse_head_and_loss_fn,
            loss_subbatch_sequence_length=1024,
            attention_dropout_prob=finetuning_args.attention_probs_dropout_prob,
            **convert_from_kwargs,
            **model_kwargs,
        )
        config.use_fused_head_and_loss_fn = model_args.use_fused_head_and_loss_fn
    else:
        config = Ernie4_5_MoeConfig.from_pretrained(**model_kwargs)

    if (
        finetuning_args.pipeline_parallel_degree > 1
        and finetuning_args.weight_quantize_algo is not None
        and config.tie_word_embeddings
    ):
        raise NotImplementedError(
            "Quantization is not supported for models with tied lm_head and word_embedding \
            weights when using Pipeline Parallelism (PP)."
        )

    if (
        config.get("moe_num_experts", None) is None
        or config.get("moe_num_experts", 0) == 0
    ):
        config.moe_group = (
            "dummy" if model_args.moe_group == "mp" else model_args.moe_group
        )

    if finetuning_args.use_huggingface_model:
        model_class = AutoModelForCausalLM
        if finetuning_args.pipeline_parallel_degree > 1:
            model_class = AutoModelForCausalLMPipe
    else:
        model_class = Ernie4_5_MoeForCausalLM
        if finetuning_args.pipeline_parallel_degree > 1:
            model_class = Ernie4_5_MoeForCausalLMPipe

    # (NOTE): paddleformers-cli currently only support finetuning ernie4_5 and ernie4_5_moe models from huggingface
    if finetuning_args.use_huggingface_model and config.model_type not in (
        "ernie4_5",
        "ernie4_5_moe",
    ):
        raise ValueError(
            f"Currently, only support ernie4_5 and ernie4_5_moe for HuggingFace model, but got {config.model_type}."
        )

    if model_args.continue_training:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            **convert_from_kwargs,
            **download_source_kwargs,
        )
    else:
        model = model_class.from_config(config, dtype=dtype)

    if not finetuning_args.reference_free and not model_args.lora:
        if finetuning_args.use_huggingface_model:
            ref_config = AutoConfig.from_pretrained(
                _attn_implementation=_attn_implementation,
                use_filtered_label_loss=model_args.use_sparse_head_and_loss_fn,
                loss_subbatch_sequence_length=1024,
                **model_kwargs,
            )
            ref_config.use_fused_head_and_loss_fn = (
                model_args.use_fused_head_and_loss_fn
            )
            if (
                ref_config.get("moe_num_experts", None) is None
                or ref_config.get("moe_num_experts", 0) == 0
            ):
                ref_config.moe_group = (
                    "dummy" if model_args.moe_group == "mp" else model_args.moe_group
                )
        else:
            ref_config = Ernie4_5_MoeConfig.from_pretrained(**model_kwargs)
            if (
                ref_config.get("moe_num_experts", None) is None
                or ref_config.get("moe_num_experts", 0) == 0
            ):
                ref_config.moe_group = (
                    "dummy" if model_args.moe_group == "mp" else model_args.moe_group
                )
        ref_model = model_class.from_config(ref_config, dtype=dtype)
        # make sure the state_dict is the same to get the same loss for first step
        ref_model.set_state_dict(model.state_dict())
    else:
        ref_model = None

    model.config.dpo_config = None

    if model.config.head_dim is None:
        del model.config.head_dim
    if ref_model is not None and ref_model.config.head_dim is None:
        del ref_model.config.head_dim

    if model_args.lora:
        from ernie.utils.peft_utils import initialize_lora_model

        model = initialize_lora_model(
            model=model,
            training_args=finetuning_args,
            model_args=model_args,
            resume_from_checkpoint=last_checkpoint is not None,
            dtype=dtype,
        )

    tokenizer_cls = (
        AutoTokenizer if weight_source["convert_from_hf"] else Ernie4_5_Tokenizer
    )
    tokenizer = tokenizer_cls.from_pretrained(
        model_args.model_name_or_path,
        **convert_from_kwargs,
        **download_source_kwargs,
    )
    logger.info("Loading model & tokenizer successfully !")

    logger.info("Start to create dataset ...")
    dataset_config = {
        "tokenizer": tokenizer,
        "max_seq_len": data_args.max_seq_len,
        "max_prompt_len": data_args.max_prompt_len,
        "random_seed": finetuning_args.seed,
        "num_replicas": finetuning_args.dataset_world_size,
        "rank": finetuning_args.dataset_rank,
        "num_samples_each_epoch": data_args.num_samples_each_epoch,
        "random_shuffle": data_args.random_shuffle,
        "greedy_intokens": data_args.greedy_intokens,
        "buffer_size": data_args.buffer_size,
        "use_attn_mask_start_row_indices": model_args.use_attn_mask_start_row_indices,
        "mask_out_eos_token": data_args.mask_out_eos_token,
    }

    if finetuning_args.max_steps == -1:
        if finetuning_args.should_load_dataset and paddle.distributed.get_rank() == 0:
            # NOTE(gongenlei): not to feed train_dataset, or the data will be wrong in next training.
            finetuning_args, _ = dpo_estimate_training(
                tokenizer, data_args, finetuning_args, config=model.config
            )

        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.barrier()
            pd_max_steps = paddle.to_tensor([finetuning_args.max_steps])
            paddle.distributed.broadcast(pd_max_steps, src=0)
            finetuning_args.max_steps = int(pd_max_steps.item())
        logger.info(
            f"Re-setting finetuning_args.max_steps to {finetuning_args.max_steps} ({finetuning_args.num_train_epochs})"
        )
        if finetuning_args.max_steps <= 0:
            raise ValueError(
                f"Invalid max_steps: {finetuning_args.max_steps}. Please check your dataset"
            )
    if finetuning_args.save_strategy == IntervalStrategy.EPOCH:
        finetuning_args.save_strategy = IntervalStrategy.STEPS
        finetuning_args.save_steps = int(
            finetuning_args.max_steps / finetuning_args.num_train_epochs
        )
    if finetuning_args.evaluation_strategy == IntervalStrategy.EPOCH:
        finetuning_args.evaluation_strategy = IntervalStrategy.STEPS
        finetuning_args.eval_steps = int(
            finetuning_args.max_steps / finetuning_args.num_train_epochs
        )
    if finetuning_args.logging_strategy == IntervalStrategy.EPOCH:
        finetuning_args.logging_strategy = IntervalStrategy.STEPS
        finetuning_args.logging_steps = int(
            finetuning_args.max_steps / finetuning_args.num_train_epochs
        )

    if finetuning_args.should_load_dataset:
        train_dataset = create_dataset(
            task_group=data_args.train_dataset_path,
            task_group_prob=data_args.train_dataset_prob,
            sub_dataset_type=data_args.train_dataset_type,
            **dataset_config,
        )

    if finetuning_args.do_eval and finetuning_args.should_load_dataset:
        eval_dataset = create_dataset(
            task_group=data_args.eval_dataset_path,
            task_group_prob=data_args.eval_dataset_prob,
            sub_dataset_type=data_args.eval_dataset_type,
            is_valid=True,
            **dataset_config,
        )
    logger.info("Creating dataset successfully ...")

    dpo_config = DPOConfig(
        beta=finetuning_args.beta,
        offset_alpha=finetuning_args.offset_alpha,
        simpo_gamma=finetuning_args.simpo_gamma,
        normalize_logps=finetuning_args.normalize_logps,
        label_smoothing=finetuning_args.label_smoothing,
        loss_type=finetuning_args.loss_type,
        pref_loss_ratio=finetuning_args.pref_loss_ratio,
        sft_loss_ratio=finetuning_args.sft_loss_ratio,
        dpop_lambda=finetuning_args.dpop_lambda,
        ref_model_update_steps=finetuning_args.ref_model_update_steps,
        reference_free=finetuning_args.reference_free,
        lora=model_args.lora,
    )

    trainer = ErnieMoEDPOTrainer(
        model=model,
        ref_model=ref_model,
        dpo_config=dpo_config,
        args=finetuning_args,
        train_dataset=(
            train_dataset
            if finetuning_args.do_train and finetuning_args.should_load_dataset
            else None
        ),
        eval_dataset=(
            eval_dataset
            if finetuning_args.do_eval and finetuning_args.should_load_dataset
            else None
        ),
        tokenizer=tokenizer,
        data_collator=partial(
            collate_fn,
            tokenizer=tokenizer,
            max_seq_len=data_args.max_seq_len,
            use_sparse_head_and_loss_fn=model_args.use_sparse_head_and_loss_fn,
            use_fused_head_and_loss_fn=model_args.use_fused_head_and_loss_fn,
            use_response_score_delta=finetuning_args.offset_alpha > 0.0,
        ),
        model_with_dpo_criterion=True,
    )

    if (
        finetuning_args.hidden_dropout_prob
        or finetuning_args.attention_probs_dropout_prob
    ):
        trainer.add_callback(LayerwiseDropoutCallback())

    if finetuning_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        if (
            finetuning_args.dpo_benchmark
            and finetuning_args.should_load_dataset
            and paddle.distributed.get_rank() == 0
        ):
            del train_dataset
            gc.collect()
            train_dataset = create_dataset(
                task_group=data_args.train_dataset_path,
                task_group_prob=data_args.train_dataset_prob,
                sub_dataset_type=data_args.train_dataset_type,
                **dataset_config,
            )
            total_effective_tokens, total_tokens = calculate_effective_tokens(
                finetuning_args, train_dataset, data_args.max_seq_len
            )
            effective_tokens_per_second = (
                total_effective_tokens / train_result.metrics["train_runtime"]
            )
            total_tokens_per_second = (
                total_tokens / train_result.metrics["train_runtime"]
            )
            effective_ratio = 100 * total_effective_tokens / total_tokens
            logger.info(
                "[timelog] {}: {:.2f} % ({}) ".format(
                    "Effective ratio",
                    effective_ratio,
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            )
            logger.info(
                "[timelog] {}: {:.2f} token/s ({}) ".format(
                    "Effective tokens per second",
                    effective_tokens_per_second,
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            )
            logger.info(
                "[timelog] {}: {:.2f} token/s ({}) ".format(
                    "Tokens per second",
                    total_tokens_per_second,
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            )

        if not finetuning_args.dpo_benchmark:
            trainer.save_model(
                merge_tensor_parallel=finetuning_args.tensor_parallel_degree > 1
            )
            if paddle.distributed.get_world_size() > 1:
                paddle.distributed.barrier()
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()

    if finetuning_args.do_eval:
        eval_result = trainer.evaluate()
        trainer.log_metrics("eval", eval_result)
        trainer.save_metrics("eval", eval_result, combined=False)
