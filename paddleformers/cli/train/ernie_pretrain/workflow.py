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

import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from functools import partial

import numpy as np
import paddle
from paddle.distributed import fleet

from paddleformers.cli.train.ernie_pretrain.src.utils import logger
from paddleformers.utils.tools import get_env_device, paddle_device

try:
    from paddle.distributed.utils.process_utils import SUCCESS_CODE, set_affinity
except ImportError:
    set_affinity = None
    SUCCESS_CODE = 0

from paddleformers.trainer import PdArgumentParser, get_last_checkpoint

try:
    from paddleformers.utils.downloader import get_static_model_on_pdc
except ImportError:
    get_static_model_on_pdc = None

from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from safetensors import safe_open

from paddleformers.cli.train.ernie_pretrain.model_config import ModelConfig
from paddleformers.cli.train.ernie_pretrain.models.ernie import ErnieMoEConfig
from paddleformers.cli.train.ernie_pretrain.models.ernie.modeling_moe import (
    ErnieMoEForCausalLM,
)
from paddleformers.cli.train.ernie_pretrain.models.ernie.modeling_pp import (
    ErnieMoEForCausalLMPipe,
)
from paddleformers.cli.train.ernie_pretrain.src.callbacks import (
    GlobalRNGCallback,
    MoECorrectionBiasAdjustCallback,
    OrthogonalCallback,
)
from paddleformers.cli.train.ernie_pretrain.src.tokenizers.tokenization_eb_v2 import (
    ErnieBotTokenizer,
)
from paddleformers.cli.train.ernie_pretrain.src.trainers import (
    PreTrainingArguments,
    PretrainingTrainer,
)
from paddleformers.cli.train.ernie_pretrain.src.utils import setup_logger_output_file
from paddleformers.cli.train.ernie_pretrain.src.utils.misc import global_training_logs
from paddleformers.cli.train.ernie_pretrain.src.utils.seed_utils import set_seed
from paddleformers.data.causal_dataset import (
    build_train_valid_test_datasets,
    check_data_split,
)
from paddleformers.datasets.collate import collate_fn
from paddleformers.datasets.loader import create_dataset as create_dataset_sft
from paddleformers.trainer import TrainingArguments
from paddleformers.trainer.unified_checkpoint import unified_checkpoint
from paddleformers.transformers.model_utils import unwrap_model

try:
    from paddleformers.trainer.trainer_utils import log_trainer_start
except ImportError:

    def log_trainer_start():
        if "MAIN_PROCESS_STARTED" not in os.environ:
            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logger.info(f"The Training Main Process Started Successfully. time: {start_time}, pid: {os.getpid()}")
            os.environ["MAIN_PROCESS_STARTED"] = "1"


log_trainer_start()


def load_huggingface_checkpoint(model, args):
    fused_rms_norm_replace = [
        ("self_attn.fused_rms_norm_linear.rms_norm_weight", "input_layernorm.weight"),
        ("self_attn.fused_rms_norm_linear.linear_weight", "self_attn.qkv_proj.weight"),
    ]
    shared_layers_prefix = "shared_layers.embed_weight_share."
    unnamed_layers = ["ernie.norm.weight", "lm_head.weight"]

    logger.info(f"Loading huggingface checkpoint from {args.model_name_or_path}")
    with open(os.path.join(args.model_name_or_path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    ep_degree = fleet.get_hybrid_communicate_group().get_expert_parallel_world_size()
    ep_rank = fleet.get_hybrid_communicate_group().get_expert_parallel_rank()
    expert_offset = (model.config.moe_num_experts // ep_degree) * ep_rank
    use_torch_format = False

    def param_to_weight(name):
        # for PP=1, we only need to substitute the fused_rms_norm and expert_id
        for src, dst in fused_rms_norm_replace:
            name = name.replace(src, dst)
        if m := re.search(r"mlp\.experts\.(\d+)", name):
            expert_id = expert_offset + int(m.group(1))
            s, e = m.span()
            name = name[:s] + f"mlp.experts.{expert_id}" + name[e:]
        if isinstance(model, ErnieMoEForCausalLM):
            return name

        # for PP>1, we also need to handle special layers and adjust layer_idx
        if name.startswith(shared_layers_prefix):
            return "ernie." + name[len(shared_layers_prefix) :]
        layer_idx, stem = name.split(".", maxsplit=1)
        if stem == "weight":
            return unnamed_layers.pop(0)
        if stem.startswith("mtp"):
            return f"ernie.{stem}"
        return f"ernie.layers.{int(layer_idx) - 1}.{stem}"

    def try_torch_format(weight_key):
        if weight_key.startswith("ernie."):
            weight_key = "model." + weight_key[6:]

        key_decompose = [weight_key]
        if ".up_gate_proj." in weight_key:
            key_decompose = [
                weight_key.replace(".up_gate_proj.", ".gate_proj."),
                weight_key.replace(".up_gate_proj.", ".up_proj."),
            ]
        elif ".qkv_proj." in weight_key:
            key_decompose = [
                weight_key.replace(".qkv_proj.", ".q_proj."),
                weight_key.replace(".qkv_proj.", ".k_proj."),
                weight_key.replace(".qkv_proj.", ".v_proj."),
            ]

        tensor_decompose = []
        for key in key_decompose:
            if not (weight_file := weight_map.get(key)):
                return None
            with safe_open(
                os.path.join(args.model_name_or_path, weight_file),
                framework="numpy",
            ) as f:
                tensor = paddle.to_tensor(f.get_tensor(key))
            if "_proj." in key or ".gate." in key:
                tensor = tensor.T.contiguous()
            tensor_decompose.append(tensor)

        if len(tensor_decompose) == 1:
            return tensor_decompose[0]
        else:
            return paddle.concat(tensor_decompose, axis=-1)

    for name, param in model.named_parameters():
        weight_key = param_to_weight(name)
        if weight_file := weight_map.get(weight_key):
            with safe_open(
                os.path.join(args.model_name_or_path, weight_file),
                framework="numpy",
            ) as f:
                weight = paddle.to_tensor(f.get_tensor(weight_key))
        elif (weight := try_torch_format(weight_key)) is not None:
            use_torch_format = True
        else:
            logger.warning(f"param `{name}`'s weight `{weight_key}` not found.")
            continue
        if use_torch_format and "lm_head" in weight_key:
            weight = weight.T.contiguous()
        if param.shape != weight.shape:
            logger.error(
                f"param `{name}`'s shape doesn't match weight `{weight_key}`: {param.shape} and {weight.shape}."
            )
        param.copy_(weight)


def get_expected_state_dict(model, **kwargs):
    fused_rms_norm_replace = [
        ("self_attn.fused_rms_norm_linear.rms_norm_weight", "input_layernorm.weight"),
        ("self_attn.fused_rms_norm_linear.linear_weight", "self_attn.qkv_proj.weight"),
    ]
    shared_layers_prefix = "embed_share."

    model = unwrap_model(model)
    hcg = fleet.get_hybrid_communicate_group()
    ep_degree = hcg.get_expert_parallel_world_size()
    ep_rank = hcg.get_expert_parallel_rank()
    expert_offset = (model.config.moe_num_experts // ep_degree) * ep_rank

    if model.config.head_dim is None:
        head_dim = model.config.hidden_size // model.config.num_attention_heads
    else:
        head_dim = model.config.head_dim
    q_dim = head_dim * model.config.num_attention_heads
    kv_dim = head_dim * model.config.num_key_value_heads

    def copy_attr(out, param):
        if hasattr(param, "is_distributed"):
            out.is_distributed = param.is_distributed
        if hasattr(param, "no_sync"):
            out.no_sync = param.no_sync
        return out

    def param_to_weight(name):
        # for PP=1, we only need to substitute the fused_rms_norm and expert_id
        for src, dst in fused_rms_norm_replace:
            name = name.replace(src, dst)
        if m := re.search(r"\.experts\.(\d+)\.", name):
            expert_id = expert_offset + int(m.group(1))
            s, e = m.span()
            name = name[:s] + f".experts.{expert_id}." + name[e:]
        if isinstance(model, ErnieMoEForCausalLM):
            return name

        # for PP>1, we also need to handle shared layers
        if name.startswith(shared_layers_prefix):
            return "ernie." + name[len(shared_layers_prefix) :]
        return name

    state_dict = {}
    for name, param in model.state_dict().items():
        name = param_to_weight(name)
        if name.startswith("ernie."):
            name = "model." + name[6:]

        if "_proj." in name or ".gate." in name or "lm_head" in name:
            param = copy_attr(param.T, param)

        if ".up_gate_proj." in name:
            gate, up = param.split(2)
            gate, up = copy_attr(gate, param), copy_attr(up, param)
            state_dict[name.replace(".up_gate_proj.", ".gate_proj.")] = gate
            state_dict[name.replace(".up_gate_proj.", ".up_proj.")] = up
        elif ".qkv_proj." in name:
            assert q_dim + kv_dim * 2 == param.shape[0]
            state_dict[name.replace(".qkv_proj.", ".q_proj.")] = param[:q_dim]
            state_dict[name.replace(".qkv_proj.", ".k_proj.")] = param[q_dim:-kv_dim]
            state_dict[name.replace(".qkv_proj.", ".v_proj.")] = param[-kv_dim:]
        else:
            state_dict[name] = param

    return state_dict


def update_model_config_from_args(config: ErnieMoEConfig, model_args: dict):
    for k, v in model_args.items():
        if hasattr(config, k):
            logger.info(f"update model config: {k} = {v}")
            setattr(config, k, v)
        else:
            logger.warning(f"model config key: {k} does not exist")
    return config


def get_tp_split_ckpt(args, path):
    tp_degree = args.tensor_model_parallel_size
    tp_rank = max(args.tensor_parallel_rank, 0)

    if tp_degree > 1:
        ckpt_path = os.path.join(path, f"tp{tp_degree:02d}", f"model_state.tp{tp_rank:02d}.pdparams")
    else:
        ckpt_path = os.path.join(path, "model_state.pdparams")
    return ckpt_path


@dataclass
class AllArguments(PreTrainingArguments):
    def __post_init__(self):
        PreTrainingArguments.__post_init__(self)


@dataclass
class ExpConfig:
    max_steps: int
    name: str
    config: dict[str, str]


def create_pretrained_dataset(args):
    assert args.input_dir is not None and len(args.input_dir.split()) > 1

    check_data_split(
        args.split,
        args.do_train,
        args.do_eval,
        args.do_predict,
    )

    train_val_test_num_samples = [
        args.per_device_train_batch_size * args.dataset_world_size * args.max_steps * args.gradient_accumulation_steps,
        args.per_device_eval_batch_size
        * args.dataset_world_size
        * args.eval_iters
        * (args.max_steps // args.eval_steps + 1),
        args.per_device_eval_batch_size * args.dataset_world_size * args.test_iters,
    ]

    train_dataset, valid_dataset, test_dataset = build_train_valid_test_datasets(
        data_prefix=args.input_dir.split(),
        data_impl="mmap",
        splits_string=args.split,
        train_val_test_num_samples=train_val_test_num_samples,
        seq_length=args.max_seq_len + args.multi_token_pred_depth,
        seed=args.seed,
        skip_warmup=True,
        data_cache_path=None,
    )

    from paddleformers.data import Stack

    def _collate_data(data, stack_fn=Stack()):
        tokens_ = stack_fn([x["text"] for x in data])

        labels = tokens_[:, 1:]
        tokens = tokens_[:, :-1]

        return {
            "input_ids": tokens,
            "labels": labels,
        }

    return train_dataset, valid_dataset, test_dataset, _collate_data


def run_ernie_pretrain(model_args, data_args, generating_args, training_args):
    if set_affinity is not None:
        set_affinity_code = set_affinity()
        if set_affinity_code == SUCCESS_CODE:
            logger.info("set affinity successed.")
        else:
            logger.info("set affinity failed.")
    os.makedirs(training_args.output_dir, exist_ok=True)
    parser = PdArgumentParser(AllArguments)

    if not hasattr(training_args, "pipeline_parallel_config"):
        training_args.pipeline_parallel_config = ""

    if getattr(training_args, "sequence_parallel", 0):
        logger.warning("disabling `partial_send_recv` when using sequence parallel")
        training_args.partial_send_recv = False

    if getattr(training_args, "bf16", False) and not getattr(training_args, "pp_delay_scale_loss", False):
        logger.warning(
            "It is recommended to enable pp_delay_scale_loss for better performance "
            "of precision when using bf16 in training"
        )
        training_args.pp_delay_scale_loss = True

    if getattr(training_args, "dp_comm_overlap", False):
        logger.warning("Pipeline dp_comm_overlap and FusedLinearWithGradAdd can not be used at the same time.")

    if getattr(training_args, "timer", False):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallel,
        )

        PipelineParallel.timer_printer = lambda _: None

    def formatv(v):
        if isinstance(v, ListConfig):
            return [formatv(vv) for vv in list(v)]
        elif isinstance(v, DictConfig):
            return {k: formatv(vv) for k, vv in dict(v).items()}
        return v

    model_args = {k: formatv(v) for k, v in asdict(model_args).items()}
    trainer_args = {k: formatv(v) for k, v in asdict(training_args).items()}
    data_args = {k: formatv(v) for k, v in asdict(data_args).items()}
    if model_args["moe_group"] == "ep":
        assert (
            trainer_args.get("expert_model_parallel_size", -1) > 1
        ), "When moe_group is 'ep', 'expert_model_parallel_size' must be set to greater than 1."
        assert (
            trainer_args.get("sharding_parallel_size", -1) > 1
        ), "sharding_parallel_size should > 1 in when moe_group is 'ep'."
        assert (
            trainer_args["sharding"][0].value == "stage1"
        ), "Hybrid expert parallel only supports sharding stage1 now."
        assert trainer_args.get("split_param", False), "Hybrid expert parallel only supports Sharding stage1 V2 now."
        assert (
            trainer_args.get("data_parallel_size", 1) == 1
        ), "Now, moe_group = 'ep' cannot be used with data_parallel_size > 1."

    (args,) = parser.parse_dict(dict(**model_args, **trainer_args, **data_args))
    args.audio_config = dict(model_args).get("model_config", {}).get("audio_config", {})
    args.use_moe = (
        getattr(training_args, "use_moe", None)
        if hasattr(training_args, "use_moe")
        else getattr(model_args, "use_moe", False)
    )
    args.moe_with_send_router_loss = (
        getattr(training_args, "moe_with_send_router_loss", None)
        if hasattr(training_args, "moe_with_send_router_loss")
        else getattr(model_args, "moe_with_send_router_loss", True)
    )
    args.eval_iters = 10
    args.test_iters = args.eval_iters * 10

    args.enable_delay_scale_loss = training_args.pp_delay_scale_loss

    model_config = model_args.get("ernie_model_config", {})
    model_config = {k: formatv(v) for k, v in model_config.items()}
    logger.info(f"model_config_from_yaml: {json.dumps(model_config, indent=4)}")

    setup_logger_output_file(training_args.output_dir, args.local_rank)
    paddle.set_device(args.device)
    np.random.seed(args.seed)
    random.seed(args.seed)
    set_seed(args.seed)

    if args.enable_optimizer_timer and hasattr(fleet.fleet, "_user_defined_strategy"):
        strategy = fleet.fleet._user_defined_strategy
        strategy.strategy.hybrid_configs.enable_optimizer_timer = args.enable_optimizer_timer
        assert strategy.hybrid_configs["enable_optimizer_timer"] is True
        logger.info("set enable_optimizer_timer to True")

    if get_env_device() == "gpu":
        prop = paddle_device.get_device_properties()
        if prop.total_memory < args.pre_alloc_memory * 1024 * 1024 * 1024:
            logger.warning("Invalid value for `pre_alloc_memory`, so pre-allocating just failed.")
        elif args.pre_alloc_memory > 0:
            logger.warning(
                f"pre-allocating a tensor whose memory capacity is {args.pre_alloc_memory} GB " "and then release it."
            )
            memory_size = int(args.pre_alloc_memory * 1024 * 1024 * 1024)
            tmp_tensor = paddle.empty([memory_size], dtype=paddle.uint8)
            print("pre allocat a tensor with shape:", tmp_tensor.shape)
            del tmp_tensor

    last_checkpoint = None
    if os.path.isdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

        output = paddle.to_tensor(preds)
        labels = paddle.to_tensor(p.label_ids)
        output = [t.astype("float32").cuda() for t in output]
        labels = [t[t != tokenizer.ignored_index] for t in labels]
        labels = [t.cuda() for t in labels]
        all_numel = (paddle.concat(labels, 0) != tokenizer.ignored_index).astype("int64").sum()
        ignored = (paddle.concat(labels, 0) == -100).astype("int64").sum()
        labels = all_numel - ignored
        output = sum(output)
        logger.info(f"output : {output.item()}, labels : {labels.item()}")
        nll_loss = output / (labels + 1.0e-6)
        ppl = paddle.exp(nll_loss)

        return {
            "nll_loss": nll_loss.item(),
            "ppl": ppl.item(),
            "num_token": labels.item(),
        }

    def register_pp_reshard_information(num_hidden_layers):
        from paddleformers.trainer.utils.reshard.pp_reshard import (
            register_index_layer_func,
            register_layername_prefix,
            regitser_extract_layer_name_func,
        )

        register_layername_prefix("column_sequence_parallel_linear")
        register_layername_prefix("row_sequence_parallel_linear")
        register_layername_prefix("linear")
        register_layername_prefix("embedding")
        register_layername_prefix("create_parameter")
        register_layername_prefix("lm_head")
        register_layername_prefix("moe_gate")
        register_layername_prefix("fused_linear")
        register_layername_prefix("layer_norm")
        register_layername_prefix("ernie_mo_elm_head_pipe")

        def extract_layer_name(param_name):
            patterns = [
                r"^ernie\.embed_tokens",
                r"^ernie\.norm",
                r"^lm_head",
                r"^ernie\.layers((\.\d+))",
            ]
            for p in patterns:
                match = re.search(p, param_name)
                if match:
                    return match.group()

        def index_layer(layer_name):
            if layer_name == "ernie.embed_tokens":
                return 0
            elif layer_name == "ernie.norm":
                return num_hidden_layers + 1
            elif layer_name == "lm_head":
                return num_hidden_layers + 2
            else:
                pattern = r"ernie\.layers((\.(\d+)))"
                match = re.search(pattern, layer_name)
                assert match
                index = int(match.group(3)) + 1
                assert index <= num_hidden_layers, f"{index} {num_hidden_layers}"
                return index

        def sname_to_tname(pp_model):
            vpp_degree = pp_model._layers._num_virtual_pipeline_stages

            sname_to_tname = dict()
            for key, param in pp_model.named_parameters():
                if vpp_degree == 1:
                    res = re.search(r"^_layers\.(\d+)((\.\w+)+)", key)
                else:
                    res = re.search(r"^_layers\.(\d+)\.(\d+)((\.\w+)+)", key)
                layer_id = int(res.group(1))
                sname_suffix = res.group(2) if vpp_degree == 1 else res.group(3)
                new_sname = "ernie"
                if layer_id > 0 and layer_id < num_hidden_layers:
                    new_sname += ".layers." + str(layer_id - 1)
                if vpp_degree == 1:
                    if layer_id == num_hidden_layers + 1:
                        new_sname += ".norm"
                    if layer_id == num_hidden_layers + 2:
                        new_sname += ".lm_head"
                else:
                    if layer_id == 0 and "embed_tokens" not in key:
                        new_sname += ".layers." + str(layer_id)
                    if layer_id == num_hidden_layers:
                        if int(res.group(2)) == 1:
                            new_sname += ".norm"
                        else:
                            new_sname = "lm_head"
                new_sname += sname_suffix
                sname_to_tname[new_sname] = param.name
            return sname_to_tname

        regitser_extract_layer_name_func(extract_layer_name)
        register_index_layer_func(index_layer)

        try:
            from paddleformers.trainer.utils.reshard.pp_reshard import (
                register_sname_to_tname_func,
            )
        except Exception:
            logger.warning(
                "Third-Party PaddleNLP doesn't support pp-sharding reshard! No need to register sname_to_tname func"
            )
        else:
            register_sname_to_tname_func(sname_to_tname)

    dtype = "float32"
    if args.fp16 and args.fp16_opt_level == "O2":
        paddle.set_default_dtype("float16")
        dtype = "float16"
    elif args.bf16:
        paddle.set_default_dtype("bfloat16")
        dtype = "bfloat16"

    if args.moe_group.lower() in {"mp", "tp", "model", "dummy"}:
        logger.info(f"disable moe flag when using moe-group={args.moe_group}")
        args.use_moe = False
    args.multi_token_pred_depth = model_config.get("multi_token_pred_depth", 0)

    cfg = ErnieMoEConfig.from_pretrained(args.model_name_or_path)
    cfg.seqlen = args.max_seq_len
    cfg.token_balance_seqlen = args.max_seq_len * args.per_device_train_batch_size
    cfg.fp16_opt_level = args.fp16_opt_level
    cfg.moe_group = args.moe_group
    cfg.dtype = dtype
    cfg.use_fp8 = args.use_fp8
    cfg.enable_mtp_magic_send = args.enable_mtp_magic_send

    ortho_loss_lambda = cfg.moe_orthogonal_loss_lambda if hasattr(cfg, "moe_orthogonal_loss_lambda") else 0.0
    if args.use_ortho_loss_callback:
        logger.info("using orthogonal loss callback")
        cfg.moe_orthogonal_loss_lambda = 0.0

    if args.tensor_model_parallel_size > 1:
        cfg.sequence_parallel = args.sequence_parallel
        cfg.tensor_model_parallel_size = max(fleet.get_hybrid_communicate_group().get_model_parallel_world_size(), 1)
        cfg.tensor_parallel_rank = max(fleet.get_hybrid_communicate_group().get_model_parallel_rank(), 0)
    else:
        cfg.sequence_parallel = False
        cfg.tensor_model_parallel_size = 1
        cfg.tensor_parallel_rank = 0
    cfg.micro_batch_size = args.per_device_train_batch_size

    tokenizer = ErnieBotTokenizer.from_pretrained(args.tokenizer_name_or_path)
    tokenizer.ignored_index = cfg.ignored_index
    logger.info(
        f"using tokenizer={type(tokenizer)}, bos:{tokenizer.bos_token_id} "
        f"eos:{tokenizer.eos_token_id} pad:{tokenizer.pad_token_id} "
    )

    cfg = update_model_config_from_args(cfg, model_config)

    if args.pipeline_model_parallel_size > 1:
        cfg.virtual_pipeline_model_parallel_size = args.virtual_pipeline_model_parallel_size
        cfg.num_acc_steps = args.gradient_accumulation_steps
        cfg.moe_with_send_router_loss = args.moe_with_send_router_loss
        cfg.enable_delay_scale_loss = args.enable_delay_scale_loss
        register_pp_reshard_information(cfg.num_hidden_layers)

        model = ErnieMoEForCausalLMPipe(cfg)
    else:
        model = ErnieMoEForCausalLM(cfg)

    if not args.from_scratch and last_checkpoint is None and args.resume_from_checkpoint is None:
        load_huggingface_checkpoint(model, args)

    # We must use non-huggingface format to save intermediate checkpoints during training.
    args.save_to_hf = False
    args.load_checkpoint_format = "unified_checkpoint"
    args.save_checkpoint_format = "sharding_io"

    cfg = model.config
    logger.info(f"using model type:{type(model)}")
    paddle.set_default_dtype("float32")

    logger.info(f"using model={type(model)}, cfg={cfg}")

    dataset_config = {
        "tokenizer": tokenizer,
        "max_seq_len": args.max_seq_len + 1,
        "random_seed": args.seed,
        "num_replicas": args.dataset_world_size,
        "rank": args.dataset_rank,
        "num_samples_each_epoch": trainer_args.get("num_samples_each_epoch", 6000000),
        "random_shuffle": True,
        "greedy_intokens": True,
        "packing": True,
        "mix_strategy": "concat",
        "encode_one_turn": True,
        "use_template": True,
        "is_pretraining": False,
    }

    if trainer_args.get("stage") == "sft":
        train_dataset = create_dataset_sft(
            task_group=trainer_args["train_dataset_path"],
            task_group_prob=trainer_args.get("train_dataset_prob", 1.0),
            sub_dataset_type=trainer_args.get("train_dataset_type", "erniekit"),
            **dataset_config,
        )
        eval_dataset = create_dataset_sft(
            task_group=trainer_args["eval_dataset_path"],
            task_group_prob=trainer_args.get("eval_dataset_prob", 1.0),
            sub_dataset_type=trainer_args.get("eval_dataset_type", "erniekit"),
            is_valid=True,
            **dataset_config,
        )
        data_collator = partial(
            collate_fn,
            tokenizer=tokenizer,
            training_args=TrainingArguments(
                output_dir=args.output_dir, num_nextn_predict_layers=args.multi_token_pred_depth
            ),
            model_args=ModelConfig(stage="SFT", use_attn_mask_startend_row_indices=True),
            max_seq_len=args.max_seq_len + 1,
        )
    else:
        train_dataset, eval_dataset, _, data_collator = create_pretrained_dataset(args)

    callbacks = []
    callbacks += [GlobalRNGCallback()]
    callbacks += [OrthogonalCallback(ortho_loss_lambda)] if args.use_ortho_loss_callback else []

    if getattr(cfg, "moe_use_aux_free", 0.0) > 0.0:
        logger.info("adding aux free callback")
        callbacks += [MoECorrectionBiasAdjustCallback(args.moe_use_aux_free_update_coef, args.sequence_parallel)]

    trainer = PretrainingTrainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    global_training_logs.accumulate = args.gradient_accumulation_steps
    checkpoint = None
    if args.resume_from_checkpoint is not None:
        checkpoint = args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics

        # After training, we use unified huggingface format to export the model.
        trainer.args.save_to_hf = True
        trainer.args.save_checkpoint_format = "unified_checkpoint"
        unified_checkpoint.get_expected_state_dict = get_expected_state_dict

        trainer.save_model(args.output_dir)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
