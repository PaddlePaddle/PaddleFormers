# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you smay not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from typing import TYPE_CHECKING

from ..utils.lazy_import import _LazyModule

import_structure = {
    "argparser": ["PdArgumentParser", "strtobool"],
    "auto_training_args": ["AutoTrainingArguments"],
    "plugins.timer": ["_Timer", "RuntimeTimer", "set_timers", "get_timers", "disable_timers", "Timers"],
    "trainer": [
        "TRAINING_ARGS_NAME",
        "strtobool",
        "is_datasets_available",
        "get_last_checkpoint",
        "has_length",
        "distributed_file",
        "broadcast_moe_optimizer",
        "nested_truncate",
        "should_skip_data",
        "set_seed",
        "unwrap_model",
        "distributed_isfile",
        "split_parallel_config",
        "get_fused_param_mappings",
        "in_auto_parallel_align_mode",
        "set_timers",
        "get_reporting_integration_callbacks",
        "speed_metrics",
        "register_sequence_parallel_allreduce_hooks",
        "get_scheduler",
        "nested_concat",
        "default_data_collator",
        "find_batch_size",
        "init_dataloader_comm_group",
        "load_sharded_checkpoint",
        "empty_device_cache",
        "Trainer",
        "broadcast_dataset_rank0_model",
        "split_inputs_sequence_dim",
        "broadcast_dp_optimizer",
        "nested_detach",
        "split_inputs_sequence_dim_load_balance",
        "obtain_optimizer_parameters_list",
        "get_env_device",
        "download_recovery_ckpt_from_pdc",
        "nested_numpify",
        "get_timers",
        "distributed_concat",
        "autocast",
        "fused_allreduce_gradients",
        "is_paddle_cuda_available",
    ],
    "trainer_callback": [
        "CallbackHandler",
        "PrinterCallback",
        "EarlyStoppingCallback",
        "DEFAULT_CALLBACKS",
        "DefaultFlowCallback",
        "TrainerControl",
        "ProgressCallback",
        "TrainerState",
        "DEFAULT_PROGRESS_CALLBACK",
        "TrainerCallback",
        "StepFlexToken",
        "FP8QuantWeightCallback",
        "MoECorrectionBiasAdjustCallback",
        "MoeExpertsGradScaleCallback",
        "MoEGateSpGradSyncCallBack",
    ],
    "trainer_utils": [
        "get_last_checkpoint",
        "EvalPrediction",
        "speed_metrics",
        "SchedulerType",
        "set_hyrbid_parallel_seed",
        "PredictionOutput",
        "IntervalStrategy",
        "get_scheduler",
        "set_seed",
        "TrainOutput",
        "log_trainer_start",
    ],
    "training_args": ["default_logdir", "TrainingArguments"],
}

if TYPE_CHECKING:
    from .argparser import *
    from .auto_training_args import *
    from .plugins.timer import *
    from .trainer import *
    from .trainer_callback import *
    from .trainer_utils import *
    from .training_args import *
else:
    from ..utils import logger

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
        extra_objects={"logger": logger},  # 额外传入
    )
