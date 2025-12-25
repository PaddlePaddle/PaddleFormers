# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from dataclasses import dataclass, field

from .trainer_utils import ShardingOption, split_parallel_config
from .training_args import TrainingArguments
from .utils import add_start_docstrings


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class AutoTrainingArguments(TrainingArguments):
    """
    Training Arguments for auto_parallel.
    """

    fused_linear: bool = field(
        default=False,
        metadata={"help": "Enable fused linear op, which will fuse matmul and bias add together."},
    )

    fused_linear_param_grad_add: bool = field(
        default=False,
        metadata={
            "help": "Enable fused_linear_param_grad pass, which should replace add_n_op with add_op for gradients accumulation."
        },
    )
    fuse_allreduce_split_to_reducescatter: bool = field(
        default=False,
        metadata={"help": "Enable fuse_allreduce_split_to_reducescatter pass."},
    )
    eliminate_transpose: bool = field(
        default=False,
        metadata={
            "help": "Enable eliminate_transpose pass, which should replace transpose with reshape when sequence parallel is enabled."
        },
    )
    use_intermediate_api: bool = field(
        default=False,
        metadata={"help": "Weather to use auto_parallel intermediate api"},
    )
    refined_ops_patterns: str = field(default=None, metadata={"help": "The pattern of refined recompute."})
    load_model_with_sharding_tensor_fusion: bool = field(
        default=False,
        metadata={
            "help": (
                "When using sharding stage1, enabling tensor fusion, and setting `load_model_with_sharding_tensor_fusion` to `True`, "
                "the model is loaded with unbalanced weights, meaning that the model weights are stored in an unbalanced format to avoid "
                "additional memory overhead. If set to `False`, the model will be loaded with balanced weights, which may increase memory "
                "consumption. This setting is only available in auto parallel to_static mode."
            )
        },
    )
    save_model_with_sharding_tensor_fusion: bool = field(
        default=False,
        metadata={
            "help": (
                "When using sharding stage1 and enabling tensor fusion, setting `save_model_with_sharding_tensor_fusion` to `True` "
                "saves the model with unbalanced weights, which helps avoid additional memory consumption. Setting it to `False` "
                "saves the model with balanced weights, which may increase memory usage but ensures uniform parameter distribution. "
                "This option allows flexibility in choosing the save format based on memory requirements. "
                "This setting is only available in auto parallel to_static mode."
            )
        },
    )
    job_schedule_profiler_start: int = field(
        default=-1,
        metadata={"help": "The step to start job_schedule_profiler."},
    )
    job_schedule_profiler_end: int = field(
        default=-1,
        metadata={"help": "The step to end job_schedule_profiler."},
    )

    auto_parallel_sync_shared_params: bool = field(
        default=False,
        metadata={
            "help": (
                "Optimize the parameter sharing between two stages in a pipeline parallel scenario, setting `auto_parallel_sync_shared_params` to `True`. "
                "It constructs identical parameters on the two stages, synchronizes gradients through allreduce, and then performs optimizer optimization"
                " independently on each stage. Currently, only one shared parameter is supported."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        assert self.enable_auto_parallel

        fused_passes = self.strategy.fused_passes

        if self.fused_linear_param_grad_add:
            fused_passes.enable = True
            fused_passes.fused_passes_list.append("fused_linear_param_grad_add_pass")

        if self.fuse_allreduce_split_to_reducescatter:
            fused_passes.enable = True
            fused_passes.fused_passes_list.append("fuse_allreduce_split_to_reducescatter_pass")

        if self.eliminate_transpose:
            fused_passes.enable = True
            fused_passes.fused_passes_list.append("eliminate_transpose")

        if self.fused_linear:
            fused_passes.enable = True
            fused_passes.fused_passes_list.append("fused_gemm_epilogue_pass")

        mp_configs = split_parallel_config(self.tensor_parallel_config)
        if "replace_with_parallel_cross_entropy" in mp_configs:
            self.strategy.mp_optimization.replace_with_parallel_cross_entropy = True

        pp_configs = split_parallel_config(self.pipeline_parallel_config)
        if "auto_parallel_sync_shared_params" in pp_configs:
            self.strategy.pipeline.auto_parallel_sync_shared_params = True

        if self.recompute:
            recompute = self.strategy.recompute
            recompute.enable = True
            recompute.refined_ops_patterns = []
            if type(self.refined_ops_patterns) == str:
                recompute.refined_ops_patterns = json.loads(self.refined_ops_patterns)
            else:
                recompute.refined_ops_patterns = (
                    self.refined_ops_patterns if self.refined_ops_patterns is not None else []
                )

    @property
    def should_load_model_with_tensor_fusion(self):
        return (
            self.enable_auto_parallel
            and self.to_static
            and ShardingOption.SHARD_OP in self.sharding
            and self.sharding_parallel_degree > 1
            and "enable_tensor_fusion" in self.sharding_parallel_config
        )
