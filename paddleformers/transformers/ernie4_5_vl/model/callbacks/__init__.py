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
""" Trainer Callback functions """

from .layerwise_dropout_callback import LayerwiseDropoutCallback
from .adaptivegradclip_callback import ClipGradByAdaptiveNormCallback
from .gc_callback import GCCallback
from .logging_callback import LoggingCallback
from .moe_correction_bias_adjust_callback import MoECorrectionBiasAdjustCallback
from .moe_logging_callback import GlobalRNGCallback
from .multimodal_interleave_callback import MultiModalInterleaveCallback
from .optimizer_callback import OptimizerCallback
from .ortho_loss_callback import OrthogonalCallback
from .pp_need_data_callback import PPNeedDataCallback
from .refinedrecompute_check_callback import RefinedRecomputeCheckCallback
from .reshard_save_then_exit_callback import ReshardSaveExitCallback
from .sp_grad_sync_callback import SPGradSyncCallback
from .stopper_callback import StopperCallback
from .tensorboard_callback import TensorBoardCallback
from .vit_trainable_callback import VitTrainableCallback

__all__ = [
    "LayerwiseDropoutCallback",
    "SPGradSyncCallback",
    "ClipGradByAdaptiveNormCallback",
    "GCCallback",
    "LoggingCallback",
    "MoECorrectionBiasAdjustCallback",
    "GlobalRNGCallback",
    "MultiModalInterleaveCallback",
    "OptimizerCallback",
    "OrthogonalCallback",
    "PPNeedDataCallback",
    "RefinedRecomputeCheckCallback",
    "ReshardSaveExitCallback",
    "StopperCallback",
    "TensorBoardCallback",
    "VitTrainableCallback",
]
