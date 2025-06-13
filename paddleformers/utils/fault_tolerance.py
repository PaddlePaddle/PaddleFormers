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

import os

try:
    from paddle.framework.recall_error import LOSS_NAN_ERROR
except ImportError:
    LOSS_NAN_ERROR = "PaddleRecall error(102): LossNan"

try:
    from paddle.framework.recall_error import LOSS_INF_ERROR
except ImportError:
    LOSS_INF_ERROR = "PaddleRecall error(104): LossInf"

PDC_DOWNLOAD_ERROR = "PaddleRecall error(105): PDCDownloadError"
# only warn msg
FC_DUMP_ERROR = "PaddleRecall error(106): FlashCheckpointDumpError"
# fatal error, must be fixed by babysitters
PC_DUMP_ERROR = "PaddleRecall error(107): PersistentCheckpointDumpError"


def is_ft_env():
    """
    Check if the current environment is a FT environment.
    """
    return "PDC_LONGJOB_ID" in os.environ
