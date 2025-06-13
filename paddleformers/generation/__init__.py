# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from .configuration_utils import GenerationConfig
from .logits_process import (
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    LogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TopKProcess,
    TopPProcess,
)
from .stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from .streamers import BaseStreamer, TextIteratorStreamer, TextStreamer
from .utils import BeamSearchScorer, GenerationMixin, get_unfinished_flag
