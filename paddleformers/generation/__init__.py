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

import sys
from typing import TYPE_CHECKING

from ..utils.lazy_import import _LazyModule

import_structure = {
    "utils": [
        "GenerationMixin",
        "MinLengthLogitsProcessor",
        "convert_dtype",
        "get_unfinished_flag",
        "LogitsProcessor",
        "BeamHypotheses",
        "RepetitionPenaltyLogitsProcessor",
        "LogitsProcessorList",
        "TopKProcess",
        "map_structure",
        "BeamSearchScorer",
        "TopPProcess",
        "get_scale_by_dtype",
        "validate_stopping_criteria",
    ],
    "model_outputs": ["ModelOutput"],
    "configuration_utils": ["GenerationConfig", "resolve_hf_generation_config_path"],
    "logits_process": [
        "MinLengthLogitsProcessor",
        "SequenceBiasLogitsProcessor",
        "NoRepeatNGramLogitsProcessor",
        "PrefixConstrainedLogitsProcessor",
        "TopPProcess",
        "LogitsWarper",
        "HammingDiversityLogitsProcessor",
        "ForcedEOSTokenLogitsProcessor",
        "ForcedBOSTokenLogitsProcessor",
        "LogitsProcessor",
        "RepetitionPenaltyLogitsProcessor",
        "TemperatureLogitsWarper",
        "TopKProcess",
        "_get_ngrams",
        "_get_generated_ngrams",
        "LogitsProcessorList",
        "NoBadWordsLogitsProcessor",
        "_calc_banned_ngram_tokens",
    ],
    "stopping_criteria": [
        "validate_stopping_criteria",
        "StoppingCriteria",
        "MaxLengthCriteria",
        "StoppingCriteriaList",
        "MaxTimeCriteria",
    ],
    "streamers": ["BaseStreamer", "TextIteratorStreamer", "TextStreamer"],
}

if TYPE_CHECKING:
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
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
