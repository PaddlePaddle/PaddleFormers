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


from .configuration_utils import PretrainedConfig
from .model_utils import PretrainedModel, register_base_model
from .tokenizer_utils import (
    PretrainedTokenizer,
    BPETokenizer,
    tokenize_chinese_chars,
    is_chinese_char,
    AddedToken,
    normalize_chars,
    tokenize_special_chars,
    convert_to_unicode,
)
from .tokenizer_utils_fast import PretrainedTokenizerFast
from .processing_utils import ProcessorMixin
from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from .image_processing_utils import ImageProcessingMixin
from .attention_utils import create_bigbird_rand_mask_idx_list
from .sequence_parallel_utils import AllGatherVarlenOp, sequence_parallel_sparse_mask_labels
from .tensor_parallel_utils import parallel_matmul, parallel_linear, fused_head_and_loss_fn
from .moe_gate import *
from .moe_layer import *

try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        GatherOp,
        ScatterOp,
        AllGatherOp,
        ReduceScatterOp,
        ColumnSequenceParallelLinear,
        RowSequenceParallelLinear,
        mark_as_sequence_parallel_parameter,
        register_sequence_parallel_allreduce_hooks,
    )
except:
    pass
from .export import export_model

# isort: split
from .bert.modeling import *
from .bert.tokenizer import *
from .bert.configuration import *

# isort: split
from .auto.configuration import *
from .auto.image_processing import *
from .auto.modeling import *
from .auto.processing import *
from .auto.tokenizer import *
from .deepseek_v2 import *
from .deepseek_v3 import *
from .ernie4_5 import *
from .llama import *
from .optimization import *
from .qwen import *
from .qwen2 import *
from .qwen2_moe import *
from .qwen3 import *
from .qwen3_moe import *
