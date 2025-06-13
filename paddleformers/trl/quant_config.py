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

from dataclasses import dataclass, field
from typing import List, Optional

__all__ = ["QuantConfig"]


@dataclass
class QuantConfig:

    # Quantization method config
    quant_type: str = field(
        default="a8w8",
        metadata={"help": "Quantization type. Supported values: weight_only_int8, weight_only_int4, a8w8, a8w8c8"},
    )

    fp8_type: List[str] = field(
        default_factory=lambda: ["e4m3", "e4m3"],
        metadata={"help": "Quantization type for (activation, weight)", "nargs": "+"},
    )

    skip_list_names: List[str] = field(
        default=lambda: [], metadata={"help": "Skip scales for quantization", "nargs": "+"}
    )

    weight_quant_method: str = field(
        default="abs_max_channel_wise",
        metadata={"help": "Weight quantization method, chosen from ['abs_max_channel_wise', 'groupwise']"},
    )

    act_quant_method: str = field(
        default="avg",
        metadata={"help": "Activation quantization method, chosen from ['abs_max', 'avg']"},
    )

    cachekv_quant_method: str = field(
        default="avg_headwise",
        metadata={"help": "KV quantization method, chosen from ['abs_max_headwise', 'avg_headwise']"},
    )

    # Piecewise Search Smooth related parameters
    search_alpha_min: float = field(
        default=0.2,
        metadata={"help": "The minimum alpha for piece search"},
    )

    search_alpha_max: float = field(
        default=0.8,
        metadata={"help": "The maximum alpha for piece search"},
    )

    search_scale_min: float = field(
        default=1.0,
        metadata={"help": "The minimum scale for piece search"},
    )

    search_scale_max: float = field(
        default=5.0,
        metadata={"help": "The maximum scale for piece search"},
    )

    # QAT related parameters
    # Not Yet support
    do_qat: bool = field(default=False, metadata={"help": "Whether to use QAT technique"})

    # PTQ related parameters
    do_ptq: bool = field(default=False, metadata={"help": "Whether to use PTQ"})
    ptq_step: int = field(default=32, metadata={"help": "Step for PTQ"})

    # Pre-quant method Shift related parameters
    shift: bool = field(default=False, metadata={"help": "Whether to use Shift"})
    shift_all_linears: bool = field(default=False, metadata={"help": "Whether to shift all linears"})
    shift_sampler: str = field(
        default="ema", metadata={"help": "The name of shift sampler, chosen from ['ema', 'none']"}
    )
    shift_step: int = field(default=32, metadata={"help": "Sample steps when shift"})

    # Pre-quant methods Smooth related parameters
    smooth: bool = field(default=False, metadata={"help": "Whether to use Smooth"})
    smooth_all_linears: bool = field(default=False, metadata={"help": "Whether to smooth all linears"})
    smooth_sampler: str = field(
        default="none", metadata={"help": "The name of smooth sampler, chosen from ['multi_step','none']"}
    )
    smooth_step: int = field(default=32, metadata={"help": "Sample steps when smooth"})
    smooth_piecewise_search: bool = field(
        default=False, metadata={"help": "The number of piece in piecewise search for smooth strategy."}
    )
    smooth_k_piece: int = field(default=3, metadata={"help": "Number of pieces for K-search"})
    smooth_search_piece: bool = field(default=False, metadata={"help": "Whether search k_piece when piecewise search"})

    # GPTQ related parameters
    do_gptq: bool = field(default=False, metadata={"help": "Whether to use GPTQ"})
    gptq_step: int = field(default=8, metadata={"help": "Step for GPTQ"})

    # AWQ related parameters, default for WINT4
    do_awq: bool = field(default=False, metadata={"help": "Whether to use AWQ Search"})
    auto_clip: bool = field(default=False, metadata={"help": "Whether to use AutoClip from AWQ"})
    awq_step: int = field(default=8, metadata={"help": "Step for AWQ Search"})
    autoclip_step: int = field(default=8, metadata={"help": "Step for AutoClip"})

    # Other config
    load_quant_model: bool = field(default=False, metadata={"help": "Whether to load quant model"})

    do_quant_debug: bool = field(default=False, metadata={"help": "Whether to use debug"})
    test_sample: Optional[str] = field(default=None, metadata={"help": "Test sample for quantization"})
