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

import copy
import json
from dataclasses import dataclass

from paddle.nn.quant.quantized_linear import _get_arch_info

quant_inference_mapping = {"avg": "abs_max", "abs_max_channel_wise": "abs_max_channel_wise", "abs_max": "abs_max"}
fp8_format_mapping = {
    "hybrid": {"weight": "float8_e4m3fn", "activation": "float8_e4m3fn", "grad_output": "float8_e5m2"},
    "e4m3": {"weight": "float8_e4m3fn", "activation": "float8_e4m3fn", "grad_output": "float8_e4m3fn"},
}


@dataclass
class QuantizationConfig:
    """
    This is the configuration class to store quantization configuration.
    Args:
        weight_quantize_algo: Weight quantization algorithm.
        quant_type: Quantization type applied to weight and activation, weight may still keep in float tensor.
        shift: Whether the model applied the shift strategy.
        smooth: Whether the model applied the smooth strategy.
        shift_smooth_all_linears: Whether the model applied shift or smooth strategy for all linears.
        quant_round_type: The quant round type, 0:-rounding to nearest ties to even， 1: -rounding to nearest ties away from zero.
        llm_int8_threshold: The threshold for llm.int8 quantization.
        qlora_weight_double_quant: Whether quant weight scale.
        qlora_weight_blocksize: Block size for weight quantization.
        qlora_weight_double_quant_block_size: Block size for weight_scale of weight weight_scale.
        weight_quant_method: The method for weight quantization.
        act_quant_method: The method for activation quantization.
        apply_online_actscale_step: Use online (per-step) activation scales for the first N steps. During these steps, activation scales are also collected to compute their mean for later use.
    """

    def __init__(
        self,
        weight_quantize_algo=None,
        quant_type=None,
        shift=False,
        smooth=False,
        shift_smooth_all_linears=False,
        quant_round_type=0,
        llm_int8_threshold=6.0,
        qlora_weight_double_quant=False,
        qlora_weight_blocksize=64,
        qlora_weight_double_quant_block_size=256,
        weight_quant_method="abs_max_channel_wise",
        act_quant_method="abs_max",
        activation_scheme=None,
        fmt=None,
        quant_method=None,
        weight_block_size=None,
        dtype=None,
        ignore_modules=None,
        group_size=-1,
        apply_hadamard=False,
        hadamard_block_size=32,
        quant_input_grad=False,
        quant_weight_grad=False,
        apply_online_actscale_step=200,
        actscale_moving_rate=0.01,
        fp8_format_type="hybrid",
        scale_epsilon=1e-8,
        dense_quant_type="",
        moe_quant_type="",
        quantization="",
        **kwargs,
    ):
        if weight_quantize_algo is not None:
            if isinstance(weight_quantize_algo, dict):
                if any(
                    algo
                    not in [
                        "weight_only_int8",
                        "weight_only_int4",
                        "llm.int8",
                        "a8w8",
                        "nf4",
                        "fp4",
                        "a8w8linear",
                        "a8w4linear",
                        "fp8linear",
                    ]
                    for algo in weight_quantize_algo
                ):
                    raise ValueError(
                        f"weight_quantize_algo:{weight_quantize_algo.keys()} not in supported list ['weight_only_int8', 'weight_only_int4', 'llm.int8', 'a8w8', 'nf4', 'fp4']"
                    )
            elif weight_quantize_algo not in [
                "weight_only_int8",
                "weight_only_int4",
                "llm.int8",
                "a8w8",
                "nf4",
                "fp4",
                "a8w8linear",
                "a8w4linear",
                "fp8linear",
            ]:
                raise ValueError(
                    f"weight_quantize_algo:{weight_quantize_algo} not in supported list ['weight_only_int8', 'weight_only_int4', 'llm.int8', 'a8w8', 'nf4', 'fp4']"
                )
        if (
            (isinstance(weight_quantize_algo, dict) and "fp8linear" in weight_quantize_algo)
            or weight_quantize_algo == "fp8linear"
        ) and _get_arch_info() not in [89, 90]:
            raise RuntimeError("fp8Linear is only supported on NVIDIA Hopper GPUs.")
        if quant_type is not None and quant_type not in [
            "weight_only_int8",
            "weight_only_int4",
            "a8w8",
            "a8w8c8",
            "a8w8_fp8",
            "a8w8c8_fp8",
        ]:
            raise ValueError(
                f"quant_type:{quant_type} not in supported list ['weight_only_int8', 'weight_only_int4', 'a8w8', 'a8w8c8', 'a8w8_fp8', 'a8w8c8_fp8']"
            )
        self.weight_quantize_algo = weight_quantize_algo
        self.quant_type = quant_type
        self.shift = shift
        self.smooth = smooth
        self.shift = shift
        self.shift_smooth_all_linears = shift_smooth_all_linears
        self.quant_round_type = quant_round_type
        self.llm_int8_threshold = llm_int8_threshold
        self.qlora_weight_double_quant = qlora_weight_double_quant
        self.qlora_weight_blocksize = qlora_weight_blocksize
        self.weight_quant_method = weight_quant_method
        self.act_quant_method = quant_inference_mapping[act_quant_method]
        self.qlora_weight_double_quant_block_size = qlora_weight_double_quant_block_size
        self.activation_scheme = activation_scheme
        self.fmt = fmt
        self.quant_method = quant_method
        self.weight_block_size = weight_block_size
        self.dtype = dtype
        self.ignore_modules = ignore_modules
        self.group_size = group_size
        self.apply_hadamard = apply_hadamard
        self.hadamard_block_size = hadamard_block_size
        self.quant_input_grad = quant_input_grad
        self.quant_weight_grad = quant_weight_grad
        self.apply_online_actscale_step = apply_online_actscale_step
        self.actscale_moving_rate = actscale_moving_rate
        self.fp8_format_type = fp8_format_type
        self.scale_epsilon = scale_epsilon
        self.dense_quant_type = dense_quant_type
        self.moe_quant_type = moe_quant_type
        self.quantization = quantization

    @property
    def fp8_format(self):
        return fp8_format_mapping[self.fp8_format_type]

    def is_weight_quantize(self):
        if isinstance(self.weight_quantize_algo, dict):
            return True
        elif self.weight_quantize_algo in [
            "weight_only_int8",
            "weight_only_int4",
            "llm.int8",
            "nf4",
            "fp4",
            "a8w8",
            "a8w8linear",
            "a8w4linear",
            "fp8linear",
        ]:
            return True
        else:
            return False

    def is_support_merge_tensor_parallel(self):
        if self.weight_quantize_algo in ["weight_only_int8", "weight_only_int4", "llm.int8", "a8w8"]:
            return False
        else:
            return True

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
        """
        Instantiates QuantizationConfig from dict
        """
        config = cls(**config_dict)

        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    def to_json_file(self, json_file_path):
        """
        Save this instance to a JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json_string(self, use_diff=True):
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_diff_dict(self):
        config_dict = self.to_dict()

        # get the default config dict

        default_config_dict = QuantizationConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict
