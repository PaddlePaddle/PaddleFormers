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

from __future__ import annotations

import logging
import os
from itertools import chain

logger = logging.getLogger(__name__)

g_has_print_recovery_log = False


def has_recovered():
    """has recovered"""
    recover_step = os.getenv("RECOVER_STEP")
    if recover_step is None:
        return True
    recover_step = int(recover_step)
    current_step = os.getenv("TRAINER_GLOBAL_STEP")
    if current_step is None:
        current_step = os.getenv("PDC_INIT_STEP")
        assert current_step is not None, (
            "TRAINER_GLOBAL_STEP or PDC_INIT_STEP should be specified"
        )
    current_step = int(current_step)
    if current_step > recover_step:
        global g_has_print_recovery_log
        if not g_has_print_recovery_log:
            logger.info(f"Recovery would be enabled in the step {current_step}")
            g_has_print_recovery_log = True
        return True
    else:
        return False


def need_recompute_in_block(layer_number, config, recompute_num_layers):
    assert recompute_num_layers is not None, (
        "recompute_num_layers cannot be none"
    )

    if recompute_num_layers < 0:
        return True

    total_num_hidden_layers = (
        config.num_empty_layers_add_in_head
        + config.num_hidden_layers
        + config.num_empty_layers_add_in_tail
    )
    vpp_size = (
        config.virtual_pipeline_model_parallel_size
        if config.virtual_pipeline_model_parallel_size
        else 1
    )
    parallel_size = config.pipeline_model_parallel_size * vpp_size
    assert total_num_hidden_layers % parallel_size == 0, (
        "num_hidden_layers must be divided by parallel_size"
    )
    chunk_size = int(total_num_hidden_layers / parallel_size)
    assert recompute_num_layers <= chunk_size
    layers = list(range(total_num_hidden_layers))
    recompute_layers = list(
        chain.from_iterable(
            [
                layers[i : i + recompute_num_layers]
                for i in range(0, len(layers), chunk_size)
            ]
        )
    )
    if layer_number in recompute_layers:
        return True
    return False


def need_recompute_in_first_n(layer_number, config, recompute_num_layers):
    assert recompute_num_layers is not None, (
        "recompute_num_layers cannot be none"
    )
    total_num_hidden_layers = (
        config.num_empty_layers_add_in_head
        + config.num_hidden_layers
        + config.num_empty_layers_add_in_tail
    )
    vpp_size = (
        config.virtual_pipeline_model_parallel_size
        if config.virtual_pipeline_model_parallel_size
        else 1
    )
    parallel_size = config.pipeline_model_parallel_size * vpp_size
    assert total_num_hidden_layers % parallel_size == 0, (
        "num_hidden_layers must be divided by parallel_size"
    )
    chunk_size = int(total_num_hidden_layers / parallel_size)
    num_layers_in_each_stage = (
        total_num_hidden_layers / config.pipeline_model_parallel_size
    )
    assert recompute_num_layers <= num_layers_in_each_stage, (
        "recompute_num_layers cannot be greater than num_layers_in_each_stage"
    )
    if vpp_size > 1:
        layers = range(total_num_hidden_layers)
        chunks = [
            layers[i * chunk_size : (i + 1) * chunk_size]
            for i in range(0, len(layers), chunk_size)
        ]
        recompute_layers = []
        for pp_stage in range(config.pipeline_model_parallel_size):
            recompute_layers_in_curr_stage = list(
                chain.from_iterable(
                    chunks[pp_stage :: config.pipeline_model_parallel_size]
                )
            )[:recompute_num_layers]
            recompute_layers += recompute_layers_in_curr_stage
    else:
        recompute_layers = []
        layers = list(range(total_num_hidden_layers))
        if config.pipeline_model_parallel_size > 1:
            for recompute_layer_id in range(recompute_num_layers):
                recompute_layers_in_curr_stage = list(
                    layers[recompute_layer_id::chunk_size]
                )
                recompute_layers += recompute_layers_in_curr_stage
        else:
            recompute_layers = list(
                range(
                    config.pipeline_model_parallel_size * recompute_num_layers
                )
            )
    if layer_number in recompute_layers:
        return True
    return False


def need_full_recompute(layer_number, config):
    if config.recompute_granularity == "full":
        if config.recompute_method == "uniform":
            assert config.recompute_num_layers == 1, (
                "don't support recompute_method=uniform wihile recompute_num_layers != 1"
            )
            return True
        elif config.recompute_method == "first_n":
            return need_recompute_in_first_n(
                layer_number, config, config.recompute_num_layers
            )
        elif config.recompute_method == "block":
            return need_recompute_in_block(
                layer_number, config, config.recompute_num_layers
            )
    return False
