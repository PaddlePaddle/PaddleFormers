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

# Referred to NVIDIA Megatron-LM https://github.com/NVIDIA/Megatron-LM.git
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import paddle
import pytest

from paddleformers.fleet.tensor_parallel.random import (
    CudaRNGStatesTracker,
    get_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from tests.multi_card_tests.tensor_parallel.test_utilities import Utils


def test_cuda_rng_states_tracker():
    rng_tracker = CudaRNGStatesTracker()
    rng_tracker.set_states({"state1": 1234})
    assert rng_tracker.get_states()["state1"] == 1234
    rng_tracker.reset()
    assert rng_tracker.get_states() == {}
    seed = 1111
    rng_tracker.add("state2", seed)
    with pytest.raises(ValueError):
        assert rng_tracker.add("state3", seed)
    with pytest.raises(ValueError):
        assert rng_tracker.add("state2", 111)
    assert rng_tracker.get_states()["state2"] is not None

    rng_tracker.fork("state2")
    paddle.cuda.manual_seed(seed)
    rng_state = paddle.cuda.get_rng_state()
    assert rng_tracker.get_states()["state2"].current_seed() == rng_state.current_seed()


def test_model_parallel_cuda_manual_seed():
    Utils.initialize_model_parallel(4, 1)
    model_parallel_cuda_manual_seed(0)
    rng_tracker = get_cuda_rng_tracker()
    assert rng_tracker.get_states()["model-parallel-rng"] is not None


if __name__ == "__main__":
    test_cuda_rng_states_tracker()
    test_model_parallel_cuda_manual_seed()
