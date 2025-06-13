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

import time
import warnings
from abc import ABC
from copy import deepcopy
from typing import Optional

import paddle


class StoppingCriteria(ABC):
    """
    Abstract base class for all stopping criteria that can be applied during
    generation.
    """

    def __call__(self, input_ids: paddle.Tensor, logits: paddle.Tensor, **kwargs):
        raise NotImplementedError(f"{self.__class__} is an abstract class. " "StoppingCriteria needs to be subclassed")


class MaxTimeCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generation exceeds some amount of time. By default, the
    time will start being counted when you initialize this function. You can override this by passing an
    `initial_time`.

    Args:
        max_time (`float`):
            The maximum allowed time in seconds for the generation.
        initial_time (`float`, *optional*, defaults to `time.time()`):
            The start of the generation allowed time.
    """

    def __init__(self, max_time: float, initial_timestamp: Optional[float] = None):
        self.max_time = max_time
        self.initial_timestamp = time.time() if initial_timestamp is None else initial_timestamp

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor, **kwargs) -> bool:
        return time.time() - self.initial_timestamp > self.max_time


class MaxLengthCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generated number of tokens exceeds `max_length`. Keep
    in mind for decoder-only type of transformers, [this will include the initial prompted tokens].

    Args:
        max_length (`int`):
            The maximum length that the output sequence can have in number of tokens.
    """

    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor, **kwargs) -> bool:
        return input_ids.shape[-1] >= self.max_length


class StoppingCriteriaList(list):
    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor, **kwargs):
        return any(criteria(input_ids, scores) for criteria in self)

    @property
    def max_length(self):
        for stopping_criterium in self:
            if isinstance(stopping_criterium, MaxLengthCriteria):
                return stopping_criterium.max_length
        return None


def validate_stopping_criteria(stopping_criteria: StoppingCriteriaList, max_length: int) -> StoppingCriteriaList:
    stopping_max_length = stopping_criteria.max_length
    new_stopping_criteria = deepcopy(stopping_criteria)
    if stopping_max_length is not None and stopping_max_length != max_length:
        warnings.warn("You set different `max_length` for stopping criteria and `max_length` parameter", UserWarning)
    elif stopping_max_length is None:
        new_stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
    return new_stopping_criteria
