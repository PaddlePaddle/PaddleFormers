# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle

from .log import logger


def static_params_to_dygraph(model, static_tensor_dict):
    """Simple tool for convert static parameters to dygraph parameters dict.

    **NOTE** The model must both support static graph and dygraph mode.

    Args:
        model (nn.Layer): the model of a neural network.
        static_tensor_dict (string): path of which locate the saved parameters in static mode.
            Usually load by `paddle.static.load_program_state`.

    Returns:
        [tensor dict]: a state dict the same as the dygraph mode.
    """
    state_dict = model.state_dict()
    # static_tensor_dict = paddle.static.load_program_state(static_params_path)

    ret_dict = dict()
    for n, p in state_dict.items():
        if p.name not in static_tensor_dict:
            logger.info("%s parameter is missing from you state dict." % n)
            continue
        ret_dict[n] = static_tensor_dict[p.name]

    return ret_dict


def dygraph_params_to_static(model, dygraph_tensor_dict, topo=None):
    """Simple tool for convert dygraph parameters to static parameters dict.

    **NOTE** The model must both support static graph and dygraph mode.

    Args:
        model (nn.Layer): the model of a neural network.
        dygraph_tensor_dict (string): path of which locate the saved parameters in static mode.

    Returns:
        [tensor dict]: a state dict the same as the dygraph mode.
    """
    state_dict = model.state_dict()

    ret_dict = dict()
    for name, parm in state_dict.items():
        if name not in dygraph_tensor_dict:
            logger.info("%s parameter is missing from you state dict." % name)
            continue

        tensor = dygraph_tensor_dict[name]
        if parm.is_distributed:
            assert topo is not None
            for dim, v in enumerate(tensor.shape):
                if parm.shape[dim] != v:
                    break

            splited = np.split(tensor, topo.mp_info.size, axis=dim)[topo.mp_info.rank]
            ret_dict[parm.name] = splited
        else:
            ret_dict[parm.name] = tensor

    return ret_dict


class TimeCostAverage(object):
    """
    Simple tool for calculating time average cost in the process of training and inferencing.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset the recorder state, and reset the `cnt` to zero.
        """
        self.cnt = 0
        self.total_time = 0

    def record(self, usetime):
        """
        Recoding the time cost in current step and accumulating the `cnt`.
        """
        self.cnt += 1
        self.total_time += usetime

    def get_average(self):
        """
        Returning the average time cost after the start of training.
        """
        if self.cnt == 0:
            return 0
        return self.total_time / self.cnt


def get_env_device():
    """
    Return the device name of running environment.
    """
    if paddle.is_compiled_with_cuda():
        return "gpu"
    if paddle.is_compiled_with_rocm():
        return "rocm"
    if paddle.is_compiled_with_xpu():
        return "xpu"
    custom_devices = paddle.device.get_all_custom_device_type()
    if custom_devices:
        return custom_devices[0]
    return "cpu"


def compare_version(version, pair_version):
    """
    Args:
        version (str): The first version string needed to be compared.
            The format of version string should be as follow : "xxx.yyy.zzz".
        pair_version (str): The second version string needed to be compared.
             The format of version string should be as follow : "xxx.yyy.zzz".
    Returns:
        int: The result of comparison. 1 means version > pair_version; 0 means
            version = pair_version; -1 means version < pair_version.

    Examples:
        >>> compare_version("2.2.1", "2.2.0")
        >>> 1
        >>> compare_version("2.2.0", "2.2.0")
        >>> 0
        >>> compare_version("2.2.0-rc0", "2.2.0")
        >>> -1
        >>> compare_version("2.3.0-rc0", "2.2.0")
        >>> 1
    """
    version = version.strip()
    pair_version = pair_version.strip()
    if version == pair_version:
        return 0
    version_list = version.split(".")
    pair_version_list = pair_version.split(".")
    for version_code, pair_version_code in zip(version_list, pair_version_list):
        if not version_code.isnumeric():
            return -1
        if not pair_version_code.isnumeric():
            return 1
        if int(version_code) > int(pair_version_code):
            return 1
        elif int(version_code) < int(pair_version_code):
            return -1
    return 0


def get_bool_ids_greater_than(probs, limit=0.5, return_prob=False):
    """
    Get idx of the last dimension in probability arrays, which is greater than a limitation.

    Args:
        probs (List[List[float]]): The input probability arrays.
        limit (float): The limitation for probability.
        return_prob (bool): Whether to return the probability
    Returns:
        List[List[int]]: The index of the last dimension meet the conditions.
    """
    probs = np.array(probs)
    dim_len = len(probs.shape)
    if dim_len > 1:
        result = []
        for p in probs:
            result.append(get_bool_ids_greater_than(p, limit, return_prob))
        return result
    else:
        result = []
        for i, p in enumerate(probs):
            if p > limit:
                if return_prob:
                    result.append((i, p))
                else:
                    result.append(i)
        return result


def get_span(start_ids, end_ids, with_prob=False):
    """
    Get span set from position start and end list.

    Args:
        start_ids (List[int]/List[tuple]): The start index list.
        end_ids (List[int]/List[tuple]): The end index list.
        with_prob (bool): If True, each element for start_ids and end_ids is a tuple aslike: (index, probability).
    Returns:
        set: The span set without overlapping, every id can only be used once .
    """
    if with_prob:
        start_ids = sorted(start_ids, key=lambda x: x[0])
        end_ids = sorted(end_ids, key=lambda x: x[0])
    else:
        start_ids = sorted(start_ids)
        end_ids = sorted(end_ids)

    start_pointer = 0
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    couple_dict = {}
    while start_pointer < len_start and end_pointer < len_end:
        if with_prob:
            start_id = start_ids[start_pointer][0]
            end_id = end_ids[end_pointer][0]
        else:
            start_id = start_ids[start_pointer]
            end_id = end_ids[end_pointer]

        if start_id == end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            end_pointer += 1
            continue
        if start_id < end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            continue
        if start_id > end_id:
            end_pointer += 1
            continue
    result = [(couple_dict[end], end) for end in couple_dict]
    result = set(result)
    return result
