# !/usr/bin/env python3

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

"""
logging callback
"""

import os

from paddleformers.trainer.trainer_callback import TrainerCallback
from paddleformers.utils.log import logger


class LoggingCallback(TrainerCallback):
    """
    A bare [`TrainerCallback`] that just prints the logs.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def on_train_begin(self, *args, **kwargs):
        """
        Callback method executed at the start of training.

        Args:
            *args: Variable length positional arguments, not used in practice.
            **kwargs: Variable length keyword arguments, not used in practice.

        Returns:
            None
        """

        def _log(x):
            return os.popen(x).read()

        logger.info(
            f"""
                HOSTNAME: {os.environ.get('HOSTNAME')}
                CLUSTER_NAME: {os.environ.get('CLUSTER_NAME')}

                CPU INFO
                    [ ] BASIC: \n{_log('lscpu')}

                GPU INFO
                    [ ] NVIDIA-SMI: \n{_log('nvidia-smi |grep NVIDIA-SMI')}
                    [ ] GPU MEMORY: \n{_log('nvidia-smi |grep Default')}

                DISK INFO
                    [ ] BASIC: \n{_log('lsblk')}

                MEMORY INFO
                    [ ] BASIC: \n{_log('free -h')}

                ENVS:
                    [ ] BASIC: \n{_log('env')}
                """
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Processes log information, including converting data_id and src_id to strings
        and adding them to the logs.
        If the `metrics_dumper` argument is provided, the log information is added to it.

        Args:
            args (Any, optional): Optional argument, defaults to None.
            state (Any, optional): Optional argument, defaults to None.
            control (Any, optional): Optional argument, defaults to None.
            logs (List[Dict], optional): Optional argument, defaults to None.
            A list of logs in dictionary format, where each dictionary contains a set of key-value pairs.
            kwargs (Dict, optional): Optional argument, defaults to an empty dict.
            Additional keyword arguments. Contains the `inputs` key,
            whose value is a dictionary that includes `data_id` and `src_id`,
            representing the data ID and source ID respectively.

        Returns:
            None: This function does not return any value.

        Raises:
            None: This function does not raise any exceptions.
        """
        _ = logs.pop("total_flos", None)
        if "inputs" in kwargs:
            data_id = kwargs["inputs"].get("data_id", None)
            src_id = kwargs["inputs"].get("src_id", None)
            data_type = kwargs["inputs"].get("data_type", None)

            if data_id is not None:
                logs = dict(
                    logs, data_id="-".join(map(str, (data_id.numpy().tolist())))
                )
            if src_id is not None:
                logs = dict(logs, src_id="-".join(map(str, (src_id.numpy().tolist()))))
            if data_type is not None:
                logs.update(data_type="-".join(map(str, (data_type.numpy().tolist()))))

        if type(logs) is dict:
            logger.info(
                ", ".join(
                    (
                        (
                            f"{k}: {v}"
                            if k == "loss" or "cur_dp" in k
                            else f"{k}: {v:e}" if v < 1e-3 else f"{k}: {v:f}"
                        )
                        if isinstance(v, float)
                        else f"{k}: {v}"
                    )
                    for k, v in logs.items()
                )
            )
            metrics_dumper = kwargs.get("metrics_dumper", None)
            if metrics_dumper is not None:
                metrics_dumper.append(logs)
        else:
            logger.info(logs)
