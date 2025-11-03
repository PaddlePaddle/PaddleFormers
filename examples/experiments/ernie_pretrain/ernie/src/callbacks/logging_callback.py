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

import logging

from paddleformers.trainer.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)


class LoggingCallback(TrainerCallback):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if "inputs" in kwargs:
            data_id = kwargs["inputs"].get("data_id", None)
            src_id = kwargs["inputs"].get("src_id", None)
            data_type = kwargs["inputs"].get("data_type", None)

            if data_id is not None:
                logs = dict(logs, data_id="-".join(map(str, (data_id.numpy().tolist()))))
            if src_id is not None:
                logs = dict(logs, src_id="-".join(map(str, (src_id.numpy().tolist()))))
            if data_type is not None:
                logs.update(data_type="-".join(map(str, (data_type.numpy().tolist()))))

        if type(logs) is dict:
            logger.info(
                ", ".join(
                    (
                        (f"{k}: {v}" if k == "loss" or "cur_dp" in k else f"{k}: {v:e}" if v < 1e-3 else f"{k}: {v:f}")
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
