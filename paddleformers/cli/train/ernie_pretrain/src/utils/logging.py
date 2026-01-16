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
import sys
from pathlib import Path

from paddleformers.utils.log import logger as paddlenlp_logger

hdl = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter(fmt="[%(levelname)s] %(asctime)s [%(filename)12s:%(lineno)5d]:    %(message)s")
hdl.setFormatter(formatter)
logger = logging.getLogger()
logger.handlers = [hdl]

bce_log = logging.getLogger("baidubce")
bce_log.handlers = []
bce_log.propagate = False
logger.setLevel(10)

bce_bns_proxy_log = logging.getLogger("bce_bns_proxy.wrapper")
bce_bns_proxy_log.disabled = True
filelock_log = logging.getLogger("filelock")
filelock_log.disabled = True

paddlenlp_logger.logger.handlers = []
paddlenlp_logger.logger.propagate = True


def setup_logger_output_file(outputpath, local_rank):
    logdir = Path(outputpath) / "log"
    logdir.mkdir(exist_ok=True)
    file_hdl = logging.FileHandler(logdir / f"workerlog.{local_rank}", mode="a", encoding="utf-8")
    formatter = logging.Formatter(
        fmt=f"[%(levelname)s] %(asctime)s [%(filename)12s:%(lineno)5d][rank-{local_rank}]:    %(message)s"
    )
    file_hdl.setFormatter(formatter)
    hdl.setFormatter(formatter)
    logger.handlers = [hdl, file_hdl]
