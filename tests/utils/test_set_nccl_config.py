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

import unittest
from pathlib import Path

from paddle.distributed import fleet

from paddleformers.trainer import PdArgumentParser, TrainingArguments
from paddleformers.trainer.trainer_utils import init_nccl_config

try:
    from paddle.distributed import create_nccl_config
except ImportError:
    create_nccl_config = None

nccl_config = """
{
    "default": {
        "proto": "ll",
        "n_channels": 1
    },
    "pp": {
        "ll_buffsize": 131072,
        "ll128_buffsize": 134217728,
        "simple_buffsize": 134217728
    },
    "tp": {
        "proto": "ll,ll128"
    }
}
"""


class TestNcclConfig(unittest.TestCase):
    def setUp(self):
        self.nccl_config_path = "/tmp/config.json"
        self.output_path = "/tmp/paddleformers_test_output"
        Path("/tmp/config.json").write_text(nccl_config)

    def test_nccl_config(self):
        # paddle version does not match
        if create_nccl_config is None:
            return
        args_dict = {"output_dir": self.output_path, "nccl_comm_group_config": self.nccl_config_path}

        parser = PdArgumentParser((TrainingArguments,))
        (args,) = parser.parse_dict(args_dict)
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {"dp_degree": 2, "mp_degree": 2, "pp_degree": 2}

        strategy = init_nccl_config(self.nccl_config_path, strategy)
        assert strategy.hybrid_configs["default_comm_group_configs"].nccl_config.protoStr == "ll"
        assert strategy.hybrid_configs["default_comm_group_configs"].nccl_config.nchannels == 1
        assert strategy.hybrid_configs["default_comm_group_configs"].nccl_config.buffsize_align == 1024
        assert strategy.hybrid_configs["mp_configs"].nccl_config.protoStr == "ll,ll128"
        assert strategy.hybrid_configs["mp_configs"].nccl_config.buffsize_align == 1024
        assert strategy.hybrid_configs["pp_configs"].coll_nccl_config.ll_buffsize == 131072
        assert strategy.hybrid_configs["pp_configs"].coll_nccl_config.ll128_buffsize == 134217728
        assert strategy.hybrid_configs["pp_configs"].coll_nccl_config.simple_buffsize == 134217728

    def tearDown(self):
        file_path = Path(self.nccl_config_path)
        if file_path.exists():
            file_path.unlink()
        output_path = Path(self.output_path)
        if output_path.exists():
            output_path.unlink()
