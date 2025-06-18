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
import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from unittest.mock import patch

from paddleformers.trainer import TrainingArguments
from paddleformers.trainer.argparser import PdArgumentParser
from paddleformers.trainer.utils.doc import add_start_docstrings
from paddleformers.transformers.configuration_utils import llmmetaclass


@dataclass
@llmmetaclass
@add_start_docstrings(TrainingArguments.__doc__)
class PreTrainingArguments(TrainingArguments):
    min_learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Minimum learning rate deacyed to."},
    )
    decay_steps: float = field(
        default=None,
        metadata={
            "help": "The steps use to control the learing rate. If the step > decay_steps, will use the min_learning_rate."
        },
    )
    enable_linear_fused_grad_add: bool = field(
        default=False,
        metadata={
            "help": "Enable fused linear grad add strategy, which will reduce elementwise add for grad accumulation in the backward of nn.Linear ."
        },
    )
    # NOTE(gongenlei): new add autotuner_benchmark
    autotuner_benchmark: bool = field(
        default=False,
        metadata={"help": "Weather to run benchmark by autotuner. True for from_scratch and pad_max_length."},
    )
    unified_checkpoint: bool = field(
        default=True,
        metadata={"help": "Enable fused linear grad add strategy."},
    )

    def __post_init__(self):
        super().__post_init__()
        # NOTE(gongenlei): new add autotuner_benchmark
        from paddleformers.trainer.trainer_utils import IntervalStrategy

        if self.autotuner_benchmark:
            self.max_steps = 5
            self.do_train = True
            self.do_export = False
            self.do_predict = False
            self.do_eval = False
            self.overwrite_output_dir = True
            self.load_best_model_at_end = False
            self.report_to = []
            self.save_strategy = IntervalStrategy.NO
            self.evaluation_strategy = IntervalStrategy.NO
            self.unified_checkpoint = False


def parse_args():
    parser = PdArgumentParser((PreTrainingArguments,))
    # Support format as "args.json --arg1 value1 --arg2 value2.”
    # In case of conflict, command line arguments take precedence.
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        model_args = parser.parse_json_file_and_cmd_lines()
    else:
        model_args = parser.parse_args_into_dataclasses()
    return model_args


def create_json_from_dict(data_dict, file_path):
    with open(file_path, "w") as f:
        json.dump(data_dict, f)


class ArgparserTest(unittest.TestCase):
    script_name = "test_argparser.py"
    args_dict = {
        "max_steps": 3000,
        "amp_master_grad": False,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "amp_custom_black_list": ["reduce_sum", "sin", "cos"],
        "adam_epsilon": 1e-08,
        "bf16": False,
        "enable_linear_fused_grad_add": False,
        "eval_steps": 3216,
        "flatten_param_grads": False,
        "fp16": 1,
        "log_on_each_node": True,
        "logging_dir": "./checkpoints/llama2_pretrain_ckpts/runs/Dec27_04-28-35_instance-047hzlt0-4",
        "logging_first_step": False,
        "logging_steps": 1,
        "lr_end": 1e-07,
        "max_evaluate_steps": -1,
        "max_grad_norm": 1.0,
        "min_learning_rate": 3e-06,
        "no_cuda": False,
        "num_cycles": 0.5,
        "num_train_epochs": 3.0,
        "output_dir": "./checkpoints/llama2_pretrain_ckpts",
    }

    def test_parse_cmd_lines(self):
        cmd_line_args = [ArgparserTest.script_name]
        for key, value in ArgparserTest.args_dict.items():
            if isinstance(value, list):
                cmd_line_args.extend([f"--{key}", *[str(v) for v in value]])
            else:
                cmd_line_args.extend([f"--{key}", str(value)])
        with patch("sys.argv", cmd_line_args):
            model_args = vars(parse_args()[0])
        for key, value in ArgparserTest.args_dict.items():
            self.assertEqual(model_args.get(key), value)

    def test_parse_json_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmpfile:
            create_json_from_dict(ArgparserTest.args_dict, tmpfile.name)
            tmpfile_path = tmpfile.name
        with patch("sys.argv", [ArgparserTest.script_name, tmpfile_path]):
            model_args = vars(parse_args()[0])
        for key, value in ArgparserTest.args_dict.items():
            self.assertEqual(model_args.get(key), value)
        os.remove(tmpfile_path)

    def test_parse_json_file_and_cmd_lines(self):
        half_size = len(ArgparserTest.args_dict) // 2
        json_part = {k: ArgparserTest.args_dict[k] for k in list(ArgparserTest.args_dict)[:half_size]}
        cmd_line_part = {k: ArgparserTest.args_dict[k] for k in list(ArgparserTest.args_dict)[half_size:]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmpfile:
            create_json_from_dict(json_part, tmpfile.name)
            tmpfile_path = tmpfile.name
        cmd_line_args = [ArgparserTest.script_name, tmpfile_path]
        for key, value in cmd_line_part.items():
            if isinstance(value, list):
                cmd_line_args.extend([f"--{key}", *[str(v) for v in value]])
            else:
                cmd_line_args.extend([f"--{key}", str(value)])
        with patch("sys.argv", cmd_line_args):
            model_args = vars(parse_args()[0])
        for key, value in ArgparserTest.args_dict.items():
            self.assertEqual(model_args.get(key), value)
        os.remove(tmpfile_path)

    def test_parse_json_file_and_cmd_lines_with_conflict(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmpfile:
            json.dump(ArgparserTest.args_dict, tmpfile)
            tmpfile_path = tmpfile.name
        cmd_line_args = [
            ArgparserTest.script_name,
            tmpfile_path,
            "--min_learning_rate",
            "2e-5",
            "--max_steps",
            "3000",
            "--log_on_each_node",
            "False",
        ]
        with patch("sys.argv", cmd_line_args):
            model_args = vars(parse_args()[0])
        self.assertEqual(model_args.get("min_learning_rate"), 2e-5)
        self.assertEqual(model_args.get("max_steps"), 3000)
        self.assertEqual(model_args.get("log_on_each_node"), False)
        for key, value in ArgparserTest.args_dict.items():
            if key not in ["min_learning_rate", "max_steps", "log_on_each_node"]:
                self.assertEqual(model_args.get(key), value)
        os.remove(tmpfile_path)
