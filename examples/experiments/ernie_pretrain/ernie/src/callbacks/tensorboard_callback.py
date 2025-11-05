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

import importlib.util
import json

from paddleformers.peft.lora import LoRAModel
from paddleformers.trainer.trainer_callback import TrainerCallback
from paddleformers.transformers import PretrainedModel
from paddleformers.utils.log import logger

try:
    from paddleformers.trainer.trainer import clear_async_save_task_queue
except Exception:
    clear_async_save_task_queue = None


def is_tensorboard_available():
    return importlib.util.find_spec("tensorboard") is not None or importlib.util.find_spec("tensorboardX") is not None


def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d


class TensorBoardCallback(TrainerCallback):
    def __init__(
        self,
        args,
        model,
        tb_writer=None,
        log_flops_per_step=False,
        log_tokens_per_step=False,
    ):
        has_tensorboard = is_tensorboard_available()
        if not has_tensorboard:
            raise RuntimeError(
                "TensorBoardCallback requires tensorboard to be installed. Either update or install tensorboardX."
            )
        if has_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._SummaryWriter = SummaryWriter
            except ImportError:
                try:
                    from tensorboardX import SummaryWriter

                    self._SummaryWriter = SummaryWriter
                except ImportError:
                    self._SummaryWriter = None
        else:
            self._SummaryWriter = None
        self.tb_writer = tb_writer

        def get_numel_item(p):
            item = p.numel().item()
            return item if item else 0

        self.model_numel = sum(
            get_numel_item(p)
            for n, p in model.named_parameters()
            if not p.stop_gradient and "embeddings" not in n and "embed_tokens" not in n
        )
        self.log_flops_per_step = log_flops_per_step
        self.log_tokens_per_step = log_tokens_per_step

    def _init_summary_writer(self, args, log_dir=None):
        log_dir = log_dir or args.logging_dir
        if self._SummaryWriter is not None:
            self.tb_writer = self._SummaryWriter(log_dir=log_dir)

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        log_dir = None

        if self.tb_writer is None:
            self._init_summary_writer(args, log_dir)

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", args.to_json_string())
            if "model" in kwargs:
                model = kwargs["model"]

                if (isinstance(model, PretrainedModel) and model.constructed_from_pretrained_config()) or isinstance(
                    model, LoRAModel
                ):
                    model.config.architectures = [model.__class__.__name__]
                    self.tb_writer.add_text("model_config", str(model.config))

                elif hasattr(model, "init_config") and model.init_config is not None:
                    model_config_json = json.dumps(model.get_model_config(), ensure_ascii=False, indent=2)
                    self.tb_writer.add_text("model_config", model_config_json)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        timers = kwargs.get("timers")
        paddle_pipeline_timers = kwargs.get("paddle_pipeline_timers")

        if self.tb_writer is None:
            self._init_summary_writer(args)

        if self.tb_writer is not None:
            logs = rewrite_logs(logs)

            total_tokens_per_step = (
                args.train_batch_size
                * args.gradient_accumulation_steps
                * args.reeao_dataset_world_size
                * args.max_seq_length
            )

            if self.log_flops_per_step:
                logger.warning("The FLOPs might be not accurate")
                flops_per_step = self.model_numel * total_tokens_per_step * 6
            else:
                flops_per_step = None

            if self.log_tokens_per_step:
                tokens_per_step = total_tokens_per_step
            else:
                tokens_per_step = None
            inputs = kwargs.get("inputs")
            data_type = inputs and inputs.get("data_type")
            if data_type is not None:
                data_type = data_type.tolist()[-1]
                logs.update(data_type=data_type)

            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, state.global_step)

                    if tokens_per_step is not None and k in ["train/loss"]:
                        self.tb_writer.add_scalar(k + "_xaxis_tokens", v, state.global_step * tokens_per_step)

                    if flops_per_step is not None and k in ["train/loss"]:
                        self.tb_writer.add_scalar(k + "_xaxis_flops", v, state.global_step * flops_per_step)

                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            if timers is not None:
                timers.write(timers.timers.keys(), self.tb_writer, state.global_step, reset=False)

            if paddle_pipeline_timers:
                for name, timer in paddle_pipeline_timers.timers.items():
                    elapsed_time = timer.elapsed(reset=False)
                    self.tb_writer.add_scalar(f"timers/{name}", elapsed_time, state.global_step)

            self.tb_writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if clear_async_save_task_queue:
            clear_async_save_task_queue()
        if self.tb_writer:
            self.tb_writer.close()
            self.tb_writer = None
