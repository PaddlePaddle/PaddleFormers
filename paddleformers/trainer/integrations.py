# Copyright 2020 The HuggingFace Team. All rights reserved.
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

# This file is modified from
#  https://github.com/huggingface/transformers/blob/main/src/transformers/integrations.py

import importlib
import json
import numbers
import os
import tempfile
from pathlib import Path

from ..peft import LoRAModel, PrefixModelForCausalLM, VeRAModel
from ..transformers import PretrainedModel
from ..utils.log import logger
from .trainer_callback import TrainerCallback


def is_visualdl_available():
    return importlib.util.find_spec("visualdl") is not None


def is_tensorboardX_available():
    return importlib.util.find_spec("tensorboardX") is not None


def is_wandb_available():
    if os.getenv("WANDB_DISABLED", "").upper() in {"1", "ON", "YES", "TRUE"}:
        return False
    return importlib.util.find_spec("wandb") is not None


def is_swanlab_available():
    return importlib.util.find_spec("swanlab") is not None


def is_ray_available():
    return importlib.util.find_spec("ray.air") is not None


def get_available_reporting_integrations():
    integrations = []
    if is_visualdl_available():
        integrations.append("visualdl")
    if is_wandb_available():
        integrations.append("wandb")
    if is_tensorboardX_available():
        integrations.append("tensorboard")
    if is_swanlab_available():
        integrations.append("swanlab")

    return integrations


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


class VisualDLCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [VisualDL](https://www.paddlepaddle.org.cn/paddle/visualdl).
    Args:
        vdl_writer (`LogWriter`, *optional*):
            The writer to use. Will instantiate one if not set.
    """

    def __init__(self, vdl_writer=None):
        has_visualdl = is_visualdl_available()
        if not has_visualdl:
            raise RuntimeError("VisualDLCallback requires visualdl to be installed. Please install visualdl.")
        if has_visualdl:
            try:
                from visualdl import LogWriter

                self._LogWriter = LogWriter
            except ImportError:
                self._LogWriter = None
        else:
            self._LogWriter = None
        self.vdl_writer = vdl_writer

    def _init_summary_writer(self, args, log_dir=None):
        log_dir = log_dir or args.logging_dir
        if self._LogWriter is not None:
            self.vdl_writer = self._LogWriter(logdir=log_dir)

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        log_dir = None

        if self.vdl_writer is None:
            self._init_summary_writer(args, log_dir)

        if self.vdl_writer is not None:
            self.vdl_writer.add_text("args", args.to_json_string())
            if "model" in kwargs and logger.logger.level < 20:
                model = kwargs["model"]
                if (
                    isinstance(model, LoRAModel)
                    or isinstance(model, PrefixModelForCausalLM)
                    or isinstance(model, VeRAModel)
                ):
                    model = kwargs["model"].model
                if isinstance(model, PretrainedModel) and model.constructed_from_pretrained_config():
                    model.config.architectures = [model.__class__.__name__]
                    self.vdl_writer.add_text("model_config", str(model.config))
                elif hasattr(model, "init_config") and model.init_config is not None:
                    model_config_json = json.dumps(model.get_model_config(), ensure_ascii=False, indent=2)
                    self.vdl_writer.add_text("model_config", model_config_json)

            if hasattr(self.vdl_writer, "add_hparams"):
                self.vdl_writer.add_hparams(args.to_sanitized_dict(), metrics_list=[])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.vdl_writer is None:
            return

        if self.vdl_writer is not None:
            logs = rewrite_logs(logs)
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.vdl_writer.add_scalar(k, v, state.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of VisualDL's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            self.vdl_writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.vdl_writer:
            self.vdl_writer.close()
            self.vdl_writer = None


class TensorBoardCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [TensorBoard](https://www.tensorflow.org/tensorboard).

    Args:
        tb_writer (`SummaryWriter`, *optional*):
            The writer to use. Will instantiate one if not set.
    """

    def __init__(self, tb_writer=None):
        has_tensorboard = is_tensorboardX_available()
        if not has_tensorboard:
            raise RuntimeError("TensorBoardCallback requires tensorboardX to be installed")

        if has_tensorboard:
            try:
                from tensorboardX import SummaryWriter

                self._SummaryWriter = SummaryWriter
            except ImportError:
                self._SummaryWriter = None
        else:
            self._SummaryWriter = None
        self.tb_writer = tb_writer

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
                if hasattr(model, "config") and model.config is not None:
                    model_config_json = model.config.to_json_string()
                    self.tb_writer.add_text("model_config", model_config_json)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writer is None:
            self._init_summary_writer(args)

        if self.tb_writer is not None:
            logs = rewrite_logs(logs)
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, state.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            self.tb_writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.tb_writer:
            self.tb_writer.close()
            self.tb_writer = None


class WandbCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that logs metrics, media, model checkpoints to [Weight and Biases](https://www.wandb.com/).
    """

    def __init__(self):
        has_wandb = is_wandb_available()
        if not has_wandb:
            raise RuntimeError("WandbCallback requires wandb to be installed. Run `pip install wandb`.")
        if has_wandb:
            import wandb

            self._wandb = wandb
        self._initialized = False
        # log model
        self._log_model = os.getenv("WANDB_LOG_MODEL", "false").lower()

    def setup(self, args, state, model, **kwargs):
        """
        Setup the optional Weights & Biases (*wandb*) integration.

        One can subclass and override this method to customize the setup if needed.
        variables:
        Environment:
        - **WANDB_LOG_MODEL** (`str`, *optional*, defaults to `"false"`):
            Whether to log model and checkpoints during training. Can be `"end"`, `"checkpoint"` or `"false"`. If set
            to `"end"`, the model will be uploaded at the end of training. If set to `"checkpoint"`, the checkpoint
            will be uploaded every `args.save_steps` . If set to `"false"`, the model will not be uploaded. Use along
            with [`TrainingArguments.load_best_model_at_end`] to upload best model.
        - **WANDB_WATCH** (`str`, *optional* defaults to `"false"`):
            Can be `"gradients"`, `"all"`, `"parameters"`, or `"false"`. Set to `"all"` to log gradients and
            parameters.
        - **WANDB_PROJECT** (`str`, *optional*, defaults to `"PaddleNLP"`):
            Set this to a custom string to store results in a different project.
        - **WANDB_DISABLED** (`bool`, *optional*, defaults to `False`):
            Whether to disable wandb entirely. Set `WANDB_DISABLED=true` to disable.
        """
        if self._wandb is None:
            return

        if args.wandb_http_proxy:
            os.environ["WANDB_HTTPS_PROXY"] = args.wandb_http_proxy

        # Check if a Weights & Biases (wandb) API key is provided in the training arguments
        if args.wandb_api_key:
            if self._wandb.api.api_key:
                logger.warning(
                    "A Weights & Biases API key is already configured in the environment. "
                    "However, the training argument 'wandb_api_key' will take precedence. "
                )
            self._wandb.login(key=args.wandb_api_key)

        self._initialized = True

        if state.is_world_process_zero:
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            combined_dict = {**args.to_dict()}

            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            trial_name = state.trial_name
            init_args = {}
            if trial_name is not None:
                init_args["name"] = trial_name
                init_args["group"] = args.run_name
            else:
                if not (args.run_name is None or args.run_name == args.output_dir):
                    init_args["name"] = args.run_name
            init_args["dir"] = args.logging_dir
            if self._wandb.run is None:
                self._wandb.init(
                    project=os.getenv("WANDB_PROJECT", "PaddleNLP"),
                    **init_args,
                )
            # add config parameters (run may have been created manually)
            self._wandb.config.update(combined_dict, allow_val_change=True)

            # define default x-axis (for latest wandb versions)
            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("train/global_step")
                self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

            # keep track of model topology and gradients
            _watch_model = os.getenv("WANDB_WATCH", "false")
            if _watch_model in ("all", "parameters", "gradients"):
                self._wandb.watch(model, log=_watch_model, log_freq=max(100, state.logging_steps))
            self._wandb.run._label(code="transformers_trainer")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model, **kwargs)

    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self._wandb is None:
            return
        if self._log_model in ("end", "checkpoint") and self._initialized and state.is_world_process_zero:
            from ..trainer import Trainer

            fake_trainer = Trainer(args=args, model=model, tokenizer=tokenizer)
            with tempfile.TemporaryDirectory() as temp_dir:
                fake_trainer.save_model(temp_dir)
                metadata = (
                    {
                        k: v
                        for k, v in dict(self._wandb.summary).items()
                        if isinstance(v, numbers.Number) and not k.startswith("_")
                    }
                    if not args.load_best_model_at_end
                    else {
                        f"eval/{args.metric_for_best_model}": state.best_metric,
                        "train/total_floss": state.total_flos,
                    }
                )
                logger.info("Logging model artifacts. ...")

                model_name = (
                    f"model-{self._wandb.run.id}"
                    if (args.run_name is None or args.run_name == args.output_dir)
                    else f"model-{self._wandb.run.name}"
                )
                artifact = self._wandb.Artifact(name=model_name, type="model", metadata=metadata)
                for f in Path(temp_dir).glob("*"):
                    if f.is_file():
                        with artifact.new_file(f.name, mode="wb") as fa:
                            fa.write(f.read_bytes())

                self._wandb.run.log_artifact(artifact)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            logs = rewrite_logs(logs)
            self._wandb.log({**logs, "train/global_step": state.global_step})

    def on_save(self, args, state, control, **kwargs):
        if self._log_model == "checkpoint" and self._initialized and state.is_world_process_zero:
            checkpoint_metadata = {
                k: v
                for k, v in dict(self._wandb.summary).items()
                if isinstance(v, numbers.Number) and not k.startswith("_")
            }
            ckpt_dir = f"checkpoint-{state.global_step}"
            artifact_path = os.path.join(args.output_dir, ckpt_dir)
            logger.info(f"Logging checkpoint artifacts in {ckpt_dir}. ...")
            checkpoint_name = (
                f"checkpoint-{self._wandb.run.id}"
                if (args.run_name is None or args.run_name == args.output_dir)
                else f"checkpoint-{self._wandb.run.name}"
            )
            artifact = self._wandb.Artifact(name=checkpoint_name, type="model", metadata=checkpoint_metadata)
            artifact.add_dir(artifact_path)
            self._wandb.log_artifact(artifact, aliases=[f"checkpoint-{state.global_step}"])


class SwanLabCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that logs metrics, media to [Swanlab](https://swanlab.cn/).
    """

    def __init__(self):
        has_swanlab = is_swanlab_available()
        if not has_swanlab:
            raise RuntimeError("SwanlabCallback requires swanlab to be installed. Run `pip install swanlab`.")
        if has_swanlab:
            import swanlab

            self._swanlab = swanlab

        self._initialized = False

    def setup(self, args, state, model, **kwargs):
        """
        Setup the optional Swanlab integration.

        One can subclass and override this method to customize the setup if needed.
        variables:
        Environment:
        - **SWANLAB_MODE** (`str`, *optional*, defaults to `"cloud"`):
            Whether to use swanlab cloud, local or disabled. Set `SWANLAB_MODE="local"` to use local. Set `SWANLAB_MODE="disabled"` to disable.
        - **SWANLAB_PROJECT** (`str`, *optional*, defaults to `"PaddleNLP"`):
            Set this to a custom string to store results in a different project.
        """

        if self._swanlab is None:
            return

        self._initialized = True

        if state.is_world_process_zero:
            logger.info('Automatic Swanlab logging enabled, to disable set os.environ["SWANLAB_MODE"] = "disabled"')

            combined_dict = {**args.to_dict()}

            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}

            trial_name = state.trial_name
            init_args = {}
            if trial_name is not None:
                init_args["name"] = trial_name
                init_args["group"] = args.run_name
            else:
                if not (args.run_name is None or args.run_name == args.output_dir):
                    init_args["name"] = args.run_name
            init_args["dir"] = args.logging_dir
            if self._swanlab.get_run() is None:
                self._swanlab.init(
                    project=os.getenv("SWANLAB_PROJECT", "PaddleNLP"),
                    **init_args,
                )
            self._swanlab.config.update(combined_dict, allow_val_change=True)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self._swanlab is None:
            return
        if not self._initialized:
            self.setup(args, state, model, **kwargs)

    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self._swanlab is None:
            return

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._swanlab is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            logs = rewrite_logs(logs)
            self._swanlab.log({**logs, "train/global_step": state.global_step}, step=state.global_step)


class AutoNLPCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [`Ray Tune`] for [`AutoNLP`]
    """

    def __init__(self):
        if not is_ray_available():
            raise RuntimeError(
                "AutoNLPCallback requires extra dependencies to be installed. Please install paddlenlp with 'pip install paddlenlp[autonlp]'."
            )
        self.session = importlib.import_module("ray.air.session")
        self.tune = importlib.import_module("ray.tune")

    # report session metrics to Ray to track trial progress
    def on_evaluate(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        metrics = kwargs.get("metrics", None)
        if self.tune.is_session_enabled() and metrics is not None and isinstance(metrics, dict):
            self.session.report(metrics)


INTEGRATION_TO_CALLBACK = {
    "visualdl": VisualDLCallback,
    "autonlp": AutoNLPCallback,
    "wandb": WandbCallback,
    "tensorboard": TensorBoardCallback,
    "swanlab": SwanLabCallback,
}


def get_reporting_integration_callbacks(report_to):
    for integration in report_to:
        if integration not in INTEGRATION_TO_CALLBACK:
            raise ValueError(
                f"{integration} is not supported, only {', '.join(INTEGRATION_TO_CALLBACK.keys())} are supported."
            )
    return [INTEGRATION_TO_CALLBACK[integration] for integration in report_to]
