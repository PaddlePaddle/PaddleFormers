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
multimodal_interleave_callback
"""
import logging
import os

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddleformers.trainer.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from paddleformers.trainer.training_args import TrainingArguments

DATATYPE_2_ID = {"mm": 0, "lm": 1, "audio": 2}


logger = logging.getLogger(__name__)


class MultiModalInterleaveCallback(TrainerCallback):
    """DataStatus

    Args:
        TrainerCallback (_type_): _description_
    """

    def __init__(self):
        """_summary_

        Args:
            modality_interleave (int): _description_
            modality_ratio (tuple): _description_
        """
        self.training_phase = None
        self.lr_state = None
        self.current_step = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Initializes or updates the data loading-related parameters
        in the TrainerState at the start of training.

        Args:
            args (TrainingArguments): Training arguments.
            state (TrainerState): The current trainer state.
            control (TrainerControl): Trainer control object.
            **kwargs: Additional optional arguments.

        Returns:
            None: No return value. The parameters in `state` are modified directly.

        """
        model = kwargs.pop("model")

        if int(os.environ.get("INJECT_VIS_EXPERT", 0)):
            optimizer = kwargs.pop("optimizer")
            logger.info("do inject vision experts")
            inject_vision_experts(model, optimizer)

        if state.trial_params is None:
            state.trial_params = {}

        if "last_lm_scheduler" in state.trial_params:
            self.lr_state = state.trial_params["last_lm_scheduler"]
            self.training_phase = state.trial_params["training_phase"]
            logger.info(
                f"load lm_scheduler stat:{self.lr_state}, training_phase: {self.training_phase}"
            )

            if hasattr(model, "update_params_stat"):
                if self.training_phase == "lm":
                    # logger.info(f"{state.global_step} -- do freeze_mm")
                    model.update_params_stat("lm", stop_gradient=False)
                    model.update_params_stat("mm", stop_gradient=True)
                    model.update_params_stat("audio", stop_gradient=True)
                elif self.training_phase == "mm":
                    model.update_params_stat("lm", stop_gradient=True)
                    model.update_params_stat("mm", stop_gradient=False)
                    model.update_params_stat("audio", stop_gradient=True)
                elif self.training_phase == "audio":
                    model.update_params_stat("lm", stop_gradient=True)
                    model.update_params_stat("mm", stop_gradient=True)
                    model.update_params_stat("audio", stop_gradient=False)

    def on_load_data_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        inputs,
        **kwargs,
    ):
        """_summary_

        Args:
            args (TrainingArguments): _description_
            state (TrainerState): _description_
            control (TrainerControl): _description_
            inputs (_type_): _description_
        """
        model = kwargs.pop("model")
        lr = kwargs.pop("lr_scheduler")
        assert (not hasattr(model, "update_params_stat")) or (
            not model.config.modality_detach
        ), "modality_detach should be False!"
        if args.modality_interleave == "gbs":
            if state.global_step != self.current_step:
                self.current_step = state.global_step
            else:
                return
        data_type = inputs.get("data_type")
        if data_type is None:
            assert hasattr(fleet.fleet, "_hcg")
            pp_group = fleet.get_hybrid_communicate_group().get_pipe_parallel_group()
            pp_size = dist.get_world_size(pp_group)
            if pp_size > 1:
                data_type_tensor = paddle.full([1], -1, dtype="int64")
                data_type_all = paddle.empty([pp_size], dtype="int64")
                dist.stream.all_gather(
                    data_type_all,
                    data_type_tensor,
                    group=pp_group,
                    use_calc_stream=True,
                )
                assert (
                    len(set([d for d in data_type_all.tolist() if d >= 0])) == 1
                ), f"data type not sync between pp group:, {data_type_all}"
                data_type = data_type_all[0]  # pp0 must have data type
        else:
            if hasattr(fleet.fleet, "_hcg"):
                data_type_tensor = paddle.to_tensor(data_type, dtype="int64")
                pp_group = (
                    fleet.get_hybrid_communicate_group().get_pipe_parallel_group()
                )
                pp_size = dist.get_world_size(pp_group)
                if pp_size > 1:
                    data_type_all = paddle.empty(
                        [pp_size, args.per_device_train_batch_size], dtype="int64"
                    )
                    dist.stream.all_gather(
                        data_type_all,
                        data_type_tensor,
                        group=pp_group,
                        use_calc_stream=True,
                    )
                    assert (
                        len(
                            set(
                                [
                                    d
                                    for d in data_type_all.reshape([-1]).tolist()
                                    if d >= 0
                                ]
                            )
                        )
                        == 1
                    ), f"data type not sync between pp group:, {data_type_all}"

        assert data_type is not None

        if (data_type == DATATYPE_2_ID["lm"]).all():
            if self.training_phase != "lm":
                # Enter LM state for the first time, rollback the lr-scheduler
                if self.lr_state is not None:
                    lr.set_state_dict(self.lr_state)
                # logger.info(f'lrset: {lr.state_dict()}')
                if hasattr(model, "update_params_stat"):
                    # logger.info(f"{state.global_step} -- do freeze_mm")
                    model.update_params_stat("lm", stop_gradient=False)
                    model.update_params_stat("mm", stop_gradient=True)
                    model.update_params_stat("audio", stop_gradient=True)
                    self.training_phase = "lm"
        elif (data_type == DATATYPE_2_ID["mm"]).all():
            if self.training_phase != "mm":
                self.lr_state = lr.state_dict()
                if hasattr(model, "update_params_stat"):
                    # logger.info(f"{state.global_step} -- do freeze_lm")
                    model.update_params_stat("lm", stop_gradient=True)
                    model.update_params_stat("mm", stop_gradient=False)
                    model.update_params_stat("audio", stop_gradient=True)
                    self.training_phase = "mm"
        elif (data_type == DATATYPE_2_ID["audio"]).all():
            if self.training_phase != "audio":
                self.lr_state = lr.state_dict()
                if hasattr(model, "update_params_stat"):
                    # logger.info(f"{state.global_step} -- do freeze_lm")
                    model.update_params_stat("lm", stop_gradient=True)
                    model.update_params_stat("mm", stop_gradient=True)
                    model.update_params_stat("audio", stop_gradient=False)
                    self.training_phase = "audio"
        else:
            assert False, f"data_type error!:{data_type}"

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """doc"""
        if self.lr_state is not None:
            state.trial_params["last_lm_scheduler"] = self.lr_state
            state.trial_params["training_phase"] = self.training_phase

            if self.lr_state["last_epoch"] == args.max_steps:
                control.should_training_stop = True
            else:
                control.should_training_stop = False


def inject_vision_experts(model, optimizer):
    """_summary_

    Args:
        model (_type_): _description_
    """

    def _gen_vis_exp_for_single_layer(model_param, layer_idx, targe_param):
        for exp_i in range(0, 6):
            for key1 in ["up_gate_proj", "down_proj"]:
                for key2 in ["weight", "bias"]:
                    key_name = (
                        f"ernie.layers.{layer_idx}.mlp.experts.{exp_i}.{key1}.{key2}"
                    )
                    target_name = (
                        f"ernie.layers.{layer_idx}.mlp.experts.{exp_i+6}.{key1}.{key2}"
                    )
                    shape = model_param[key_name].shape
                    model_param[key_name] = paddle.to_tensor(model_param[key_name])
                    assert model_param[key_name].dtype == paddle.bfloat16
                    if key2 == "weight" and key1 == "down_proj":
                        model_param[key_name] = model_param[key_name] / 6
                        targe_param[target_name] = model_param[key_name][
                            : shape[0] // 3, :
                        ]
                    elif key2 == "weight" and key1 == "up_gate_proj":
                        # targe_param[target_name] = model_param[key_name][:shape[0]//3, :]
                        up_weight, gate_weight = paddle.split(
                            model_param[key_name], num_or_sections=2, axis=1
                        )
                        num_cols_to_take = shape[1] // 2 // 3  # 9216 / 2 / 3
                        up_weight_part = up_weight[:, :num_cols_to_take]
                        gate_weight_part = gate_weight[:, :num_cols_to_take]
                        targe_param[target_name] = paddle.concat(
                            [up_weight_part, gate_weight_part], axis=1
                        )
                    elif key2 == "bias" and key1 == "down_proj":
                        targe_param[target_name] = model_param[key_name]
                    elif key2 == "bias" and key1 == "up_gate_proj":
                        assert len(model_param[key_name].shape) == 1
                        up_weight, gate_weight = paddle.split(
                            model_param[key_name], num_or_sections=2
                        )
                        targe_param[target_name] = paddle.concat(
                            [up_weight[: shape[0] // 6], gate_weight[: shape[0] // 6]],
                            axis=0,
                        )
        return targe_param

    layer_idx = [
        int(key.split(".")[2])
        for key in model.state_dict().keys()
        if "mlp.experts" in key
    ]
    layer_idx = sorted(list(set(layer_idx)))
    v_exps = {}
    for layer in layer_idx:
        v_exps = _gen_vis_exp_for_single_layer(model.state_dict(), layer, v_exps)
    logger.info("generate visual experts successfully")

    model_param = model.state_dict()
    model_param.update(v_exps)
    logger.info(model.set_state_dict(model_param))

    pipeline_name_mapping = {
        v: k for k, v in getattr(model, "_pipeline_name_mapping", {}).items()
    }
    pname_sname = {}
    for n, p in model.named_parameters():
        pname_sname[p.name] = pipeline_name_mapping.get(n, n)
    # logger.info(f"mapping - {pname_sname}")
    # logger.info(f"vision-experts-keys --- {v_exps.keys()}")
    assert (
        len(optimizer._master_weights.keys()) > 0
    ), "no master weights found in optimizer"

    def _varbase_help(param, tmp_tensor):
        tmp_tensor._share_buffer_to(param)
        tmp_tensor._clear()

    for pname in optimizer._master_weights:
        if pname_sname[pname] in v_exps:
            assert (
                optimizer._master_weights[pname].shape
                == v_exps[pname_sname[pname]].shape
            )
            # logger.info(f"params-before-- {optimizer._master_weights[pname]}")
            _varbase_help(
                optimizer._master_weights[pname],
                v_exps[pname_sname[pname]].cast("float32"),
            )
            # logger.info(f"params-after-- {optimizer._master_weights[pname]}")
            logger.info(f"update {pname_sname[pname]} master weights ")
