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

"""PEFT utils"""
from paddleformers.peft import LoRAConfig, LoRAModel
from paddleformers.utils.log import logger


def initialize_lora_model(model, training_args, model_args, resume_from_checkpoint, dtype):
    """Initialize LoRAModel"""

    logger.info("Start to wrap model with LoRA config ...")
    if model_args.lora_path is None or resume_from_checkpoint:
        # If resume from checkpoint, LoRA adatper will be overwritten in the checkpoint loading process.
        if training_args.use_huggingface_model:
            target_modules = [
                ".*q_proj.*",
                ".*k_proj.*",
                ".*v_proj.*",
                ".*o_proj.*",
                ".*up_proj.*",
                ".*gate_proj.*",
                ".*down_proj.*",
                ".*spatial_linear.0.*",
                ".*spatial_linear.2.*",
                ".*temporal_linear.0.*",
                ".*temporal_linear.2.*",
            ]
        else:
            target_modules = [
                ".*qkv_proj.*",
                ".*o_proj.*",
                ".*up_gate_proj.*",
                ".*down_proj.*",
                ".*spatial_linear.0.*",
                ".*spatial_linear.2.*",
                ".*temporal_linear.0.*",
                ".*temporal_linear.2.*",
            ]
        if model_args.rslora_plus:
            model_args.rslora = True
            model_args.lora_plus_scale = 4
            model_args.lora_alpha = 4

        if training_args.weight_quantize_algo is not None:
            if model_args.rslora or model_args.lora_plus_scale != 1.0:
                logger.info("Weight quantization is not supported in LoRA+ and RsLoRA.")
        if model_args.lora_alpha == -1:
            if model_args.rslora:
                model_args.lora_alpha = 4
            else:
                model_args.lora_alpha = 2 * model_args.lora_rank
        lora_config = LoRAConfig(
            target_modules=target_modules,
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            rslora=model_args.rslora,
            lora_plus_scale=model_args.lora_plus_scale,
            tensor_parallel_degree=training_args.tensor_parallel_degree,
            dtype=dtype,
            head_dim=model.config.text_config.hidden_size // model.config.text_config.num_attention_heads,
            base_model_name_or_path=model_args.model_name_or_path,
        )
        model = LoRAModel(model, lora_config)
    else:
        model = LoRAModel.from_pretrained(
            model=model,
            lora_path=model_args.lora_path,
        )

    model.mark_only_lora_as_trainable()
    model.print_trainable_parameters()
    logger.info("Wraping model with LoRA config successfully !")
    return model
