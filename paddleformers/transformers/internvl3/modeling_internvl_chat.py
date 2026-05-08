# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import warnings
from typing import List, Optional, Tuple, Union

import paddle
import paddle.nn.functional as F
from paddle import nn

from ..llama.modeling import LlamaForCausalLM
from ..model_outputs import CausalLMOutputWithPast
from ..model_utils import PretrainedModel
from ..qwen2.modeling import Qwen2ForCausalLMDeprecated
from ..utils import logger
from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel


class InternVLChatModel(PretrainedModel):
    config_class = InternVLChatConfig
    main_input_name = "pixel_values"
    base_model_prefix = "language_model"
    _no_split_modules = ["InternVisionEncoderLayer", "Qwen2DecoderLayer", "LlamaDecoderLayer"]
    _supports_flex_attn = True

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template or "internvl2_5"
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio**2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        config.vision_config.use_flash_attn = bool(
            use_flash_attn and getattr(config.vision_config, "use_flash_attn", False)
        )

        logger.info(f"num_image_token: {self.num_image_token}")
        logger.info(f"ps_version: {self.ps_version}")

        self.vision_model = vision_model if vision_model is not None else InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            architecture = config.llm_config.architectures[0]
            if architecture == "LlamaForCausalLM":
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif architecture == "Qwen2ForCausalLM":
                self.language_model = Qwen2ForCausalLMDeprecated(config.llm_config)
            else:
                raise NotImplementedError(f"{architecture} is not implemented.")

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size
        projector_dim = vit_hidden_size * int(1 / self.downsample_ratio) ** 2
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(projector_dim),
            nn.Linear(projector_dim, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

    @classmethod
    def _gen_aoa_config(cls, config: InternVLChatConfig):
        llm_source_prefix = "language_model.model."
        llm_target_prefix = "language_model.model."
        vision_source_prefix = "vision_model."
        vision_source_embeddings_prefix = "vision_model.embeddings."
        vision_target_prefix = "vision_model."
        projector_prefix = "mlp1."
        architecture = config.llm_config.architectures[0]

        aoa_statements = [
            f"{llm_source_prefix}embed_tokens.weight -> {llm_target_prefix}embed_tokens.weight",
            f"{llm_source_prefix}norm.weight -> {llm_target_prefix}norm.weight",
            f"{llm_source_prefix}layers.$LAYER_ID.input_layernorm.weight -> {llm_target_prefix}layers.$LAYER_ID.input_layernorm.weight",
            f"{llm_source_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> {llm_target_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
            f"{llm_source_prefix}layers.$LAYER_ID.self_attn.o_proj.weight^T -> {llm_target_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
            f"{llm_source_prefix}layers.$LAYER_ID.mlp.down_proj.weight^T -> {llm_target_prefix}layers.$LAYER_ID.mlp.down_proj.weight",
            f"{vision_source_embeddings_prefix}class_embedding -> {vision_target_prefix}embeddings.class_embedding",
            f"{vision_source_embeddings_prefix}patch_embedding.weight -> {vision_target_prefix}embeddings.patch_embedding.weight",
            f"{vision_source_embeddings_prefix}patch_embedding.bias -> {vision_target_prefix}embeddings.patch_embedding.bias",
            f"{vision_source_embeddings_prefix}position_embedding -> {vision_target_prefix}embeddings.position_embedding",
            f"{vision_source_prefix}encoder.layers.$LAYER_ID.attn.qkv.weight^T -> {vision_target_prefix}encoder.layers.$LAYER_ID.attn.qkv.weight",
            f"{vision_source_prefix}encoder.layers.$LAYER_ID.attn.qkv.bias -> {vision_target_prefix}encoder.layers.$LAYER_ID.attn.qkv.bias",
            f"{vision_source_prefix}encoder.layers.$LAYER_ID.attn.proj.weight^T -> {vision_target_prefix}encoder.layers.$LAYER_ID.attn.proj.weight",
            f"{vision_source_prefix}encoder.layers.$LAYER_ID.attn.proj.bias -> {vision_target_prefix}encoder.layers.$LAYER_ID.attn.proj.bias",
            f"{vision_source_prefix}encoder.layers.$LAYER_ID.norm1.weight -> {vision_target_prefix}encoder.layers.$LAYER_ID.norm1.weight",
            f"{vision_source_prefix}encoder.layers.$LAYER_ID.norm1.bias -> {vision_target_prefix}encoder.layers.$LAYER_ID.norm1.bias",
            f"{vision_source_prefix}encoder.layers.$LAYER_ID.norm2.weight -> {vision_target_prefix}encoder.layers.$LAYER_ID.norm2.weight",
            f"{vision_source_prefix}encoder.layers.$LAYER_ID.norm2.bias -> {vision_target_prefix}encoder.layers.$LAYER_ID.norm2.bias",
            f"{vision_source_prefix}encoder.layers.$LAYER_ID.ls1 -> {vision_target_prefix}encoder.layers.$LAYER_ID.ls1",
            f"{vision_source_prefix}encoder.layers.$LAYER_ID.ls2 -> {vision_target_prefix}encoder.layers.$LAYER_ID.ls2",
            f"{vision_source_prefix}encoder.layers.$LAYER_ID.mlp.fc1.weight^T -> {vision_target_prefix}encoder.layers.$LAYER_ID.mlp.fc1.weight",
            f"{vision_source_prefix}encoder.layers.$LAYER_ID.mlp.fc1.bias -> {vision_target_prefix}encoder.layers.$LAYER_ID.mlp.fc1.bias",
            f"{vision_source_prefix}encoder.layers.$LAYER_ID.mlp.fc2.weight^T -> {vision_target_prefix}encoder.layers.$LAYER_ID.mlp.fc2.weight",
            f"{vision_source_prefix}encoder.layers.$LAYER_ID.mlp.fc2.bias -> {vision_target_prefix}encoder.layers.$LAYER_ID.mlp.fc2.bias",
            f"{projector_prefix}0.weight -> {projector_prefix}0.weight",
            f"{projector_prefix}0.bias -> {projector_prefix}0.bias",
            f"{projector_prefix}1.weight^T -> {projector_prefix}1.weight",
            f"{projector_prefix}1.bias -> {projector_prefix}1.bias",
            f"{projector_prefix}3.weight^T -> {projector_prefix}3.weight",
            f"{projector_prefix}3.bias -> {projector_prefix}3.bias",
        ]

        if architecture == "Qwen2ForCausalLM":
            aoa_statements += [
                f"{llm_source_prefix}layers.$LAYER_ID.self_attn.q_proj.weight^T, {llm_source_prefix}layers.$LAYER_ID.self_attn.k_proj.weight^T, {llm_source_prefix}layers.$LAYER_ID.self_attn.v_proj.weight^T -> {llm_target_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.llm_config.num_attention_heads}, num_key_value_groups={config.llm_config.num_key_value_heads}",
                f"{llm_source_prefix}layers.$LAYER_ID.self_attn.q_proj.bias, {llm_source_prefix}layers.$LAYER_ID.self_attn.k_proj.bias, {llm_source_prefix}layers.$LAYER_ID.self_attn.v_proj.bias -> {llm_target_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.llm_config.num_attention_heads}, num_key_value_groups={config.llm_config.num_key_value_heads}, axis=0",
                f"{llm_source_prefix}layers.$LAYER_ID.mlp.gate_proj.weight^T, {llm_source_prefix}layers.$LAYER_ID.mlp.up_proj.weight^T -> {llm_target_prefix}layers.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn",
            ]
        elif architecture == "LlamaForCausalLM":
            aoa_statements += [
                f"{llm_source_prefix}layers.$LAYER_ID.self_attn.q_proj.weight^T -> {llm_target_prefix}layers.$LAYER_ID.self_attn.q_proj.weight",
                f"{llm_source_prefix}layers.$LAYER_ID.self_attn.k_proj.weight^T -> {llm_target_prefix}layers.$LAYER_ID.self_attn.k_proj.weight",
                f"{llm_source_prefix}layers.$LAYER_ID.self_attn.v_proj.weight^T -> {llm_target_prefix}layers.$LAYER_ID.self_attn.v_proj.weight",
                f"{llm_source_prefix}layers.$LAYER_ID.mlp.gate_proj.weight^T -> {llm_target_prefix}layers.$LAYER_ID.mlp.gate_proj.weight",
                f"{llm_source_prefix}layers.$LAYER_ID.mlp.up_proj.weight^T -> {llm_target_prefix}layers.$LAYER_ID.mlp.up_proj.weight",
            ]
        else:
            raise NotImplementedError(f"Unsupported architecture for aoa conversion: {architecture}")

        if config.tie_word_embeddings:
            aoa_statements.append(f"{llm_source_prefix}embed_tokens.weight -> language_model.lm_head.weight")
        else:
            aoa_statements.append("language_model.lm_head.weight -> language_model.lm_head.weight")

        return {"aoa_statements": aoa_statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config: InternVLChatConfig):
        aoa_config = cls._gen_aoa_config(config)
        aoa_config["aoa_config_reverse"] = True
        return aoa_config

    def pixel_shuffle(self, hidden_states, scale_factor=0.5):
        batch_size, width, height, channels = hidden_states.shape
        hidden_states = hidden_states.reshape(
            [batch_size, width, int(height * scale_factor), int(channels / scale_factor)]
        )
        hidden_states = hidden_states.transpose([0, 2, 1, 3])
        hidden_states = hidden_states.reshape(
            [
                batch_size,
                int(height * scale_factor),
                int(width * scale_factor),
                int(channels / (scale_factor * scale_factor)),
            ]
        )
        if self.ps_version == "v1":
            warnings.warn(
                "In ps_version 'v1', the height and width have not been swapped back, which results in a transposed image."
            )
        else:
            hidden_states = hidden_states.transpose([0, 2, 1, 3])
        return hidden_states

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        height = width = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape([vit_embeds.shape[0], height, width, -1])
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape([vit_embeds.shape[0], -1, vit_embeds.shape[-1]])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def _merge_input_embeds_with_vision_features(self, input_ids, input_embeds, vit_embeds):
        batch_size, seq_len, hidden_size = input_embeds.shape
        flat_input_embeds = input_embeds.reshape([batch_size * seq_len, hidden_size])
        flat_input_ids = input_ids.reshape([batch_size * seq_len])
        selected = flat_input_ids == self.img_context_token_id
        selected_indices = paddle.nonzero(selected).reshape([-1])
        flat_vit_embeds = vit_embeds.reshape([-1, hidden_size]).astype(flat_input_embeds.dtype)

        if selected_indices.shape[0] != flat_vit_embeds.shape[0]:
            raise ValueError(
                f"Image features and image tokens do not match, tokens: {selected_indices.shape[0]}, "
                f"features: {flat_vit_embeds.shape[0]}"
            )

        flat_input_embeds = paddle.scatter(flat_input_embeds, selected_indices, flat_vit_embeds, overwrite=True)
        return flat_input_embeds.reshape([batch_size, seq_len, hidden_size])

    def _build_position_ids(self, attention_mask: paddle.Tensor) -> paddle.Tensor:
        position_ids = paddle.cumsum(attention_mask.astype("int64"), axis=-1) - 1
        return paddle.where(attention_mask > 0, position_ids, paddle.zeros_like(position_ids))

    def _apply_repetition_penalty(
        self, logits: paddle.Tensor, token_ids: paddle.Tensor, penalty: float
    ) -> paddle.Tensor:
        if penalty == 1.0:
            return logits

        updated_logits = logits.clone()
        batch_size = token_ids.shape[0]
        for batch_idx in range(batch_size):
            seen_token_ids = paddle.unique(token_ids[batch_idx].astype("int64"))
            seen_logits = paddle.gather(updated_logits[batch_idx], seen_token_ids, axis=0)
            penalized_logits = paddle.where(seen_logits < 0, seen_logits * penalty, seen_logits / penalty)
            scatter_index = seen_token_ids.unsqueeze(-1)
            updated_logits[batch_idx] = paddle.scatter(
                updated_logits[batch_idx], scatter_index, penalized_logits, overwrite=True
            )
        return updated_logits

    def _infer_img_context_token_id(self, input_ids: paddle.Tensor) -> Optional[int]:
        pad_token_id = getattr(self.config.llm_config, "pad_token_id", None)
        best_token_id = None
        best_run = 0

        input_ids_list = input_ids.numpy().tolist()
        for sequence in input_ids_list:
            prev_token_id = None
            current_run = 0
            for token_id in sequence:
                if token_id == prev_token_id:
                    current_run += 1
                else:
                    prev_token_id = token_id
                    current_run = 1

                if token_id == pad_token_id:
                    continue

                if current_run > best_run:
                    best_run = current_run
                    best_token_id = token_id

        min_expected_run = max(8, self.num_image_token // 4)
        if best_token_id is not None and best_run >= min_expected_run:
            logger.warning(
                f"img_context_token_id is not initialized, inferred token id {best_token_id} from repeated runs."
            )
            return int(best_token_id)

        return None

    def forward(
        self,
        pixel_values: Optional[paddle.Tensor],
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        image_grid_thw: Optional[paddle.Tensor] = None,
        pixel_values_videos: Optional[paddle.Tensor] = None,
        video_grid_thw: Optional[paddle.Tensor] = None,
        input_features: Optional[paddle.Tensor] = None,
        feature_attention_mask: Optional[paddle.Tensor] = None,
        image_flags: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("input_ids must be provided for InternVLChatModel.")
        if self.img_context_token_id is None:
            inferred_token_id = self._infer_img_context_token_id(input_ids)
            if inferred_token_id is None:
                raise ValueError(
                    "img_context_token_id is not initialized and could not be inferred from input_ids. "
                    "Please call chat()/batch_chat() or set it manually."
                )
            self.img_context_token_id = inferred_token_id

        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        vit_embeds = self.extract_feature(pixel_values)
        if image_flags is not None:
            image_flags = image_flags.squeeze(-1)
            vit_embeds = vit_embeds[image_flags == 1]
        input_embeds = self._merge_input_embeds_with_vision_features(input_ids, input_embeds, vit_embeds)

        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            loss = F.cross_entropy(
                shift_logits.reshape([-1, shift_logits.shape[-1]]),
                shift_labels.reshape([-1]),
                reduction="mean",
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None if isinstance(outputs, tuple) else outputs.past_key_values,
            hidden_states=None if isinstance(outputs, tuple) else outputs.hidden_states,
            attentions=None if isinstance(outputs, tuple) else outputs.attentions,
        )

    @paddle.no_grad()
    def batch_chat(
        self,
        tokenizer,
        pixel_values,
        questions,
        generation_config,
        num_patches_list=None,
        history=None,
        return_history=False,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
        image_counts=None,
    ):
        if history is not None or return_history:
            raise NotImplementedError("Now multi-turn chat is not supported in batch_chat.")
        if image_counts is not None:
            num_patches_list = image_counts
            warnings.warn("`image_counts` is deprecated. Please use `num_patches_list` instead.", stacklevel=2)
        if num_patches_list is None:
            if pixel_values is None:
                num_patches_list = [0] * len(questions)
            elif len(questions) == 1:
                num_patches_list = [pixel_values.shape[0]]
            else:
                raise ValueError(
                    "batch_chat requires `num_patches_list` when `pixel_values` contains multiple samples."
                )
        self.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        queries = []
        template = None
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and "<image>" not in question:
                question = "<image>\n" + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            queries.append(query.replace("<image>", image_tokens, 1))

        tokenizer.padding_side = "left"
        model_inputs = tokenizer(queries, return_tensors="pd", padding=True)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        generation_config["eos_token_id"] = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        if isinstance(generation_output, paddle.Tensor):
            generation_output = generation_output.numpy().tolist()
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        return [response.split(template.sep.strip())[0].strip() for response in responses]

    @paddle.no_grad()
    def chat(
        self,
        tokenizer,
        pixel_values,
        question,
        generation_config,
        history=None,
        return_history=False,
        num_patches_list=None,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
    ):
        if history is None and pixel_values is not None and "<image>" not in question:
            question = "<image>\n" + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []

        self.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        template = get_conv_template(self.template)
        template.system_message = self.system_message
        history = [] if history is None else history
        for old_question, old_answer in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace("<image>", image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors="pd")
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        generation_config["eos_token_id"] = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        if isinstance(generation_output, paddle.Tensor):
            generation_output = generation_output.numpy().tolist()
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()

        if return_history:
            history = history + [(question, response)]
            return response, history
        return response

    @paddle.no_grad()
    def generate(
        self,
        pixel_values: Optional[paddle.Tensor] = None,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        visual_features: Optional[paddle.Tensor] = None,
        generation_config=None,
        output_hidden_states: Optional[bool] = None,
        **generate_kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided for generation.")
        if self.img_context_token_id is None:
            raise ValueError(
                "img_context_token_id is not initialized. Please call chat()/batch_chat() or set it manually."
            )

        generation_config = generation_config or {}
        max_new_tokens = int(generation_config.get("max_new_tokens", generate_kwargs.pop("max_new_tokens", 128)))
        eos_token_id = generation_config.get("eos_token_id", generate_kwargs.pop("eos_token_id", None))
        do_sample = generation_config.get("do_sample", generate_kwargs.pop("do_sample", False))
        repetition_penalty = float(
            generation_config.get("repetition_penalty", generate_kwargs.pop("repetition_penalty", 1.0))
        )
        top_k = generation_config.get("top_k", generate_kwargs.pop("top_k", None))
        top_p = generation_config.get("top_p", generate_kwargs.pop("top_p", None))
        temperature = generation_config.get("temperature", generate_kwargs.pop("temperature", None))
        pad_token_id = generation_config.get("pad_token_id", generate_kwargs.pop("pad_token_id", None))
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_ids = [int(token_id) for token_id in eos_token_id]
        elif eos_token_id is None:
            eos_token_ids = []
        else:
            eos_token_ids = [int(eos_token_id)]

        if do_sample:
            raise NotImplementedError("InternVLChatModel.generate currently only supports greedy decoding.")
        if top_k not in (None, 0, 1):
            raise NotImplementedError("InternVLChatModel.generate does not support top-k sampling yet.")
        if top_p not in (None, 1.0):
            raise NotImplementedError("InternVLChatModel.generate does not support top-p sampling yet.")
        if temperature not in (None, 1.0):
            raise NotImplementedError("InternVLChatModel.generate does not support temperature sampling yet.")
        if generate_kwargs:
            unsupported_args = ", ".join(sorted(generate_kwargs.keys()))
            raise NotImplementedError(f"Unsupported generation arguments: {unsupported_args}")

        if pixel_values is not None:
            vit_embeds = visual_features if visual_features is not None else self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            input_embeds = self._merge_input_embeds_with_vision_features(input_ids, input_embeds, vit_embeds)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = paddle.ones([batch_size, seq_length], dtype="int64")
        else:
            attention_mask = attention_mask.astype("int64")

        position_ids = self._build_position_ids(attention_mask)
        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
            output_hidden_states=output_hidden_states,
        )
        past_key_values = outputs.past_key_values
        next_token_logits = self._apply_repetition_penalty(outputs.logits[:, -1, :], input_ids, repetition_penalty)
        next_tokens = paddle.argmax(next_token_logits, axis=-1).astype("int64").unsqueeze(-1)
        generated_tokens = [next_tokens]
        generated_sequence = paddle.concat([input_ids, next_tokens], axis=-1)

        if eos_token_ids:
            eos_tensor = paddle.to_tensor(eos_token_ids, dtype="int64")
            finished = paddle.any(next_tokens == eos_tensor.reshape([1, -1]), axis=-1, keepdim=True)
        else:
            finished = paddle.zeros([batch_size, 1], dtype="bool")

        current_attention_mask = attention_mask

        for _ in range(max_new_tokens - 1):
            if bool(paddle.all(finished).item()):
                break

            current_attention_mask = paddle.concat(
                [current_attention_mask, paddle.ones([batch_size, 1], dtype=current_attention_mask.dtype)], axis=-1
            )
            position_ids = paddle.sum(current_attention_mask, axis=-1, keepdim=True).astype("int64") - 1

            outputs = self.language_model(
                input_ids=next_tokens,
                attention_mask=current_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                output_hidden_states=output_hidden_states,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = self._apply_repetition_penalty(
                outputs.logits[:, -1, :], generated_sequence, repetition_penalty
            )
            next_tokens = paddle.argmax(next_token_logits, axis=-1).astype("int64").unsqueeze(-1)

            if eos_token_ids:
                next_finished = paddle.any(next_tokens == eos_tensor.reshape([1, -1]), axis=-1, keepdim=True)
                pad_fill_id = eos_token_ids[0] if eos_token_ids else pad_token_id
                if pad_fill_id is None:
                    pad_fill_id = 0
                pad_fill = paddle.full_like(next_tokens, pad_fill_id)
                next_tokens = paddle.where(finished, pad_fill, next_tokens)
                finished = paddle.logical_or(finished, next_finished)

            generated_tokens.append(next_tokens)
            generated_sequence = paddle.concat([generated_sequence, next_tokens], axis=-1)

        return paddle.concat(generated_tokens, axis=1)


class InternVLChatForConditionalGeneration(InternVLChatModel):
    pass


__all__ = ["InternVLChatForConditionalGeneration", "InternVLChatModel"]
