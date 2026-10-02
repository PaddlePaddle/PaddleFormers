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
"""
GBQ function.
"""
import gc
import nntplib
import os
import time
import numpy as np
import ipdb
import copy

import paddle
from paddle import nn
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from paddle.distributed.fleet.meta_parallel import ColumnParallelLinear, RowParallelLinear
from paddle.quantization import PTQ

from paddleformers.transformers.qwen3_moe.modeling import Qwen3MoeDecoderLayer, Qwen3MoeModel

from paddlenlp.utils.log import logger
from paddlenlp.quantization.quantization_linear import (
            ColumnParallelQuantizationLinear,
            QuantizationLinear,
           RowParallelQuantizationLinear,
        )

import paddleslim
from paddleslim.quant.advanced import GPTQ
from paddleslim.common.wrapper_function import FuncWrapper

from quant_utils import (
    init_params,
    _clear_params,
    prepare_qconfig,
    get_scales,
    save_scales,
    save_moe_quant_w4a8_model,
    get_ptq_params,
    apply_gptq,
    show_progress,
    load_sharded_checkpoint
)
from custom_attention import QuantizedCustomAttentionLayer

def get_mean_scale_for_moe(act_scales, num_experts=128):
    new_act_scales={}
    for k, value in act_scales.items():
        if '.mlp.experts' in k:
            idx = k.split(".")[2]
            gate_proj_scale = []
            up_proj_scale = []
            down_proj_scale = []
            for j in range(num_experts):
                if act_scales["model.layers.{}.mlp.experts.{}.up_proj.activation_quanter".format(idx, j)] > 0.0:
                    gate_proj_scale.append(act_scales["model.layers.{}.mlp.experts.{}.gate_proj.activation_quanter".format(idx, j)])
                    up_proj_scale.append(act_scales["model.layers.{}.mlp.experts.{}.up_proj.activation_quanter".format(idx, j)])
                    down_proj_scale.append(act_scales["model.layers.{}.mlp.experts.{}.down_proj.activation_quanter".format(idx, j)])
            gate_mean=sum(gate_proj_scale)/len(gate_proj_scale)
            up_mean=sum(up_proj_scale)/len(up_proj_scale)
            down_mean=sum(down_proj_scale)/len(down_proj_scale)
            for j in range(num_experts):
                if act_scales["model.layers.{}.mlp.experts.{}.up_proj.activation_quanter".format(idx, j)] > 0.0:
                    new_act_scales["model.layers.{}.mlp.experts.{}.gate_proj.activation_quanter".format(idx, j)]=act_scales["model.layers.{}.mlp.experts.{}.gate_proj.activation_quanter".format(idx, j)]
                    new_act_scales["model.layers.{}.mlp.experts.{}.up_proj.activation_quanter".format(idx, j)]=act_scales["model.layers.{}.mlp.experts.{}.up_proj.activation_quanter".format(idx, j)]
                    new_act_scales["model.layers.{}.mlp.experts.{}.down_proj.activation_quanter".format(idx, j)]=act_scales["model.layers.{}.mlp.experts.{}.down_proj.activation_quanter".format(idx, j)]
                else:
                    new_act_scales["model.layers.{}.mlp.experts.{}.gate_proj.activation_quanter".format(idx, j)]=gate_mean
                    new_act_scales["model.layers.{}.mlp.experts.{}.up_proj.activation_quanter".format(idx, j)]=up_mean
                    new_act_scales["model.layers.{}.mlp.experts.{}.down_proj.activation_quanter".format(idx, j)]=down_mean
        else:
            new_act_scales[k]=value
    return new_act_scales

@paddle.no_grad()
def apply_block_gptq(model, predictor, ptq_dials, tgt_dials, args):
    """
    Gradual Block Quantization for IQ
    Only once complete calibration process
    PSS, AWQ, AutoClip, GPTQ and PTQ calibration process are all in here
    ptq_dials : batch_source_texts
    tgt_dials: batch_target_texts
    """

    logger.info("Starting block quantization...")
    last_layer_outputs = []
    pp_id = args.pp_id
    dp_degree = 1
    activation, weight, cachekv, q_config = prepare_qconfig(args)
    act_scales = {}
    weight_scales = {}
    cachekv_scales = {}
    model_to_quant = {}
    best_quant_policies = {}
    try:
        hcg = fleet.get_hybrid_communicate_group()
        rank = hcg.get_model_parallel_rank()
        nranks = hcg.get_model_parallel_world_size()
        dp_id = hcg.get_data_parallel_rank()
    except:
        rank = dist.get_rank()
        nranks = dist.get_world_size()
        dp_id = 0
    if args.lazy_load:
        if nranks == 1:
            state_dict = load_sharded_checkpoint(args.model_name_or_path, return_numpy=True)
        else:
            # For EB3.5, should load from xx.pdparams file now
            model_path = os.path.join(args.model_name_or_path, f"model_state.tp0{rank}.pdparams") 
            state_dict = paddle.load(model_path, return_numpy=True)
        ptq_state_dict = {}

    def get_block_out(sub_layer, layer_out, use_flash_attention, return_output=True):

        with paddle.amp.auto_cast(dtype="bfloat16"):
            decode_out = sub_layer(
                layer_out[0].cuda(), 
                attention_mask=layer_out[1].cuda() if not use_flash_attention else None, 
                position_embeddings=(layer_out[2][0].cuda(), layer_out[2][1].cuda())
                )
        if return_output:
            return decode_out

    num_layers = model.config.num_hidden_layers + 2
    start = time.perf_counter()
    block_index = 0
    
    model.to("cpu")
    paddle.device.cuda.empty_cache()
    gc.collect()
    time.sleep(10)
    paddle.device.cuda.empty_cache()

    for sub_name, sub_layer in model.named_sublayers():
        logger.info(f'processing: {sub_name} - {sub_layer.full_name()} - {type(sub_layer)}')
        if 'embed_tokens' in sub_name:
            logger.info(f"Block {block_index}: {sub_name}")
            sub_layer.to("gpu")
            # get embedding output
            logger.debug("Getting Embedding Output")
            if args.lazy_load:
                init_params(sub_layer, state_dict, sub_name, args.dtype)
                logger.debug(f"{sub_name} init params done")
            in_tokens = []
            for count, text in enumerate(ptq_dials):
                # TODO 显存问题
                if count>499:
                    break
                tokens = predictor._preprocess(text, tgt_dials[count])
                in_tokens.append(tokens)
            logger.info(f"ALL samples: {len(in_tokens)}")
            for idx in range(0, len(in_tokens)):
                logger.info(f'embed_tokens infer step: {idx}')
                input_map = in_tokens[idx]
                if input_map is None:
                    print('input map is None')
                    continue
                input_ids = input_map["input_ids"]
                # print("input_ids:", input_ids.tolist())
                attention_mask = input_map["attention_mask"] if "attention_mask" in input_map else None
                position_ids = input_map["position_ids"] if "position_ids" in input_map else None
                if position_ids is None:
                    past_length = 0
                    position_ids = paddle.arange(
                        past_length, paddle.shape(input_ids)[-1] + past_length, dtype=input_ids.dtype
                    )
                    position_ids = position_ids.unsqueeze(0)
                    position_ids = paddle.expand_as(position_ids, input_ids)
                
                embedding_output = sub_layer(input_ids)
                position_embeddings = model.model.rotary_emb(embedding_output, position_ids)

                batch_size, seq_length = input_ids.shape[:2]
                attention_mask = (
                    paddle.ones((batch_size, seq_length), dtype=paddle.bool)
                    if attention_mask is None
                    else attention_mask
                )
                attention_mask = Qwen3MoeModel._prepare_decoder_attention_mask(
                    attention_mask, (batch_size, seq_length), 0, embedding_output.dtype
                )  # [bs, 1, seq_len, seq_len]
                if model.config.use_flash_attention:
                    attention_mask = None if ((paddle.triu(attention_mask) == attention_mask).all().item()) else attention_mask

                if args.offload_data:
                    layer_out = (embedding_output.cpu(), attention_mask.cpu(), (position_embeddings[0].cpu(), position_embeddings[1].cpu()))
                else:
                    layer_out = (embedding_output, attention_mask, position_embeddings)
                last_layer_outputs.append(layer_out)
                del embedding_output, attention_mask, position_ids, position_embeddings

            del in_tokens

            show_progress(start, block_index, num_layers)
            sub_layer.to("cpu")
            paddle.device.cuda.empty_cache()
            gc.collect()

        elif isinstance(sub_layer, Qwen3MoeDecoderLayer):
            sub_layer.to("gpu")
            logger.info(f"Block {block_index}: {sub_name}")
            block_index += 1

            layer_idx = int(sub_name.split('.')[-1])

            if args.lazy_load:
                layer_name = 'layers.' + sub_name.split('.')[-1]
                init_params(sub_layer, state_dict, sub_name, args.dtype)
                logger.info(f"{layer_name} init done")

            cur_layer_outputs = []

            # Original layers outputs
            for idx, layer_out in enumerate(last_layer_outputs):
                logger.info(f'Decoder Layer infer step: {idx}')
                decode_out = get_block_out(sub_layer, layer_out, model.config.use_flash_attention)
                if args.offload_data:
                    cur_layer_outputs.append((decode_out.cpu(), layer_out[1].cpu(), (layer_out[2][0].cpu(), layer_out[2][1].cpu())))
                else:
                    cur_layer_outputs.append((decode_out, layer_out[1], layer_out[2]))

            # GPTQ for WINT4
            if args.gptq:
                logger.debug("Step: GPTQ")
                gptq = apply_gptq(sub_layer, predictor, args, ptq_dials, create_only=True)
                for idx, layer_out in enumerate(last_layer_outputs):
                    # if idx >=128:
                    #     break
                    get_block_out(sub_layer, layer_out, model.config.use_flash_attention, return_output=False)
                    logger.debug(f"gptq: {idx}")
                gptq.fasterquantmoe()
                del gptq
                paddle.device.cuda.empty_cache()
                gc.collect()
            
            # PTQ linears in current transformer block
            logger.debug("Step: PTQ Preparation")
            for cur_layer_name, linear_layer in sub_layer.named_sublayers():
                if type(linear_layer) in [ColumnParallelLinear, RowParallelLinear, paddle.nn.Linear]:
                    q_config.add_name_config([linear_layer.full_name()], activation=activation, weight=weight)
                    logger.debug(f"w4a8: {cur_layer_name} {linear_layer.full_name()}")
                    
                if type(linear_layer) in [FuncWrapper]:
                    # set both act and weight for attention, actually act-k and act-v are quantized
                    q_config.add_name_config([linear_layer.full_name()], weight=cachekv[0], activation=cachekv[1],)
                    logger.debug(f"[Cache-KV Quant] {linear_layer.full_name()}")
            ptq = PTQ(q_config)
            sub_layer = ptq.quantize(sub_layer, inplace=True)

            # PTQ sampling
            for cur_layer_name, cur_layer in sub_layer.named_sublayers():
                if type(cur_layer) in [ColumnParallelQuantizationLinear, QuantizationLinear, RowParallelQuantizationLinear]:
                    cur_layer.remove_dequantize_weight()
            if args.quant_type in ["WINT4", "WINT8", "W4A16", "W8A16"]:
                # only one forward needed for weight only
                get_block_out(sub_layer, layer_out[0], model.config.use_flash_attention, return_output=False)
            else:
                ptq_step = 0
                for layer_out in last_layer_outputs:
                    ptq_step += 1
                    logger.debug(f"ptq: {ptq_step}")
                    get_block_out(sub_layer, layer_out, model.config.use_flash_attention, return_output=False)

            act_scales, weight_scales, cachekv_scales = get_scales(model, act_scales, weight_scales, \
                            cachekv_scales, dp_degree, nranks, rank, best_quant_policies)
            sub_layer = ptq.convert(sub_layer, inplace=True)

            if args.lazy_load:
                ptq_state_dict = get_ptq_params(sub_layer, ptq_state_dict, sub_name) 

            for cur_layer_name, cur_layer in sub_layer.named_sublayers():
                if type(cur_layer) in [ColumnParallelQuantizationLinear, QuantizationLinear, RowParallelQuantizationLinear]:
                    cur_layer.remove_dequantize_weight()
            del last_layer_outputs
            last_layer_outputs = cur_layer_outputs

            del cur_layer_outputs

            if args.lazy_load:
                _clear_params(sub_layer, state_dict, sub_name)
            show_progress(start, block_index, num_layers)
            sub_layer.to("cpu")
            paddle.device.cuda.empty_cache()
            gc.collect()

    act_scales=get_mean_scale_for_moe(act_scales)
    save_scales(args, act_scales, weight_scales, cachekv_scales, mp_id=rank, dp_id=dp_id)

    
    paddle.device.cuda.empty_cache()
    gc.collect()
    time.sleep(60*int(rank))


    if nranks == 1:
        model_path = os.path.join(args.save_path, "model_state.pdparams")
    else:
        model_path = os.path.join(args.save_path, f"model_state.tp0{rank}.pdparams")

    if args.lazy_load:
        # get uncleared params
        for k, v in state_dict.items():
            ptq_state_dict[k] = v
        # save model first, since init new params may cause gpu memory overflow
        save_quant_model(ptq_state_dict, model_path, dp_id=dp_id)
        for k, v in ptq_state_dict.items():
            ptq_state_dict[k] = paddle.to_tensor(v, dtype=args.dtype)
            if 'scale' in k:
                ptq_state_dict[k] = ptq_state_dict[k].cast('float32')
        # cleared params are not initialized, need re-init
        for k, v in model.state_dict().items():
            if not v._is_initialized():
                v.get_tensor()._share_data_with(ptq_state_dict[k].get_tensor())
        model.set_state_dict(ptq_state_dict)
    else:
        gc.collect()
        # time.sleep(30*int(rank))
        paddle.device.cuda.empty_cache()
        gc.collect()
        save_moe_quant_w4a8_model(args,model.state_dict(), model_path, pp_id=pp_id, weight_scales=weight_scales)
        logger.info(f"Save quant model to {args.save_path}")
        # time.sleep(40*int(8-rank))
    logger.debug("-------------------gptq Done------------------")
