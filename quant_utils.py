# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
PTQ functions.
"""
import json
import os
import random
import time
import numpy as np
from functools import partial
import paddle
import paddleslim
import paddle.distributed.fleet as fleet
from paddle.distributed.fleet.meta_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from paddle.nn.quant import weight_quantize
import sys
import os

from paddle.quantization import PTQ, QAT, QuantConfig
from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
from paddlenlp.peft.lora import LoRALinear
from paddlenlp.peft.lora.lora_quant_layers import QuantedLoRALinear
from paddleslim.utils.log import logger
from paddlenlp.transformers.model_utils import _add_variant, load_state_dict
from paddleslim.quant.advanced import (
    AutoClip,
    AWQSearch,
    EMASampler,
    LayerWiseQuantError,
    MultiStepSampler,
    PieceWiseSearch,
    SmoothSearchV2,
    Shift,
    ReorderFFNWeight,
    Smooth,
    GPTQ,
    moe_shared_scale,
    TokenWiseClipping
)
from paddleslim.quant.advanced.utils import find_parent_layer_and_sub_name
from paddleslim.quant.layers import (
    QuantizedColumnParallelLinear,
    QuantizedRowParallelLinear
)
from paddleslim.common.wrapper_function import FuncWrapper
from custom_attention import QuantizedCustomAttentionLayer
from abq import AdaptiveBaggingQuant
from paddleslim.quant.observers import (
    AbsMaxChannelWiseWeightObserver,
    AbsmaxObserver,
    AvgHeadwiseObserver,
    GroupWiseWeightObserver,
    TokenQuantileObserver,
    KCacheChannelWiseObserver,
    AsymCacheKVObserver
)
from paddleslim.quant.quanters import PACTQuanter
from paddleslim.quant.quanters.channel_wise_abs_max import (
    FakeQuanterChannelWiseAbsMaxObserver,
)
from paddleslim.quant.observers.abs_max_weight import AbsMaxChannelWiseWeightObserverLayer
from paddleslim.quant.observers.abs_max import AbsmaxObserverLayer
from paddleslim.quant.observers.token_quantile import TokenQuantileObserverLayer
from paddleslim.quant.observers.groupwise import GroupWiseWeightObserverLayer
from paddleslim.quant.observers.avg_headwise import AvgHeadwiseObserverLayer
from paddleslim.quant.observers.kcache_channelwise import KCacheChannelWiseObserverLayer
from paddleslim.quant.observers.asym_cachekv import AsymCacheKVObserverLayer
from paddleslim.quant.observers.abs_max_tokenwise import AbsmaxTokenwiseObserverLayer
import paddle.distributed as dist

def load_sharded_checkpoint(folder, variant=None, return_numpy=False):
    """

    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        folder (`str` or `os.PathLike`): A path to a folder containing the sharded checkpoint.
        variant (`str`): The model variant.

    """
    # Load the index
    pdparams_file = os.path.join(folder, _add_variant("model_state.pdparams", variant))
    lora_pdparams_file = os.path.join(folder, _add_variant("lora_model_state.pdparams", variant))
    safetensors_file = os.path.join(folder, _add_variant("model.safetensors", variant))
    if os.path.isfile(pdparams_file):
        return paddle.load(pdparams_file, return_numpy=return_numpy)
    if os.path.isfile(lora_pdparams_file):
        return paddle.load(lora_pdparams_file, return_numpy=return_numpy)
    if os.path.isfile(safetensors_file):
        try:
            from paddlenlp.utils.safetensors import fast_load_file as safe_load_file
        except:
            from safetensors.numpy import load_file as safe_load_file

        state_dict = safe_load_file(safetensors_file)
        if not return_numpy:
            for key in list(state_dict.keys()):
                if isinstance(state_dict[key], np.ndarray):
                    state_dict[key] = paddle.Tensor(state_dict.pop(key), zero_copy=True)
        return state_dict

    index_file = os.path.join(folder, _add_variant(PADDLE_WEIGHTS_INDEX_NAME, variant))
    safe_index_file = os.path.join(folder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
    safe_master_file = os.path.join(folder, _add_variant(SAFE_MASTER_WEIGHTS_INDEX_NAME, variant))
    safe_peft_file = os.path.join(folder, _add_variant(SAFE_PEFT_WEIGHTS_INDEX_NAME, variant))

    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)
    safe_master_present = os.path.isfile(safe_master_file)
    safe_peft_present = os.path.isfile(safe_peft_file)

    load_safe = False
    load_index = None
    if safe_index_present:
        load_safe = True  # load safe due to preference
        load_index = safe_index_file
    elif safe_master_present:
        load_safe = True
        load_index = safe_master_file
    elif index_present:
        load_index = index_file
    elif safe_peft_present:
        load_safe = True
        load_index = safe_peft_file
    else:
        raise ValueError(f"Could not find {index_file} or {safe_index_file} or {safe_peft_file}")

    if load_safe:
        try:
            from paddlenlp.utils.safetensors import fast_load_file as safe_load_file
        except:
            from safetensors.numpy import load_file as safe_load_file

    with open(load_index, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))
    loader = safe_load_file if load_safe else partial(paddlenlp_load, map_location="np" if return_numpy else "cpu")

    ret = {}
    for shard_file in tqdm(shard_files):
        state_dict = loader(os.path.join(folder, shard_file))
        ret.update(state_dict)

    if not return_numpy:
        for key in list(ret.keys()):
            if isinstance(ret[key], np.ndarray):
                ret[key] = paddle.Tensor(ret.pop(key), zero_copy=True)

    return ret


def show_progress(start, idx, steps):
    """
    Show progress
    """
    c = idx / steps * 100
    a = "*" * int(c)
    b = "·" * (100 - int(c))
    dur = time.perf_counter() - start
    logger.info("\r{:.2f}%[{}->{}] Cost time {:.2f}s".format(c, a, b, dur))
    time.sleep(0.1)


def get_ptq_params(model, ptq_state_dict, sub_name):
    """
    Get ptq params from quant model
    """
    for name, param in model.named_parameters():
        full_name = sub_name + "." + name
        ptq_state_dict[full_name] = np.array(param.value().get_tensor())
    return ptq_state_dict

@paddle.no_grad()
def _clear_params(model, state_dict=None, sub_name=None):
    """
    Clear params
    """
    for k, v in model.state_dict().items():
        # 清除参数的值
        v.value().get_tensor()._clear()
        # if state_dict is not None:
        #     拼接参数名
        #    name = sub_name + "." + k
        #    if name in state_dict:
        #     如果拼接后的参数名在state_dict中存在
        #    if name in state_dict:
        #            从state_dict中删除该参数
        #        del state_dict[sub_name + "." + k]


def init_params(sub_layer, state_dict, sub_name, dtype):
    """
    Init params and set state_dict
    """
    new_dict = {}
    for k, v in state_dict.items():
        if sub_name in k:
            weight_name = k.replace(sub_name + ".", "")
            # load from numpy, so we need to convert to bfloat16 firstly and then cast to other dtype
            new_dict[weight_name] = paddle.to_tensor(v, dtype='bfloat16').cast(dtype).cuda()
    for k, v in sub_layer.state_dict().items():
        if not v._is_initialized():
            v.get_tensor()._share_data_with(new_dict[k].get_tensor())
    sub_layer.set_state_dict(new_dict)
 

def prepare_qconfig(args):
    """
    Prepare qconfig
    """
    if 'C8' in args.quant_type:
        quant_type = args.quant_type.replace('C8', '')
        cachekv_quant = True
        cachekv_quant_bits = 8
    elif 'C4' in args.quant_type:
        quant_type = args.quant_type.replace('C4', '')
        cachekv_quant = True
        cachekv_quant_bits = 4
    elif 'C2' in args.quant_type:
        quant_type = args.quant_type.replace('C2', '')
        cachekv_quant = True
        cachekv_quant_bits = 2
    else:
        quant_type = args.quant_type.replace('C16', '')
        cachekv_quant = False

    q_config = QuantConfig(activation=None, weight=None)
    if quant_type == "W8A8":
        activation = AbsmaxObserver(quant_bits=8)
        weight = AbsMaxChannelWiseWeightObserver(quant_bits=8)
    elif quant_type in ["WINT4", "W4A16"]:
        activation = None
        weight = GroupWiseWeightObserver(quant_bits=4, group_size=args.group_size)
    elif quant_type in ["WINT8", "W8A16"]:
        activation = None
        weight = AbsMaxChannelWiseWeightObserver(quant_bits=8)
    elif quant_type == "W4A8":
        activation = AbsmaxObserver(quant_bits=8)
        weight = AbsMaxChannelWiseWeightObserver(quant_bits=4)
    else:
        raise ValueError("quant_type should be in ['W8A8', 'WINT4', 'WINT8', 'W4A8', 'W4A16', 'W8A16']")
 
    q_config.add_qat_layer_mapping(ColumnParallelLinear, QuantizedColumnParallelLinear)
    q_config.add_qat_layer_mapping(RowParallelLinear, QuantizedRowParallelLinear)

    cachekv = None
    if cachekv_quant: 
        if cachekv_quant_bits == 8:
            cachekv = [AvgHeadwiseObserver(quant_bits=cachekv_quant_bits, moving_avg=True, quant_axis=1,do_fp8_quant=True),
            AvgHeadwiseObserver(quant_bits=cachekv_quant_bits, moving_avg=True, quant_axis=1,do_fp8_quant=True)]
            # cachekv = [KCacheChannelWiseObserver(quant_bits=cachekv_quant_bits, symmetric=True), \
            #         KCacheChannelWiseObserver(quant_bits=cachekv_quant_bits, symmetric=True)]
            q_config.add_qat_layer_mapping(FuncWrapper, QuantizedCustomAttentionLayer)
        elif cachekv_quant_bits == 4:
            if args.abq:
                cachekv = [AsymCacheKVObserver(quant_bits=cachekv_quant_bits, symmetric=False, quant_axis=[1, 3]), \
                    AsymCacheKVObserver(quant_bits=cachekv_quant_bits, symmetric=False, quant_axis=[1, 3])]
            else:
                cachekv = [KCacheChannelWiseObserver(quant_bits=cachekv_quant_bits, symmetric=False), \
                    KCacheChannelWiseObserver(quant_bits=cachekv_quant_bits, symmetric=False)]
            q_config.add_qat_layer_mapping(FuncWrapper, QuantizedCustomAttentionLayer)
        else:
            raise ValueError('cachekv_quant_bits should be 8 or 4, 2bit is not supported for now.')
    return activation, weight, cachekv, q_config

 
def get_scales(model, act_scales, weight_scales, cachekv_scales, 
               dp_degree=1, mp_degree=1, mp_id=0, best_quant_policies=None):
    """
    get scales
    """
    def gather_scale(cur_layer, dp_degree, mp_degree, mp_id):
        scale = cur_layer.scales()
        if dp_degree > 1:
            scale_list = []
            paddle.distributed.all_gather(scale_list, scale)
            gathered_scale = paddle.concat(
                            [
                                paddle.reshape_(
                                    scale_list[r * mp_degree + mp_id],
                                    shape=[1] + scale_list[r * mp_degree + mp_id].shape) for r in range(dp_degree)
                            ],
                            axis=0).max(axis=0, keepdim=False)
            paddle.assign(gathered_scale, cur_layer._scale)
            return gathered_scale
        else:
            return scale
    
    def gather_min_max(cur_layer, max_values, min_values, dp_degree, mp_degree, mp_id, quant_bits):
        bnt = (1 << (quant_bits - 1)) - 1
        qmin = -bnt - 1
        qmax = bnt
        if dp_degree > 1:
            max_list, min_list = [], []
            paddle.distributed.all_gather(max_list, max_values)
            gathered_max = paddle.concat(
                            [
                                paddle.reshape_(
                                    max_list[r * mp_degree + mp_id],
                                    shape=[1] + max_list[r * mp_degree + mp_id].shape) for r in range(dp_degree)
                            ],
                            axis=0).max(axis=0, keepdim=False)
            paddle.distributed.all_gather(min_list, min_values)
            gathered_min = paddle.concat(
                            [
                                paddle.reshape_(
                                    min_list[r * mp_degree + mp_id],
                                    shape=[1] + min_list[r * mp_degree + mp_id].shape) for r in range(dp_degree)
                            ],
                            axis=0).min(axis=0, keepdim=False)
        else:
            gathered_max = max_values
            gathered_min = min_values
        gathered_scale = gathered_max - gathered_min
        gathered_scale = paddle.to_tensor(gathered_scale / float(qmax - qmin), dtype="float32")
        gathered_zp = qmin - paddle.round(gathered_min / gathered_scale)
        gathered_zp = paddle.clip(gathered_zp, qmin, qmax)
        cur_layer._scale = gathered_scale
        cur_layer._zero_point = gathered_zp
        return gathered_scale, gathered_zp

    for cur_name, cur_layer in model.named_sublayers():
        if 'layer.' in cur_name:
            cur_name = cur_name.replace('layer.', '')
        if type(cur_layer) in [AbsMaxChannelWiseWeightObserverLayer, GroupWiseWeightObserverLayer] \
                and "_observer" not in cur_name:
            scale = gather_scale(cur_layer, dp_degree, mp_degree, mp_id)
            weight_scales[cur_name] = scale.cast("float32").numpy().tolist()
        if type(cur_layer) in [AbsmaxObserverLayer, TokenQuantileObserverLayer, AbsmaxTokenwiseObserverLayer] and "_observer" not in cur_name:
            scale = gather_scale(cur_layer, dp_degree, mp_degree, mp_id)
            if type(scale) in [int]:
                act_scales[cur_name] = float(scale)
            else:
                act_scales[cur_name] = float(scale.cast("float32"))
            logger.debug(f"{cur_name}, {act_scales[cur_name]}")
        # 对量化层只能包一层oberserver，需要下述代码
        # if type(cur_layer) in [AbsmaxObserverLayer]:
        #     scale = gather_scale(cur_layer, dp_degree, mp_degree, mp_id)
        #     if type(scale) in [int]:
        #         act_scales[cur_name] = float(scale)
        #     else:
        #         act_scales[cur_name] = float(scale.cast("float32"))
        #     logger.debug(f"{cur_name}, {act_scales[cur_name]}")
        if type(cur_layer) in [AvgHeadwiseObserverLayer, KCacheChannelWiseObserverLayer] \
             and "_observer" not in cur_name:
            # hard code for inference
            cur_name = cur_name.replace('attn_func.activation_quanter_v', 'cachev_matmul.activation_quanter')
            cur_name = cur_name.replace('attn_func.activation_quanter_k', 'cachek_matmul.activation_quanter')
            scale = gather_scale(cur_layer, dp_degree, mp_degree, mp_id)
            cachekv_scales[cur_name] = scale.cast("float32").numpy().tolist()
            logger.debug(f"{cur_name}, {cachekv_scales[cur_name][0]}")
            # save zeropints in scale file if its not 0 or list of 0s
            cachekv_scales[cur_name + '.zero_point'] = cur_layer.zero_points().cast("float32").numpy().tolist()
            logger.debug(f"{cur_name + '.zero_point'}, {cachekv_scales[cur_name + '.zero_point']}")
        if type(cur_layer) == AsymCacheKVObserverLayer and "_observer" not in cur_name:
            cur_name = cur_name.replace('attn_func.activation_quanter_v', 'cachev_matmul.activation_quanter')
            cur_name = cur_name.replace('attn_func.activation_quanter_k', 'cachek_matmul.activation_quanter')
            layerid = int(cur_name.split('.')[3])
            kv_flag = cur_name.split('.')[5][5] + "_int4"
            kv_max_name = cur_name.split('.')[5][5] + "_max"
            kv_min_name = cur_name.split('.')[5][5] + "_min"
            best_kv_max = best_quant_policies[layerid].get(kv_max_name)
            best_kv_min = best_quant_policies[layerid].get(kv_min_name)

            scales, zps = gather_min_max(cur_layer, best_kv_max, best_kv_min, dp_degree, 
                                         mp_degree, mp_id, cur_layer._quant_bits)
            kv_losses = best_quant_policies[layerid].get('kv_loss')
            cachekv_scales[cur_name] = scales.cast("float32").numpy().tolist()
            cachekv_scales[cur_name + '.zero_point'] = zps.cast("float32").numpy().tolist()
            logger.debug(f"quant_bits: {cur_layer._quant_bits}, kv_losses: {kv_losses}, \
                scales: {cur_layer.scales()}, zps: {cur_layer.zero_points()}")
    return act_scales, weight_scales, cachekv_scales


def save_scales(args, act_scales, weight_scales, cachekv_scales, mp_id=0, dp_id=0):
    """
    save scales
    """
    if dp_id == 0:
        if act_scales:
            with open(f"{args.save_path}/act_scales_{mp_id}.json", "w") as outfile:
                json.dump(act_scales, outfile)
                logger.debug("save act scales")
        if weight_scales:
            with open(f"{args.save_path}/weight_scales_{mp_id}.json", "w") as outfile:
                json.dump(weight_scales, outfile)
                logger.debug("save weight scales")
        if cachekv_scales:
            with open(f"{args.save_path}/cachekv_scales_{mp_id}.json", "w") as outfile:
                json.dump(cachekv_scales, outfile)
                logger.debug("save cachekv scales")
 

def save_quant_model(state_dict, save_path, dp_id=0):
    """
    Save quant model
    """
    if dp_id == 0:
        new_scale_dict = {}
        for k, v in state_dict.items():
            # hard code for inference
            # if 'weight_quanter._dequanter._scales' in k:
            #     continue
            if 'layer.' in k:
                new_scale_dict[k.replace('layer.', '')] = v
            else:
                new_scale_dict[k] = v
        # for k, v in state_dict.items():
        #             if 'ernie.layers.' in k:
        #                 new_k=k.split('ernie.layers.')[1].split('.')[0]
        #                 orin_index=int(new_k)
        #                 new_k=str(int(new_k)+27)
        #                 new_k='ernie.layers.'+new_k+k.split('ernie.layers.'+str(orin_index))[1]
        #                 new_scale_dict[new_k] = v
        #             else:
        #                 new_scale_dict[k] = v
        # del state_dict
        # import gc
        # gc.collect()
        for k,v in new_scale_dict.items():
            if 'lm_head' in k:
                continue
            if '.experts' in k:
                continue
            else:
                if '_proj.weight' in k and len(v.shape) == 2:
                    paddle.assign(v.cast(paddle.int8), v)
        paddle.save(new_scale_dict, save_path)
        logger.info(f"Save model to {save_path}")

def merge_and_valid_shared_weights(tensor_list):
    assert len(tensor_list) > 0, "smooth weights or shift biases must not be empty"
    ret_tensor = tensor_list[0]
    for i in range(1, len(tensor_list)):
        cur_tensor = tensor_list[i]
        compare = paddle.where(ret_tensor==cur_tensor, 0, 1)
        assert paddle.sum(compare) == 0, "smooth or shift is not shared"
    return ret_tensor

def save_moe_quant_w4a8_model(args,state_dict, save_path, pp_id=0, weight_scales=None, share_smooth=True):
    """
    Save quant model
    """
    num_experts = 64
    paddle.set_default_dtype("bfloat16")
    paddle.set_device("cpu")
    ffn_hidden_size = 28672
    hidden_size = 8192
    convert_scale_dict={}
    
    for k, v in state_dict.items():
        if 'quanter._scales' in k or "quanter._zero_point" in k:
            continue
        else:
            if 'layer.' in k: continue  
            
            elif 'model.layers.' in k:
                if '.mlp.experts' in k:
                    if ".weight" in k and ("up_proj" in k or "gate_proj" in k or "down_proj" in k ):
                        v = v.cast(paddle.int8)
                        convert_scale_dict[k] = v
                        logger.debug('casting {} to int8'.format(k))
                else:
                    convert_scale_dict[k] = v
            else:
                convert_scale_dict[k] = v

    paddle.save(convert_scale_dict, save_path)
    logger.info(f"Save model to {save_path}") 

def save_moe_quant_model(state_dict, save_path, pp_id=0):
    """
    Save quant model
    """
    num_experts = 48
    paddle.set_default_dtype("bfloat16")
    paddle.set_device("cpu")
    ffn_hidden_size = 36864
    num_layers = 27
    num_attention_heads = 96
    num_key_value_heads = 8
    hidden_size = 12288
    mp_size = 8
    num_experts = 48
    export_model_type = 'WINT8'
    int8_moe_method = "weight-only-int4"
    convert_scale_dict = {}
    hcg = fleet.get_hybrid_communicate_group()
    rank = hcg.get_model_parallel_rank()
    ffn_hidden_size = ffn_hidden_size // mp_size
    for k, v in state_dict.items():
        if k.endswith("self_attn.qkv_proj.weight"):
            idx = k.split(".")[2]
            idx_for_save = str(int(idx) + pp_id*27)
            print("idx", idx)
            up_gate_proj_weight = []
            up_gate_proj_bias = []
            down_proj_weight = []
            down_proj_bias = []
            for j in range(num_experts):
                up_gate_proj_weight.append(paddle.to_tensor(state_dict["ernie.layers.{}.mlp.experts.{}.up_gate_proj.weight".format(idx, j)], dtype=paddle.get_default_dtype()))
                up_gate_proj_bias.append(paddle.to_tensor(state_dict["ernie.layers.{}.mlp.experts.{}.up_gate_proj.bias".format(idx, j)], dtype=paddle.get_default_dtype()))
                down_proj_weight.append(paddle.to_tensor(state_dict["ernie.layers.{}.mlp.experts.{}.down_proj.weight".format(idx, j)], dtype=paddle.get_default_dtype()))
                down_proj_bias.append(paddle.to_tensor(state_dict["ernie.layers.{}.mlp.experts.{}.down_proj.bias".format(idx, j)], dtype=paddle.get_default_dtype()))
            # hard code for inference
            # if 'weight_quanter._dequanter._scales' in k:
            #     continue
            ffn1_weight_tensor = paddle.to_tensor(paddle.concat(up_gate_proj_weight, axis=0), dtype=paddle.get_default_dtype()).reshape([num_experts, hidden_size, -1])
            ffn1_weight_tensor_list = []
            ffn1_weight_scale_tensor_list = []
            for i in range(num_experts):
                ffn1_weight_tensor_i, ffn1_weight_scale_tensor_i = weight_quantize(
                        ffn1_weight_tensor[i], algo="weight_only_int4", arch=80
                    )
                ffn1_weight_tensor_list.append(ffn1_weight_tensor_i.reshape([hidden_size, ffn_hidden_size // mp_size]))
                ffn1_weight_scale_tensor_list.append(ffn1_weight_scale_tensor_i)
            ffn1_weight_tensor = paddle.concat(ffn1_weight_tensor_list, axis=0)
            ffn1_weight_scale_tensor_list = paddle.concat(ffn1_weight_scale_tensor_list, axis=0)
            convert_scale_dict["ffn1_weights_scales_{}".format(idx_for_save)] = ffn1_weight_scale_tensor_list.cast(paddle.get_default_dtype()).reshape([num_experts, -1])
            convert_scale_dict["ffn1_weights_{}".format(idx_for_save)] = ffn1_weight_tensor.reshape([num_experts, hidden_size, -1])
            convert_scale_dict["ffn1_biases_{}".format(idx_for_save)] = paddle.to_tensor(paddle.concat(up_gate_proj_bias, axis=0), dtype=paddle.get_default_dtype()).reshape([num_experts, -1])
            ffn2_weight_tensor = paddle.to_tensor(paddle.concat(down_proj_weight, axis=0), dtype=paddle.get_default_dtype()).reshape([num_experts, -1, hidden_size])
            ffn2_baisss = paddle.concat(down_proj_bias, axis=0)
            print("rank", rank)
            if rank > 0:
                ffn2_baisss.zero_()
                print(f'removing bias for rank:{rank}')
            else:
                print(f'keeping bias for rank:{rank}')

            ffn2_weight_tensor = paddle.to_tensor(paddle.concat(down_proj_weight, axis=0), dtype=paddle.get_default_dtype()).reshape([num_experts, -1, hidden_size])
            ffn2_weight_tensor_list = []
            ffn2_weight_scale_tensor_list = []
            for i in range(num_experts):
                ffn2_weight_tensor_i, ffn2_weight_scale_tensor_i = weight_quantize(
                        ffn2_weight_tensor[i], algo="weight_only_int4", arch=80
                    )
                ffn2_weight_tensor_list.append(ffn2_weight_tensor_i.reshape([ffn_hidden_size // mp_size, hidden_size // 2]))
                ffn2_weight_scale_tensor_list.append(ffn2_weight_scale_tensor_i)
            ffn2_weight_tensor = paddle.concat(ffn2_weight_tensor_list, axis=0)
            ffn2_weight_scale_tensor_list = paddle.concat(ffn2_weight_scale_tensor_list, axis=0)
            convert_scale_dict["ffn2_weights_scales_{}".format(idx_for_save)] = ffn2_weight_scale_tensor_list.cast(paddle.get_default_dtype()).reshape([num_experts, -1])
            convert_scale_dict["ffn2_weights_{}".format(idx_for_save)] = ffn2_weight_tensor.reshape([num_experts, ffn_hidden_size // mp_size, -1])

    if k.endswith("self_attn.qkv_proj.weight"):
            idx = k.split(".")[2]
    for k, v in state_dict.items():
        if '.experts' in k:
                continue
        elif 'weight_quanter._dequanter._scales' in k:
                continue
        else:
            if pp_id>0:
                if 'lm_head' in k:
                    continue
                elif 'layer.' in k:
                    if 'ernie.layers.' in k:
                        new_k=k.split('ernie.layers.')[1].split('.')[0]
                        orin_index=int(new_k)
                        new_k=str(int(new_k)+27*pp_id)
                        new_k='ernie.layers.'+new_k+k.split('ernie.layers.'+str(orin_index))[1]
                        convert_scale_dict[new_k.replace('layer.', '')] = v
                    else:
                        convert_scale_dict[k.replace('layer.', '')] = v
                else:
                    if 'ernie.layers.' in k:
                        new_k=k.split('ernie.layers.')[1].split('.')[0]
                        orin_index=int(new_k)
                        new_k=str(int(new_k)+27*pp_id)
                        new_k='ernie.layers.'+new_k+k.split('ernie.layers.'+str(orin_index))[1]
                        convert_scale_dict[new_k] = v
                    else:
                        convert_scale_dict[k] = v
            else:          
                if 'layer.' in k:
                    if 'ernie.layers.' in k:
                        new_k=k.split('ernie.layers.')[1].split('.')[0]
                        orin_index=int(new_k)
                        new_k=str(int(new_k)+27*pp_id)
                        new_k='ernie.layers.'+new_k+k.split('ernie.layers.'+str(orin_index))[1]
                        convert_scale_dict[new_k.replace('layer.', '')] = v
                    else:
                        convert_scale_dict[k.replace('layer.', '')] = v
                else:
                    if 'ernie.layers.' in k:
                        new_k=k.split('ernie.layers.')[1].split('.')[0]
                        orin_index=int(new_k)
                        new_k=str(int(new_k)+27*pp_id)
                        new_k='ernie.layers.'+new_k+k.split('ernie.layers.'+str(orin_index))[1]
                        convert_scale_dict[new_k] = v
                    else:
                        convert_scale_dict[k] = v

    paddle.save(convert_scale_dict, save_path)
    logger.info(f"Save model to {save_path}")

def calibration(predictor, ptq_dials, args, max_step=None, smooth=None):
    """
    Calibration
    """
    max_step = len(ptq_dials) if max_step is None else max_step
    with paddle.no_grad():
        for idx in range(0, len(ptq_dials), args.batch_size):
            batch_dials = ptq_dials[idx : idx + args.batch_size]
            out = predictor.predict_ptq(batch_dials)
            if smooth is not None and out != {"result": 'No result for invalid input'}:
                smooth.step += 1
            if idx % 10 == 0:
                logger.debug(f"Sample Step: {idx}")
            if idx >= max_step:
                break
 
def apply_shift(model, predictor, args, ptq_model_config, ptq_dials, create_only=False):
    """
    Shift
    """
    shift_sampler = EMASampler()
    shift = Shift(
        model=model,
        model_config=ptq_model_config,
        sample_function=shift_sampler,
        shift_all_linears=True,
    )
    if create_only:
        return shift
    calibration(predictor, ptq_dials, args)
    shift.update_weight()
    del shift, shift_sampler
 
def apply_smooth(model, predictor, args, ptq_model_config, ptq_dials, 
                    max_step=None, no_search=False, create_only=False):
    """
    Smooth
    """
    logger.debug("------------------Start Smooth-------------------")
    smooth_sampler = MultiStepSampler()
    if args.smooth_method == "smoothquant":
        if args.smooth_search_v2:
            search_func = SmoothSearchV2(
                weight_bits_length=8,
                act_bits_length=8,
                search_min=0.1,
                search_step=100,
                weight_quant_method='abs_max_channel_wise',
                act_quant_method="abs_max",
                dp_degree=args.data_parallel_degree,
                )
        else:
            search_func = PieceWiseSearch(
                k_piece=args.k_piece,
                bits_length=8,
                search_piece=args.search_piece,
                search_alpha_min=0.1,
                search_alpha_max=0.9,
                search_scale_min=1.0,
                search_scale_max=10.0,
                use_clip=args.use_clip,
                weight_quant_method="abs_max_channel_wise",
                act_quant_method="abs_max",
                dp_degree=args.data_parallel_degree,
            )
    elif args.smooth_method == "awq":
        search_func = AWQSearch(
            n_grid=20,
            bits_length=4,
            weight_quant_method="abs_max_channel_wise",
        )
    smooth = Smooth(
        model,
        ptq_model_config,
        alpha=0.5,
        smooth_all_linears=True,
        sample_function=smooth_sampler,
        search_function=search_func if not no_search else None,
        start_sample_step=args.start_sample_step,
        smooth_method=args.smooth_method,
    )
    if create_only:
        return smooth
        
    calibration(predictor, ptq_dials, args, max_step=max_step, smooth=smooth)
    smooth.update_weight()
    del smooth, smooth_sampler, search_func
 
    if args.load_smooth_model and not args.load_quant_model:
        logger.info(f"Load model checkpoint from {args.load_smooth_path}")
        model_dict = load_sharded_checkpoint(args.load_smooth_path, return_numpy=True)
        model.set_dict(model_dict)
 
    if args.save_smooth_model:
        try:
            hcg = fleet.get_hybrid_communicate_group()
            rank = hcg.get_model_parallel_rank()
            nranks = hcg.get_model_parallel_world_size()
            dp_id = hcg.get_data_parallel_rank()
        except:
            rank = dist.get_rank()
            nranks = dist.get_world_size()
            dp_id = 0
        if nranks == 1:
            model_path = os.path.join(args.save_smooth_path, "model_state.pdparams")
        else:
            model_path = os.path.join(args.save_path, f"model_state.tp0{rank}.pdparams")
        save_quant_model(model.state_dict(), model_path, dp_id=dp_id)
 
def apply_token_wise_clipping(model, predictor, args, ptq_model_config, ptq_dials, max_step=None):
    """
    token wise clipping
    """
    logger.debug("------------------Start Token_wise_clipping-------------------")
    token_wise_clipping = TokenWiseClipping(
        model,
        ptq_model_config,
        )
    max_step = len(ptq_dials) if max_step is None else max_step

    fp_input, fp_output = [], []

    with paddle.no_grad():
        for idx in range(0, len(ptq_dials), args.batch_size):
            batch_dials = ptq_dials[idx : idx + args.batch_size]
            input_map = predictor.preprocess(batch_dials, extra_infos=None, fast_ptq_sampling=True)
            if input_map is None:
                continue
            fp_input.append(input_map)
            output = model(**input_map)
            fp_output.append(output[0])
            if idx % 10 == 0:
                logger.debug(f"Token Wise Clipping Sample Step: {idx}")
            if idx >= max_step:
                break
        token_wise_clipping.token_wise_clipping(fp_input, fp_output)

def apply_autoclip(model, predictor, args, ptq_dials, create_only=False):
    """
    AutoClip
    """
    logger.debug("-------------------Start AutoClip------------------")
    smooth_sampler = MultiStepSampler()
    auto_clip = AutoClip(model, weight_bits=4, sample_function=smooth_sampler, n_grid=20, max_shrink=0.5)
    if create_only:
        return auto_clip
    calibration(predictor, ptq_dials, args, max_step=len(ptq_dials) - args.start_sample_step)
    auto_clip.auto_clip()

def apply_gptq(model, predictor, args, ptq_dials, create_only=False):
    """
    GPTQ
    """
    gptq = GPTQ(model, 
        quant_bits=4,
        weight_quant_method='abs_max_channel_wise',
        blocksize=128,
        percdamp=.2,
        groupsize=args.group_size,
        actorder=False,
    )
    if create_only:
        return gptq
    calibration(predictor, ptq_dials, args)
    gptq.fasterquant()

def apply_moe_shared_scale(model, predictor, args, ptq_dials, create_only=False):
    """
    GPTQ
    """
    moe_sharedscale = moe_shared_scale(model, 
        quant_bits=8,
        quant_method='abs_max',
    )
    if create_only:
        return moe_sharedscale
    calibration(predictor, ptq_dials, args)
    moe_sharedscale.search_best_scale()
 
def apply_analysis(model, predictor, args, ptq_dials):
    """
    Calcualte quant error for each layer
    Return a list [skip_layer_name]
    """
    logger.debug("-------------------Start Analysis------------------")
    analysis_loss_dict = {}
    skip_list_analysis = []
    for cur_name, cur_layer in model.named_sublayers():
        if type(cur_layer) in [ColumnParallelLinear, RowParallelLinear, paddle.nn.Linear]:
            parent_layer, sub_name = find_parent_layer_and_sub_name(model, cur_name)
            cur_quant_layer = LayerWiseQuantError(cur_layer)
            setattr(parent_layer, sub_name, cur_quant_layer)
 
    for idx in range(0, len(ptq_dials), args.batch_size):
        batch_dials = ptq_dials[idx : idx + args.batch_size]
        predictor.predict_ptq(batch_dials)
        if idx % 10 == 0:
            logger.debug(f"Sample Error Step: {idx}")
 
    for cur_name, analysis_layer in model.named_sublayers():
        if type(analysis_layer) == LayerWiseQuantError:
            loss = paddle.to_tensor(analysis_layer.losses, dtype="float32").mean()
            parent_layer, sub_name = find_parent_layer_and_sub_name(model, cur_name)
            setattr(parent_layer, sub_name, analysis_layer.layer)
            analysis_loss_dict[analysis_layer.layer.full_name()] = float(loss)
            del analysis_layer
 
    ranklist = sorted(analysis_loss_dict, key=analysis_loss_dict.get, reverse=True)
 
    for i, name in enumerate(ranklist):
        logger.debug(f"layer name: {name}, loss: {analysis_loss_dict[name]}")
        if analysis_loss_dict[name] > 5:
            skip_list_analysis.append(name)
    logger.debug(f"skip length: {len(skip_list_analysis)}, skip list: {skip_list_analysis}")
    return skip_list_analysis
 
 
def load_quant_model(model, args, ptq_dials, skip_list_analysis):
    """
    Load quantized model and its scales
    """
    activation, weight, cachekv, q_config = prepare_qconfig(args)
    for cur_name, cur_layer in model.named_sublayers():
        if "out_linear" in cur_name:
            continue
        if cur_layer.full_name() in skip_list_analysis:
            logger.debug(f"skip: {cur_name}, {cur_layer.full_name()}")
            continue
        if type(cur_layer) in [ColumnParallelLinear, RowParallelLinear, paddle.nn.Linear]:
            q_config.add_name_config([cur_layer.full_name()], activation=activation, weight=weight)
        if type(cur_layer) in [FuncWrapper]:
            # set both act and weight for attention, actually act-k and act-v are quantized
            q_config.add_name_config([cur_layer.full_name()], weight=cachekv[0], activation=cachekv[1])
        
    ptq = PTQ(q_config)
    model = ptq.quantize(model, inplace=True)
 
    logger.info("Load quant model...")
    rank = dist.get_rank()
    nranks = dist.get_world_size()
    if activation is not None:
        with open(f"{args.load_quant_path}/act_scales_{rank}.json") as outfile:
            act_scales = json.load(outfile)
    else:
        act_scales = {}
    # if 'C8' in args.quant_type or 'C4' in args.quant_type:
    #     with open(f"{args.load_quant_path}/cachekv_scales_{rank}.json") as outfile:
    #         cachekv_scales = json.load(outfile)
    # else:
    cachekv_scales = {}
    with open(f"{args.load_quant_path}/weight_scales_{rank}.json") as outfile:
        weight_scales = json.load(outfile)
 
    for cur_name, cur_layer in model.named_sublayers():
        if 'layer.' in cur_name:
            cur_name = cur_name.replace('layer.', '')
        if hasattr(cur_layer, 'scales'):
            if type(cur_layer) in [AbsMaxChannelWiseWeightObserverLayer, GroupWiseWeightObserverLayer]:
                cur_layer._scale = paddle.to_tensor(weight_scales[cur_name], dtype=args.dtype)
            if type(cur_layer) in [AbsmaxObserverLayer, TokenQuantileObserverLayer]:
                assert activation is not None, "AbsmaxObserverLayer must set observer"
                cur_layer._scale = paddle.to_tensor(act_scales[cur_name], dtype=args.dtype)
            if type(cur_layer) in [AvgHeadwiseObserverLayer, KCacheChannelWiseObserverLayer]:
                cur_name = cur_name.replace('attn_func.activation_quanter_v', 'cachev_matmul.activation_quanter')
                cur_name = cur_name.replace('attn_func.activation_quanter_k', 'cachek_matmul.activation_quanter')
                cur_layer._scale = paddle.to_tensor(cachekv_scales[cur_name], dtype=args.dtype)
                if cur_name + '.zero_point' in cachekv_scales:
                    cur_layer._zero_point = paddle.to_tensor(cachekv_scales[cur_name + '.zero_point'], dtype=args.dtype)
    model = ptq.convert(model, inplace=True)
    if nranks == 1:
        model_path = os.path.join(args.load_quant_path, "model_state.pdparams")
        model_dict = load_sharded_checkpoint(args.load_quant_path, return_numpy=True)
    else:
        model_path = os.path.join(args.load_quant_path, f"model_state.tp0{rank}.pdparams")
        model_dict = paddle.load(model_path, return_numpy=True)
    logger.info(f"Load model checkpoint from {model_path}")
    cur_model_dict = model.state_dict().keys()
    for key in cur_model_dict:
        if 'layer.' in key and 'scale' not in key:
            saved_key = key.replace('layer.', '')
            if saved_key in model_dict:
                model_dict[key] = np.array(model_dict[saved_key], dtype=np.uint16)
        # hard code here
        if 'scale' in key or 'scales' in key:
            scales_key = '.'.join(key.split('.')[:-2])
            new_key = key
            if 'layer.' in scales_key:
                scales_key = scales_key.replace('layer.', '')
                new_key = key.replace('layer.', '')
            if 'activation_quanter' in key:
                new_key = new_key.replace('cachev_matmul.activation_quanter', 'attn_func.activation_quanter_v')
                new_key = new_key.replace('cachek_matmul.activation_quanter', 'attn_func.activation_quanter_k')
                model_dict[key] = np.array(model_dict[new_key], dtype=np.float32)
            if 'weight_quanter' in key:
                model_dict[key] = np.array(weight_scales[scales_key], dtype=np.float32)
            else:
                model_dict[key] = np.array(model_dict[key], dtype=np.float32)
        if 'zero_point' in key:
            model_dict[key] = np.array(model.state_dict()[key], dtype=np.float32)
    model.set_dict(model_dict) 
 

def apply_ptq(model, predictor, args, ptq_dials, skip_list_analysis):
    """
    PTQ calibration process and save quantized model
    """
    # logger.info("-------------------GPTQ start------------------")

    # gptq = GPTQ(model, 
    #     quant_bits=4,
    #     weight_quant_method='abs_max_channel_wise',
    #     blocksize=128,
    #     percdamp=.1,
    #     actorder=True
    # )

    # calibration(predictor, ptq_dials, args,max_step=5)
    # gptq.fasterquant()
    # logger.info("-------------------GPTQ Done------------------")
    dp_degree = args.data_parallel_degree
    try:
        hcg = fleet.get_hybrid_communicate_group()
        rank = hcg.get_model_parallel_rank()
        nranks = hcg.get_model_parallel_world_size()
        dp_id = hcg.get_data_parallel_rank()
    except:
        rank = dist.get_rank()
        nranks = dist.get_world_size()
        dp_id = 0
    logger.info("-------------------Start PTQ------------------")
    activation, weight, cachekv, q_config = prepare_qconfig(args)
    weight_4bit = AbsMaxChannelWiseWeightObserver(quant_bits=4)
    for cur_name, cur_layer in model.named_sublayers():
        if "out_linear" in cur_name:
            logger.debug(f"skip {cur_name} {cur_layer.full_name()}")
            continue
        if cur_layer.full_name() in skip_list_analysis:
            logger.debug(f"skip {cur_name} {cur_layer.full_name()}")
            continue
        # if "experts" in cur_name:
        #     q_config.add_name_config([cur_layer.full_name()], activation=activation, weight=weight_4bit)
        #     logger.debug(f"weight_w_4bit: {cur_name} {cur_layer.full_name()}")
        #     continue
        if type(cur_layer) in [ColumnParallelLinear, RowParallelLinear, paddle.nn.Linear]:
            # if ('linear2' in cur_layer.full_name() or 'linear2' in cur_name) and args.token_clip:
            #     logger.debug(f'token clip layer: {cur_layer.full_name()}, {cur_name}')
            #     activation1 = TokenQuantileObserver(quant_bits=8, percentile=1.0)
            #     q_config.add_name_config([cur_layer.full_name()], activation=activation1, weight=weight)
            # if "experts" in cur_name:
            #     q_config.add_name_config([cur_layer.full_name()], activation=activation, weight=weight_4bit)
            #     logger.debug(f"weight_w_4bit_using_gptq: {cur_name} {cur_layer.full_name()}")
            # else:
                # q_config.add_name_config([cur_layer.full_name()], activation=activation, weight=weight)
                q_config.add_name_config([cur_layer.full_name()], activation=None, weight=weight)
        if type(cur_layer) in [FuncWrapper]:
            # set both act and weight for attention, actually act-k and act-v are quantized
            q_config.add_name_config([cur_layer.full_name()], weight=cachekv[0], activation=cachekv[1])
 
    ptq = PTQ(q_config)
    model = ptq.quantize(model, inplace=True)
    args.token_clip=False

    if args.token_clip:
        apply_token_wise_clipping(model, predictor, args, None, ptq_dials, max_step=16)

    for idx in range(0, len(ptq_dials), args.batch_size):
        batch_dials = ptq_dials[idx : idx + args.batch_size]
        predictor.predict_ptq(batch_dials)
        if idx % 10 == 0:
            logger.info(f"Sample PTQ Step: {idx}")
        if idx >= 128:
            break

    best_quant_policies = None
    if args.abq:
        best_quant_policies = apply_abq(model, predictor, args, ptq_dials, max_step=args.abq_step)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
 
    # act_scales, weight_scales, cachekv_scales = get_scales(
    #     model, {}, {}, {}, dp_degree, nranks, rank, best_quant_policies)
    # save_scales(args, act_scales, weight_scales, cachekv_scales, mp_id=rank, dp_id=dp_id)
    
    model = ptq.convert(model, inplace=True)

    act_scales, weight_scales, cachekv_scales = get_scales(
        model, {}, {}, {}, dp_degree, nranks, rank, best_quant_policies)
    save_scales(args, act_scales, weight_scales, cachekv_scales, mp_id=rank, dp_id=dp_id)

    if nranks == 1:
        model_path = os.path.join(args.save_path, "model_state.pdparams")
    else:
        model_path = os.path.join(args.save_path, f"model_state.tp0{rank}.pdparams")
    state_dict = model.state_dict()
    save_quant_model(state_dict, model_path, dp_id=dp_id)
    logger.info(f"Save quant model to {args.save_path}")
    logger.info("-------------------PTQ Done------------------")
 
def apply_abq(model, predictor, args, ptq_dials, max_step=16):
    """
    ABQ process, search best quantization policy
    """
    max_step = min(max_step, int(len(ptq_dials) // args.batch_size))
    abq = AdaptiveBaggingQuant(args, model, max_step)
    for cur_name, cur_layer in model.named_sublayers():
        if type(cur_layer) == QuantizedCustomAttentionLayer:
            cur_layer.quant_info = abq.quant_info
            cur_layer.enable_fake_quant = True
    with paddle.no_grad():
        for idx in range(0, len(ptq_dials), args.batch_size):
            batch_dials = ptq_dials[idx : idx + args.batch_size]
            input_map = predictor.preprocess(batch_dials, extra_infos=None, fast_ptq_sampling=True)
            if input_map is None:
                continue
            output = model(**input_map)
            if idx % 10 == 0:
                logger.debug(f"ABQ Sample Step: {idx}")
            if idx >= max_step:
                break
        abq.search()
    logger.info("========cachekv search done========")
    return abq.best_quant_policies

 
def apply_layerwise_quant(model, predictor, args, ptq_dials, skip_list_analysis, layer_num=4):
    """
    Quant model layer by layer
    For each layer, complete calibration process will be repeated
    """
    logger.debug("-------------------Start Layerwise PTQ------------------")
    dp_degree = args.data_parallel_degree
    try:
        hcg = fleet.get_hybrid_communicate_group()
        rank = hcg.get_model_parallel_rank()
        nranks = hcg.get_model_parallel_world_size()
        dp_id = hcg.get_data_parallel_rank()
    except:
        rank = dist.get_rank()
        nranks = dist.get_world_size()
        dp_id = 0
    all_layers = []
    for _, cur_layer in model.named_sublayers():
        if type(cur_layer) in [ColumnParallelLinear, RowParallelLinear, paddle.nn.Linear]:
            all_layers.append(cur_layer.full_name())
        if type(cur_layer) in [FuncWrapper]:
            all_layers.append(cur_layer.full_name())
    activation, weight, cachekv, q_config = prepare_qconfig(args)
 
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    act_scales = {}
    weight_scales = {}
    cachekv_scales = {}
    for i in range(0, len(all_layers), layer_num):
        cur_layer_name = all_layers[i : i + layer_num]
        cachekv_name = []
        for n in cur_layer_name:
            if 'cache_kv' in n:
                cachekv_name.append(n)
        for n in cachekv_name:
            cur_layer_name.remove(n)
        for skip_layer in skip_list_analysis:
            if skip_layer in cur_layer_name:
                cur_layer_name.remove(skip_layer)
 
        logger.debug(f"Quantizing step {i} / {len(all_layers)}")
        logger.debug(f"{cur_layer_name} {cachekv_name}")
        if not cur_layer_name:
            continue
        
        q_config.add_name_config(cur_layer_name, activation=activation, weight=weight)
        # set both act and weight for attention, actually act-k and act-v are quantized
        q_config.add_name_config(cachekv_name, weight=cachekv[0], activation=cachekv[1])
        ptq = PTQ(q_config)
        model = ptq.quantize(model, inplace=True)
        for idx in range(0, len(ptq_dials), args.batch_size):
            batch_dials = ptq_dials[idx : idx + args.batch_size]
            predictor.predict_ptq(batch_dials)
            if idx % 10 == 0:
                logger.info(f"Sample PTQ Step: {idx}")
 
        act_scales, weight_scales, cachekv_scales = get_scales(model, act_scales, weight_scales, \
                cachekv_scales, dp_degree, nranks, rank)
        model = ptq.convert(model, inplace=True)

    
 
    save_scales(args, act_scales, weight_scales, cachekv_scales, rank, dp_id)
 
    rank = dist.get_rank()
    nranks = dist.get_world_size()
    if nranks == 1:
        model_path = os.path.join(args.save_path, "model_state.pdparams")
    else:
        model_path = os.path.join(args.save_path, f"model_state.tp0{rank}.pdparams")
    save_quant_model(model.state_dict(), model_path, dp_id=dp_id)
    logger.info(f"Save quant model to {args.save_path}")
    logger.debug("-------------------Layerwise PTQ Done------------------")
 

def create_qat_model(model, args, dtype):
    """
    Create QAT model
    """
    q_config = QuantConfig(activation=None, weight=None)
    q_config.add_qat_layer_mapping(LoRALinear, QuantedLoRALinear)
    q_config.add_qat_layer_mapping(ColumnParallelLinear, QuantizedColumnParallelLinear)
    q_config.add_qat_layer_mapping(RowParallelLinear, QuantizedRowParallelLinear)
    if args is None or args.quant_type == "W8A8":
        activation = PACTQuanter(quanter=FakeQuanterWithAbsMaxObserver(), init_value=20, dtype=dtype)
        weight = FakeQuanterChannelWiseAbsMaxObserver(bit_length=8, dtype=dtype)
    elif args.quant_type in ["WINT4", "W4A16"]:
        activation = None
        weight = FakeQuanterChannelWiseAbsMaxObserver(bit_length=4, dtype=dtype)
    elif args.quant_type in ["WINT8", "W8A16"]:
        activation = None
        weight = FakeQuanterChannelWiseAbsMaxObserver(bit_length=8, dtype=dtype)
    elif args.quant_type == "W4A8":
        activation = PACTQuanter(quanter=FakeQuanterWithAbsMaxObserver(), init_value=20, dtype=dtype)
        weight = FakeQuanterChannelWiseAbsMaxObserver(bit_length=4, dtype=dtype)
    else:
        raise ValueError("quant_type should be one of ['W8A8', 'WINT4', 'WINT8', 'W4A8', 'W4A16', 'W8A16']")
    for cur_name, cur_layer in model.named_sublayers():
        if "out_linear" in cur_name:
            continue
        if type(cur_layer) in [ColumnParallelLinear, RowParallelLinear, paddle.nn.Linear]:
            q_config.add_name_config([cur_layer.full_name()], activation=activation, weight=weight)
 
    qat = QAT(q_config)
    model = qat.quantize(model, inplace=True)
    return model
 