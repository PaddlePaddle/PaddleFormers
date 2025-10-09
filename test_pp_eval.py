import os
import json
import random
import numpy as np
from safetensors import safe_open

import paddle
from paddle.distributed import fleet
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed as dist
from paddle.io import Dataset, BatchSampler, DataLoader
paddle.set_printoptions(linewidth=160)

from transformers import AutoTokenizer
from paddleformers.transformers import AutoConfig
from paddleformers.transformers.qwen3_next import Qwen3NextForCausalLMPipe

strategy = fleet.DistributedStrategy()
model_parallel_size = 1
data_parallel_size = 1
pipeline_parallel_size = 4
batch_size = pipeline_parallel_size * 2  # 1f1b will fail if too small
strategy.hybrid_configs = {
    "dp_degree": data_parallel_size,
    "mp_degree": model_parallel_size,
    "pp_degree": pipeline_parallel_size,
}
strategy.pipeline_configs = {
    "accumulate_steps": batch_size,
    "micro_batch_size": 1,
}

fleet.init(is_collective=True, strategy=strategy)


def set_random_seed(seed, dp_id, rank_id):
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + dp_id + rank_id)
    print("seed: ", seed)
    print("rank_id: ", rank_id)
    print("dp_id: ", dp_id)


hcg = fleet.get_hybrid_communicate_group()
world_size = hcg.get_model_parallel_world_size()
dp_id = hcg.get_data_parallel_rank()
pp_id = hcg.get_stage_id()
rank_id = dist.get_rank()
set_random_seed(1024, dp_id, rank_id)

model_path = "/home/work/Qwen/Qwen3-Next-80B-A3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

input_ids = [tokenizer("今天是周五,明天是周六,后天是周日,那么大后天是周几?")["input_ids"]]
input_ids = paddle.to_tensor(input_ids)

query_length = input_ids.shape[-1]
dtype = paddle.bfloat16
input_ids = input_ids.expand([batch_size, -1])  # repeat input for batch_size times
causal_mask = paddle.zeros([batch_size, 1, query_length, query_length], dtype)  # placeholder only

config = AutoConfig.from_pretrained(model_path)

origin_dtype = paddle.get_default_dtype()
paddle.set_default_dtype(dtype)
model = Qwen3NextForCausalLMPipe(config, num_stages=pipeline_parallel_size, topology=hcg._topo)
model = fleet.distributed_model(model)
paddle.set_default_dtype(origin_dtype)

resolved_archive_file, resolved_sharded_files, sharded_metadata, is_sharded = (
    Qwen3NextForCausalLMPipe._resolve_model_file_path(model_path)
)

with open(resolved_archive_file) as f:
    weight_map = json.load(f)["weight_map"]

rear_weights = ["model.norm.weight", "lm_head.weight"]

for key, param in model.state_dict().items():
    _, pp_layer_idx, local_name = key.split('.', maxsplit=2)
    pp_layer_idx = int(pp_layer_idx)

    if pp_layer_idx == 0:
        weight_key = f"model.{local_name}"
    elif local_name == "weight":
        weight_key = rear_weights.pop(0)
    else:
        weight_key = f"model.layers.{pp_layer_idx - 1}.{local_name}"

    weight_file = weight_map[weight_key]
    # print('loading:', key, param.dtype, '->', weight_key, ':', weight_file)

    with safe_open(os.path.join(model_path, weight_file), framework="np") as f:
        tensor = f.get_tensor(weight_key)

    need_transpose = False
    for transpose_key in Qwen3NextForCausalLMPipe.transpose_weight_keys:
        if f'.{transpose_key}.' in key:
            need_transpose = True
            break
    if need_transpose:
        tensor = tensor.T

    assert list(tensor.shape) == list(param.shape), (
        f"param: {param.shape} {param.dtype}, "
        f"tensor: {tensor.shape} {tensor.dtype}, "
        f"transpose: {int(need_transpose)}"
    )

    tensor = paddle.to_tensor(tensor).to(param)
    param.copy_(tensor)

x = (input_ids, causal_mask)
logits = model.eval_batch([x, x])

if pp_id == pipeline_parallel_size - 1:
    logits = logits[0]
    print('logits:', logits)
    np.save("/work/paddle.logits.npy", logits.float().numpy())
else:
    print('done')

'''
cv 0,1,2,3
rm -rf log; python -m paddle.distributed.launch test_pp_eval.py
'''
