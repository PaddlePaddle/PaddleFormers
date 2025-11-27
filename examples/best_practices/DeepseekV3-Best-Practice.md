DeepSeek-V3 全参数微调实践

近期，我们成功组织并完成了 DeepSeek-V3（671B）模型的全参数微调实验。本次实践旨在验证超大规模模型在特定业务场景下的可控性与实际落地能力，同时系统探索全参数微调在性能优化、训练效率提升及资源调度等方面的关键技术路径。以下为我们构建的整体解决方案以及在实践过程中积累的经验与教训总结。

#### 项目亮点
* 参考 HuggingFace Transformers 等主流训练框架，补全了训练流程中的全部关键逻辑，包括 Multi-Token Prediction 与 MOE 训练组件，完成整体 modeling 组网的编写。
* 实现了融合 Sharding 并行、PP（流水线并行）、SP（序列并行）、TP（张量并行）与 EP（专家并行）的混合并行训练方案，并引入 subbatch 处理、优化器 offload 等优化技术，在16台机器环境下支持 DeepSeek-V3模型的全量微调，同时可处理128K 长度的长文本输入。

#### 快速开始
##### 环境配置
1. 安装 PaddleFormers

```
git clone https://github.com/PaddlePaddle/PaddleFormers.git
pip install -r requirements.txt
pip install -e .
```
2. 安装 PaddlePaddle

```
# cuda12.9
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu129/
#更多安装版本参考 https://www.paddlepaddle.org.cn/
```
3. 硬件配置

为充分发挥 EP（专家并行）训练优势，推荐使用基于 Hopper 架构的 GPU。

##### 数据准备
本次实验采用 ErnieKit 格式类数据集，以下是部分数据样例

```
{"src": "Give three tips for staying healthy.", "tgt": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."}
{"src": "What are the three primary colors?", "tgt": "The three primary colors are red, blue, and yellow."}
{"src": "Describe the structure of an atom.", "tgt": "An atom is made up of a nucleus, which contains protons and neutrons, surrounded by electrons that travel in orbits around the nucleus. The protons and neutrons have a positive charge, while the electrons have a negative charge, resulting in an overall neutral atom. The number of each particle determines the atomic number and the type of atom."}
{"src": "How can we reduce air pollution?", "tgt": "There are a number of ways to reduce air pollution, such as shifting to renewable energy sources, encouraging the use of public transportation, prohibiting the burning of fossil fuels, implementing policies to reduce emissions from industrial sources, and implementing vehicle emissions standards. Additionally, individuals can do their part to reduce air pollution by reducing car use, avoiding burning materials such as wood, and changing to energy efficient appliances."}
{"src": "Describe a time when you had to make a difficult decision.", "tgt": "I had to make a difficult decision when I was working as a project manager at a construction company. I was in charge of a project that needed to be completed by a certain date in order to meet the client’s expectations. However, due to unexpected delays, we were not able to meet the deadline and so I had to make a difficult decision. I decided to extend the deadline, but I had to stretch the team’s resources even further and increase the budget. Although it was a risky decision, I ultimately decided to go ahead with it to ensure that the project was completed on time and that the client’s expectations were met. The project was eventually successfully completed and this was seen as a testament to my leadership and decision-making abilities."}
{"src": "Identify the odd one out.Twitter, Instagram, Telegram", "tgt": "Telegram"}
```
##### 训练配置
##### 4K 配置
```
### data
train_dataset_type: erniekit
eval_dataset_type: erniekit
train_dataset_path: "/root/train.json" # 数据集存放路径
train_dataset_prob: "1.0"
eval_dataset_path: "/root/eval.json" # 数据集存放路径
eval_dataset_prob: "1.0"
max_seq_len: 4096
num_samples_each_epoch: 6000000
packing: true

### model
model_name_or_path: "/root/huggingface_model/DeepSeek-V3-bf16/" # 模型存放路径
convert_from_hf: true

### finetuning
# base
stage: SFT
fine_tuning: full
do_train: true
do_eval: false
per_device_eval_batch_size: 1
per_device_train_batch_size: 1
num_train_epochs: 1
num_nextn_predict_layers: 1
max_steps: 100
evaluation_strategy: steps
save_steps: 100
save_total_limit: 1
save_strategy: steps
logging_steps: 1
gradient_accumulation_steps: 16
logging_dir: ./vdl_log
output_dir: ./checkpoints/dsv3
disable_tqdm: true
eval_accumulation_steps: 1
load_best_model_at_end: false
eval_with_do_generation: false
metric_for_best_model: "loss"
hybrid_parallel_topo_order: "sharding_first"
unified_checkpoint: true
unified_checkpoint_config: "ignore_merge_optimizer"

# train
warmup_steps: 30
learning_rate: 2.2e-05
continue_training: true

# performance
tensor_parallel_degree: 1
sequence_parallel: false
pipeline_parallel_degree: 8
sharding_parallel_degree: 16
use_expert_parallel: true
expert_parallel_degree: 16
tensor_parallel_config: "enable_delay_scale_loss sync_param sync_grad"
pipeline_parallel_config: "enable_delay_scale_loss disable_partial_send_recv disable_batch_p2p_comm"
sharding_parallel_config: "split_param"
recompute: true
recompute_use_reentrant: true
recompute_granularity: "full"
sharding: stage1
bf16: true
amp_master_grad: true
fp16_opt_level: O2
use_flash_attention: true
use_attn_mask_startend_row_indices: true
using_fake_gate: false
pre_alloc_memory: 60
tensorwise_offload_optimizer: true
use_fused_rms_norm: true
moe_subbatch_token_num: 0


```
##### 32K 配置
```
### data
train_dataset_type: erniekit
eval_dataset_type: erniekit
train_dataset_path: "/root/train.json" # 数据集存放路径
train_dataset_prob: "1.0"
eval_dataset_path: "/root/eval.json" # 数据集存放路径
eval_dataset_prob: "1.0"
max_seq_len: 32768
num_samples_each_epoch: 6000000
packing: true

### model
model_name_or_path: "/root/huggingface_model/DeepSeek-V3-bf16/" # 模型存放路径
convert_from_hf: true

### finetuning
# base
stage: SFT
fine_tuning: full
do_train: true
do_eval: false
per_device_eval_batch_size: 1
per_device_train_batch_size: 1
num_train_epochs: 1
num_nextn_predict_layers: 1
max_steps: 100
evaluation_strategy: no
save_steps: 100
save_total_limit: 1
save_strategy: no
logging_steps: 1
gradient_accumulation_steps: 16
logging_dir: ./vdl_log
output_dir: ./checkpoints/dsv3
disable_tqdm: true
eval_accumulation_steps: 1
load_best_model_at_end: false
eval_with_do_generation: false
metric_for_best_model: "loss"
hybrid_parallel_topo_order: "sharding_first"
unified_checkpoint: true
unified_checkpoint_config: "ignore_merge_optimizer"

# train
warmup_steps: 30
learning_rate: 7e-06
continue_training: true

# performance
tensor_parallel_degree: 8
sequence_parallel: true
pipeline_parallel_degree: 8
sharding_parallel_degree: 2
use_expert_parallel: true
expert_parallel_degree: 16
tensor_parallel_config: "enable_delay_scale_loss sync_param sync_grad"
pipeline_parallel_config: "enable_delay_scale_loss disable_partial_send_recv disable_batch_p2p_comm"
sharding_parallel_config: "split_param"
recompute: true
recompute_use_reentrant: true
recompute_granularity: "full"
sharding: stage1
bf16: true
amp_master_grad: true
fp16_opt_level: O2
use_flash_attention: true
use_attn_mask_startend_row_indices: true
using_fake_gate: false
pre_alloc_memory: 60
tensorwise_offload_optimizer: true
use_fused_rms_norm: true
moe_subbatch_token_num: 0
```
##### 128K 配置
```
### data
train_dataset_type: erniekit
eval_dataset_type: erniekit
train_dataset_path: "/root/train.json" # 数据集存放路径
train_dataset_prob: "1.0"
eval_dataset_path: "/root/eval.json" # 数据集存放路径
eval_dataset_prob: "1.0"
max_seq_len: 131072
num_samples_each_epoch: 6000000
packing: true

### model
model_name_or_path: "/root/huggingface_model/DeepSeek-V3-bf16/" # 模型存放路径
convert_from_hf: true

### finetuning
# base
stage: SFT
fine_tuning: full
do_train: true
do_eval: false
per_device_eval_batch_size: 1
per_device_train_batch_size: 1
num_train_epochs: 1
num_nextn_predict_layers: 1
max_steps: 100
evaluation_strategy: no
save_steps: 100
save_total_limit: 1
save_strategy: no
logging_steps: 1
gradient_accumulation_steps: 16
logging_dir: ./vdl_log
output_dir: ./checkpoints/dsv3
disable_tqdm: true
eval_accumulation_steps: 1
load_best_model_at_end: false
eval_with_do_generation: false
metric_for_best_model: "loss"
hybrid_parallel_topo_order: "sharding_first"
unified_checkpoint: true
unified_checkpoint_config: "ignore_merge_optimizer"

# train
warmup_steps: 30
learning_rate: 7e-06
continue_training: true

# performance
tensor_parallel_degree: 8
sequence_parallel: true
pipeline_parallel_degree: 8
sharding_parallel_degree: 2
use_expert_parallel: true
expert_parallel_degree: 16
tensor_parallel_config: "enable_delay_scale_loss sync_param sync_grad"
pipeline_parallel_config: "enable_delay_scale_loss disable_partial_send_recv disable_batch_p2p_comm"
sharding_parallel_config: "split_param"
recompute: true
recompute_use_reentrant: true
recompute_granularity: "full"
sharding: stage1
bf16: true
amp_master_grad: true
fp16_opt_level: O2
use_flash_attention: true
use_attn_mask_startend_row_indices: true
using_fake_gate: false
pre_alloc_memory: 60
tensorwise_offload_optimizer: true
use_fused_rms_norm: true
moe_subbatch_token_num: 1024
```
##### 启动脚本
```
source /root/formers_venv/bin/activate #python环境存放位置
export NCCL_IB_GID_INDEX=3
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_IB_TRAFFIC_CLASS=162
export NVSHMEM_BOOTSTRAP=UID

unset NVSHMEM_HCA_LIST
unset NVSHMEM_ENABLE_NIC_PE_MAPPING


NNODES={num_nodes} MASTER_ADDR={your_master_addr} MASTER_PORT={your_master_port} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
paddleformers-cli train /root/PaddleFormers/examples/config/dsv3_config.yaml #配置存放路径
```
```
# 推荐使用mpirun进行多机启动
mpirun bash run.sh
```
#### 实验效果
##### 实验配置
|方案|机器数|seq_len|sharding|tp|sp|pp|ep|tokens/s/card|数据来源|
|-|-|-|-|-|-|-|-|-|-|
|Paddle|16机|4K|16|1|fasle|8|16|203|自测|
||16机|32K|2|8|true|8|16|182|自测|
||16机|128K|2|8|true|8|16|124|自测|

##### 收敛效果：
* 在4K 长度的上下文场景下，100个 step ，loss 收敛效果

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=6d0e08d571724f3dae1d17582d4d60d5&docGuid=NkKYhoYXikuIv- "")
* 在32K 长度的上下文场景下，100个 step， loss 收敛效果

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=5b1fdfc029b24dfaa2b2742c51be49c2&docGuid=NkKYhoYXikuIv- "")
* 在128K 长度的上下文场景下，100个 step，loss 收敛效果

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=1e4942bd29bc4185b75da7488b1e9172&docGuid=NkKYhoYXikuIv- "")


#### 实验总结
* 在大规模参数场景下，优化器状态往往无法完全驻留于 GPU 显存，因此需采用 Offload 技术，以内存空间换取显存容量，确保训练任务持续执行。
* 面对长序列输入时，前向计算过程中的激活值峰值随 token 数量急剧上升，极易耗尽显存。此时可引入 Subbatch 方法，通过分段计算以时间换空间，保障训练流程的稳定推进。
* 在 MoE 模型中，专家间负载不均衡也可能引发 OOM 错误。为此，合理引入 AuxLoss 及其无辅助损失机制至关重要。以下是实验过程中总结的关键注意事项：
    * Gate 计算隔离：e_score_correction_bias 应仅用于门控权重计算，避免传递至后续 FFN 模块。
    * AuxLoss 计算适配：在 SP 或 Subbatch 等并行策略下，需注意 seq_len 的实际取值，确保损失计算正确。
    * 配置调整：Hugging Face 所提供的部分配置（如 aux_loss_alpha）需结合具体训练场景进行针对性调优。
