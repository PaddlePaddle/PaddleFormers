DeepSeek-V3 全参数微调实践

近期我们组织并完成了对 DeepSeek-V3（671B）模型的全参数微调实验。本次实践旨在验证大规模模型在特定业务场景下的可控性与可落地性，并探索全参数微调在性能优化、训练效率及资源调度等方面的关键技术路径。以下是我们的整体解决方案，包含训练的完整代码，以及实践过程中的经验教训。

#### 项目亮点
* 参考 huggingface tranformers，paddlenlp 等训练框架，补齐训练过程中的全部逻辑，包含 Multi-Token Prediction、MOE 训练组件，完成 modeling 的编写。
* 实现了基于 Sharding 并行、PP 并行、SP 并行、TP 并行、EP 并行混合并行方案，添加了 subbatch， offload optimizer 等优化，支持 DeepSeek-V3的全量微调、16机下可实现支持128K 长文全量微调训练。
* 总结了模型训练过程中的经验教训及解决方案。

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

推荐使用 hopper 架构 GPU，以能够支持 EP 并行的训练方式

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
{
    "model_name_or_path": "/root/huggingface_model/huggingface/deepseek-ai/DeepSeek-V3-bf16/",
    "dataset_name_or_path": "/root/data/",
    "output_dir": "/root/checkpoints/sft_ckpts",
    "train_dataset_path": "/root/data//train.json",
    "train_dataset_prob": "1.0",
    "train_dataset_type": "erniekit",
    "eval_dataset_path": "/root/data//dev.json",
    "eval_dataset_prob": "1.0",
    "eval_dataset_type": "erniekit",
    "packing": true,
    "max_seq_len": 4096,
    "hybrid_parallel_topo_order": "sharding_first",
    "aux_loss_alpha": 0.0001,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 128,
    "per_device_eval_batch_size": 1,
    "eval_accumulation_steps": 1,
    "learning_rate": 2.2e-05,
    "warmup_steps": 30,
    "logging_steps": 1,
    "evaluation_strategy": "no",
    "save_strategy": "no",
    "bf16": true,
    "amp_master_grad": true,
    "fp16_opt_level": "O2",
    "do_train": true,
    "do_eval": false,
    "disable_tqdm": true,
    "use_expert_parallel": true,
    "expert_parallel_degree": 16,
    "continue_training": true,
    "pipeline_parallel_config": "enable_delay_scale_loss disable_partial_send_recv disable_batch_p2p_comm",
    "tensor_parallel_config": "enable_delay_scale_loss",
    "load_best_model_at_end": false,
    "eval_with_do_generation": false,
    "metric_for_best_model": "loss",
    "recompute": true,
    "recompute_use_reentrant": true,
    "recompute_granularity": "full",
    "save_total_limit": 1,
    "tensor_parallel_degree": 1,
    "sequence_parallel": false,
    "pipeline_parallel_degree": 8,
    "sharding_parallel_degree": 16,
    "sharding": "stage1",
    "unified_checkpoint": true,
    "unified_checkpoint_config": "ignore_merge_optimizer",
    "save_steps": 99,
    "use_flash_attention": true,
    "flash_mask": false,
    "using_fake_gate": false,
    "pre_alloc_memory": 60,
    "tensorwise_offload_optimizer": true,
    "use_fused_rms_norm": true,
    "max_steps": 100,
    "sharding_parallel_config": "split_param",
    "tensor_parallel_output": true,
    "num_nextn_predict_layers": 1,
    "convert_from_hf": true,
    "use_attn_mask_startend_row_indices": true
  }
```
##### 128K 配置
```
{
    "model_name_or_path": "/root/huggingface_model/huggingface/deepseek-ai/DeepSeek-V3-bf16/",
    "dataset_name_or_path": "/root/data/",
    "output_dir": "/root/checkpoints/sft_ckpts",
    "train_dataset_path": "/root/data//train.json",
    "train_dataset_prob": "1.0",
    "train_dataset_type": "erniekit",
    "eval_dataset_path": "/root/data//dev.json",
    "eval_dataset_prob": "1.0",
    "eval_dataset_type": "erniekit",
    "packing": true,
    "max_seq_len": 131072,
    "hybrid_parallel_topo_order": "sharding_first",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "per_device_eval_batch_size": 1,
    "eval_accumulation_steps": 1,
    "learning_rate": 7e-06,
    "warmup_steps": 30,
    "max_grad_norm": 1.0,
    "logging_steps": 1,
    "aux_loss_alpha": 0.0001,
    "evaluation_strategy": "no",
    "save_strategy": "steps",
    "bf16": true,
    "amp_master_grad": true,
    "fp16_opt_level": "O2",
    "do_train": true,
    "do_eval": false,
    "disable_tqdm": true,
    "use_expert_parallel": true,
    "expert_parallel_degree": 16,
    "continue_training": false,
    "pipeline_parallel_config": "enable_delay_scale_loss disable_partial_send_recv disable_batch_p2p_comm",
    "tensor_parallel_config": "enable_delay_scale_loss",
    "load_best_model_at_end": false,
    "eval_with_do_generation": false,
    "metric_for_best_model": "loss",
    "recompute": true,
    "recompute_use_reentrant": true,
    "recompute_granularity": "full",
    "save_total_limit": 1,
    "tensor_parallel_degree": 8,
    "sequence_parallel": true,
    "pipeline_parallel_degree": 8,
    "sharding_parallel_degree": 2,
    "sharding": "stage1",
    "unified_checkpoint": true,
    "unified_checkpoint_config": "ignore_merge_optimizer",
    "save_steps": 120,
    "use_flash_attention": true,
    "flash_mask": true,
    "using_fake_gate": false,
    "pre_alloc_memory": 60,
    "tensorwise_offload_optimizer": true,
    "use_fused_rms_norm": true,
    "max_steps": 100,
    "sharding_parallel_config": "split_param",
    "tensor_parallel_output": true,
    "num_nextn_predict_layers": 1,
    "convert_from_hf": true,
    "use_attn_mask_startend_row_indices": true,
    "moe_subbatch_token_num": 1024
  }


```
##### 启动脚本
```
source /root/work/formers_venv/bin/activate
export NCCL_IB_GID_INDEX=3
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_IB_TRAFFIC_CLASS=162
export NVSHMEM_BOOTSTRAP=UID

unset NVSHMEM_HCA_LIST
unset NVSHMEM_ENABLE_NIC_PE_MAPPING


python3.10 -m paddle.distributed.launch \
    --log_dir output-ep/paddle_distributed_logs \
    --run_mode=collective \
    run_finetune.py \
    /root/work/PaddleFormers/examples/config/deepseek/xxxx.json > run-ep.log 2>&1 &


```
```
# 推荐使用mpirun进行多机启动
mpirun bash run.sh
```
#### 实验效果
* 在4K 长度的上下文场景下，100个 step ，loss 收敛效果

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=6d0e08d571724f3dae1d17582d4d60d5&docGuid=NkKYhoYXikuIv- "")
* 在128K 长度的上下文场景下，100个 step，loss 收敛效果

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=e72af01ac9664478b36b378b0d9bd833&docGuid=NkKYhoYXikuIv- "")








#### 实验总结
* 在超大参数场景下，优化器状态往往无法全部驻留 GPU 显存，因此需要 offload ，以显存换空间，确保训练可以继续跑起来。
* 在长序列场景下，次前向的激活峰值随 token 数迅速增长，很快撑爆显存，因此需要 subbatch，用计算换空间，确保训练可以继续跑起来。
* 在 MOE 场景下，专家间的负载不均也可能导致训练因 OOM 终止，对此 auxloss 及 auxloss-free 机制十分重要，以下是实验过程中发现的易错点：
    * e_score_correction_bias 仅在计算 gate 时使用，不要传递到后续的 ffn


> Note that the bias term is only used for routing. The gating value, which will be multiplied with the FFN output, is still derived from the original affinity score 𝑠𝑖,𝑡.
    * auxloss 计算再 SP、subbatch 等场景下要注意 seq_len 的计算取值
    * huggingface 中上传的部分 config 需要自行在不同场景下调整，例如 aux_loss_alpha

> For auxiliary-loss-free load balancing, we set the bias update speed 𝛾 to 0.001 for the first 14.3T tokens, and to 0.0 for the remaining 500B tokens. For the balance loss, we set 𝛼 to 0.0001, just to avoid extreme imbalance within any single sequence.
