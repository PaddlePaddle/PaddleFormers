# 1. 基础配置与训练控制

```shell
  --output_dir
                        模型预测结果和检查点的输出目录。(`str`, 必须)

  --overwrite_output_dir
                        是否覆盖输出目录的内容。(`bool`, 可选, 默认为 `False`)

  --do_train / --do_eval / --do_predict / --do_export
                        是否进行 训练/评估/预测/导出 任务。(`bool`, 可选)

  --seed
                        设置随机种子，确保可复现性。(`int`, 可选, 默认为 42)

  --resume_from_checkpoint
                        指向一个有效的检查点目录路径，用于恢复训练。若不指定，将从头开始训练。(`str`, 可选)

  --evaluation_strategy
                        评估策略。支持 `no` (不评估), `steps` (按步数评估), `epoch` (按周期评估)。
                        默认值为 `no`。(`str` 或 `IntervalStrategy`, 可选)

  --ignore_data_skip
                        恢复训练时，是否跳过通过 DataLoader 快速过掉已训练数据这一步。
                        如果设置为 `True`，训练会立即开始，但可能会重复训练部分数据。默认为 `False`。(`bool`, 可选)

  --log_on_each_node
                        在多机分布式训练中，是否在每个节点上都打印日志。
                        如果为 `False`，则仅在主节点打印。默认为 `True`。(`bool`, 可选)

  --to_static
                        是否启用静态图模式（`jit.to_static` 或 `distributed.to_static`）进行训练。
                        静态图模式通常能带来更好的性能和显存优化。(`bool`, 可选, 默认为 `False`)

  --per_device_train_batch_size
                        用于训练的每个 GPU/CPU 的 batch 大小。(`int`, 可选, 默认为 8)

  --per_device_eval_batch_size
                        用于评估的每个 GPU/CPU 的 batch 大小。(`int`, 可选, 默认为 8)

  --save_on_each_node
                        在多机分布式训练中，是否在每个节点上都保存模型和检查点。
                        默认为 `False`（仅在主节点保存）。在非共享存储环境（如无 NAS）中，
                        必须设置为 `True` 以确保每个节点都有权重备份。(`bool`, 可选)

  --dataloader_num_workers
                        用于数据加载的子进程数量。默认为 0（在主进程加载）。
                        建议设置为 > 0（如 4 或 8）以加速数据预处理。(`int`, 可选)

  --dataloader_drop_last
                        是否丢弃最后一个不完整的 Batch（当数据集大小不能被 Batch Size 整除时）。(`bool`, 可选, 默认为 `False`)

  --distributed_dataloader
                        是否使用分布式数据加载器。在混合并行场景下（特别是使用了 Tensor Parallel 或 Pipeline Parallel 时），
                        可能需要开启此选项以确保数据正确切分。(`bool`, 可选, 默认为 `False`)

  --logging_dir
                        VisualDL、TensorBoard 等日志的保存目录。
                        默认为 `output_dir/runs/CURRENT_DATETIME_HOSTNAME`。(`str`, 可选)

  --run_name
                        运行的描述符，通常用于日志记录（如 WandB 的 group name）。
                        默认为 `output_dir`。(`str`, 可选)

  --dataloader_shuffle
                        是否对训练数据加载器进行 Shuffle（打乱）。
                        默认为 `True`。(`bool`, 可选)

  --lazy_data_processing
                        是否启用懒加载数据处理，用于节省初始化内存。
                        默认为 `True`。(`bool`, 可选)

  --pad_token_id
                        Padding Token 的 ID，在统计 `trained_tokens` 时会忽略此 ID。
                        默认为 0。(`int`, 可选)

  --convert_from_hf
                        是否从 HuggingFace safetensors 格式加载模型。
                        默认为 `True`。(`bool`, 可选)

  --disable_tqdm
                        是否禁用 tqdm 进度条。
                        如果日志级别设置为 warn 或更低，默认为 `True`，否则为 `False`。(`bool`, 可选)

  --use_intermediate_api
                        当启用自动并行 (`enable_auto_parallel=True`) 时，是否使用中间层 API 构建图。
                        默认为 `True`。(`bool`, 可选)

  --remove_unused_columns
                        是否在使用 `datasets.Dataset` 时自动移除模型 `forward` 方法未使用的列。
                        默认为 `True`。(`bool`, 可选)

  --new_special_tokens_path
                        存储额外的特殊 Token 的文本文件路径，用于扩展模型词汇表。
                        (`str`, 可选)

  --custom_register_path
                        自定义注册路径，用于加载自定义 template 和 mm_plugin。若不指定，则只注册默认部分。 (`str`, 可选)
```

# 2. 优化器与学习率调度

```shell
  --learning_rate
                        优化器的初始学习率。(`float`, 可选, 默认为 5e-5)

  --weight_decay
                        权重衰减系数。(`float`, 可选, 默认为 0.0)

  --optim
                        使用的优化器名称，支持 `adamw` (默认), `adamw_mini`, `adamw_custom`。
                        (`str`, 可选, 默认为 `adamw`)

  --num_train_epochs
                        训练的总轮数（Epochs）。(`float`, 可选, 默认为 1.0)

  --max_steps
                        训练的总步数。如果设置且大于 0，将覆盖 num_train_epochs。(`int`, 可选)

  --lr_scheduler_type
                        学习率调度器类型，支持 `linear`, `cosine`, `constant`, `constant_with_warmup`, `polynomial`。
                        (`str`, 可选, 默认为 `linear`)

  --warmup_steps / --warmup_ratio

                        线性预热的步数或比例。如果同时设置，`warmup_steps` 的优先级高于 `warmup_ratio`。
                        (`int` / `float`, 可选)

  --gradient_accumulation_steps
                        梯度累积步数。(`int`, 可选, 默认为 1)

  --max_grad_norm
                        梯度裁剪（Gradient Clipping）的最大范数。
                        默认为 1.0。用于防止梯度爆炸，对大模型训练稳定性至关重要。(`float`, 可选)

  --adam_beta1
                        AdamW 优化器的 beta1 超参数。默认为 0.9。(`float`, 可选)

  --adam_beta2
                        AdamW 优化器的 beta2 超参数。默认为 0.999。(`float`, 可选)

  --adam_epsilon
                        AdamW 优化器的 epsilon 超参数。默认为 1e-8。(`float`, 可选)

  --use_lowprecision_moment
                        是否使用 16-bit (低精度) 保存 AdamW 的动量（moment），用于节省显存。
                        默认为 `False`。(`bool`, 可选)

  --num_cycles
                        余弦调度器（cosine scheduler）中的波浪数。
                        默认为 0.5（即从最大值下降到 0 的半个余弦周期）。(`float`, 可选)

  --lr_end
                        多项式调度器（polynomial scheduler）的最终学习率。
                        默认为 1e-7。(`float`, 可选)

  --power
                        多项式调度器（polynomial scheduler）的幂次因子。
                        默认为 1.0。(`float`, 可选)

  --min_lr
                        余弦调度器（cosine scheduler）的最小学习率。
                        默认为 0.0。(`float`, 可选)
```

# 3. 分布式与并行策略 (PaddleFormers 特性)

```shell
  --sharding
                        启用 Paddle Sharding 数据并行策略。字符串配置，支持组合：
                        - `stage1`: 仅切分优化器状态。
                        - `stage2`: 切分优化器状态和梯度。
                        - `stage3`: 切分参数、梯度和优化器状态。
                        - `offload`: 可与 stage2/3 组合使用，将参数卸载到 CPU。
                        示例: `"stage1"`, `"stage2 offload"`。
                        (`str`, 可选)

  --sharding_parallel_size
                        Sharding 组的大小。默认为 -1（全局 Sharding）。
                        例如在多机训练中，可将其设置为 8，表示仅在单机内部 8 卡间进行 Sharding。
                        (`int`, 可选)

  --stage1_tensor_fusion
                        Sharding 并行配置，融合小 Tensor 通信。
                        默认为 `False`。(`bool`, 可选)

  --stage1_overlap
                        Sharding 并行配置，Stage1 通信计算重叠。
                        默认为 `False`。(`bool`, 可选)

  --stage2_overlap
                        Sharding 并行配置，Stage2 通信计算重叠。
                        默认为 `False`。(`bool`, 可选)

  --split_param
                        Sharding 并行配置，仅 Stage1 可用，切分参数以节省显存。
                        默认为 `False`。(`bool`, 可选)

  --sd_release_grads
                        Sharding 并行配置，提前释放梯度以减少峰值显存。
                        默认为 `False`。(`bool`, 可选)

  --fuse_optimizer_states
                        Sharding 并行配置，融合优化器状态（启用 Zero Cost Checkpoint 时必须开启）。
                        默认为 `False`。(`bool`, 可选)

  --sharding_parallel_mesh_dimension
                        指定 Sharding 在并行网格中对应的维度名称。
                        默认为 `dp`。(`str`, 可选)

  --sharding_comm_buffer_size_MB
                        Sharding 通信时熔合梯度的大小 (MB)。
                        默认为 -1 (即使用默认值 256MB)。(`int`, 可选)

  --sharding_offload_opt_buffersize_GB
                        使用 `hack_offload_optimizer` 时优化器 Offload 的 buffer 大小 (GB)。
                        默认为 -1（全部 Offload）。(`int`, 可选)

  --reorder_pipeline_priority
                        控制并行的执行顺序。默认为 `False` (PP 优先)。
                        设置为 `True` 时为 Sharding 优先，可能会改变通信拓扑的构建顺序。(`bool`, 可选)

  --split_norm_comm
                        是否开启单路 Sharding 时，将 Global Norm 的通信拆分为 PP 通信和 MP 通信分别进行。
                        默认为 `False`。(`bool`, 可选)

  --save_sharded_model
                        当使用 Sharding Stage 1 时，是否启用传统的 Sharded 保存模式（每个 rank 仅保存其负责的模型部分）。
                        默认为 `False`。(`bool`, 可选)

  --load_sharded_model
                        是否加载分散保存的 Sharded 模型（配合 `save_sharded_model` 使用）。
                        默认为 `False`。(`bool`, 可选)

  --tensor_model_parallel_size
                        张量并行（Tensor Parallelism）的并行度。(`int`, 可选)

  --mp_async_allreduce
                        张量并行的配置，异步通信。
                        默认为 `False`。(`bool`, 可选)

  --tp_delay_scale_loss
                        张量并行的配置，延迟 Loss 缩放。
                        默认为 `False`。(`bool`, 可选)

  --pipeline_model_parallel_size
                        流水线并行（Pipeline Parallelism）的并行度。(`int`, 可选)

  --send_recv_overlap
                        流水线并行的配置，是否将梯度发送/接收与GPU计算重叠以减少通信开销。
                        默认为 `False`。(`bool`, 可选)

  --split_backward
                        流水线并行的配置，是否将反向传播分解为多阶段以减少峰值显存。
                        默认为 `False`。(`bool`, 可选)

  --p2p_cache_shape
                        流水线并行的配置，启用不可变最大序列长度。
                        默认为 `True`。(`bool`, 可选)

  --use_dualpipev
                        流水线并行的配置，启用 DualPipe 调度。
                        默认为 `False`。(`bool`, 可选)

  --pp_delay_scale_loss
                        流水线并行的配置，重要精度选项。
                        默认为 `True`。(`bool`, 可选)

  --batch_p2p_comm
                        流水线并行的配置，启用批处理 P2P 通信。
                        默认为 `False`。(`bool`, 可选)

  --clear_every_step_cache
                        流水线并行的配置，清理每步缓存。
                        默认为 `False`。(`bool`, 可选)

  --sep_parallel_size
                        序列并行（Sequence Parallelism）的并行度。(`int`, 可选)

  --context_parallel_size
                        上下文并行（Context Parallelism）的并行度。(`int`, 可选)

  --use_expert_parallel
                        是否启用混合专家模型（MoE）的专家并行。(`bool`, 可选)

  --expert_model_parallel_size
                        专家并行的并行度。(`int`, 可选)

  --router_aux_loss_coef
                        MoE 模型的辅助损失（Auxiliary loss）权重系数。(`float`, 可选, 默认为 0.0001)

  --expert_max_capacity
                        MoE 专家的最大 Token 容量。(`int`, 可选)

  --expert_min_capacity
                        MoE 专家的最小 Token 容量。(`int`, 可选)

  --enable_auto_parallel
                        是否启用自动并行模式（Auto Parallel）。(`bool`, 可选)

  --hybrid_parallel_topo_order
                        混合并行通信拓扑顺序，影响通信效率。
                        支持选项: `pp_first` (dp -> pp -> sharding -> mp), `sharding_first` (dp -> sharding -> pp -> mp)。
                        (`str`, 可选, 默认为 `sharding_first`)

  --dp_allreduce_avg_in_gradinent_scale
                        数据并行的高级配置，在梯度缩放时使用 AllReduce Avg 替代 Sum+Scale。
                        默认为 `False`。(`bool`, 可选)

  --gradient_sync_after_accumulate
                        数据并行的高级配置，在梯度累积步后进行同步。
                        默认为 `False`。(`bool`, 可选)

  --sp_allreduce_avg_in_gradinent_scale
                        序列并行的高级配置字符串。在梯度缩放时使用 AllReduce Avg 替代 Sum+Scale。
                        默认为 `False`。(`bool`, 可选)

  --force_reshard_pp
                        即使脚本设置的 PP degree 与模型一致，是否强制重新切分流水线并行策略。
                        默认为 `False`。(`bool`, 可选)

  --split_inputs_sequence_dim
                        在使用序列并行 (Sequence Parallel) 时，是否在序列维度对输入数据进行切分。
                        默认为 `True`。(`bool`, 可选)

  --ddp_find_unused_parameters
                        在使用分布式数据并行 (DDP) 时，是否查找未使用的参数。
                        如果启用梯度重计算 (Recompute)，默认为 `False`；否则默认为 `True`。(`bool`, 可选)

  --sequence_parallel
                        是否正式启用序列并行（Sequence Parallel）。
                        需配合 `sep_parallel_degree` 使用。默认为 `False`。(`bool`, 可选)

  --fuse_sequence_parallel_allreduce
                        是否使用融合的序列并行 AllReduce 操作以提升性能。
                        默认为 `False`。(`bool`, 可选)

  --hybrid_parallel_expert_grad_scale
                        专家并行（Expert Parallel）下专家梯度的缩放因子。
                        用于平衡不同并行度下的梯度数值。如果不设置，将自动根据 TP 和 EP 并行度计算。(`float`, 可选)

  --nccl_comm_group_config
                        NCCL 通信组的配置文件路径。
                        用于对通信组进行细粒度的控制（如 buffer 大小等）。默认为 `None`。(`str`, 可选)
```

# 4. 精度与性能优化

```shell
  --fp16 / --bf16
                        是否使用 float16 / bfloat16 混合精度训练。(`bool`, 可选)

  --fp16_opt_level
                        混合精度的优化等级，可选 `O1` (混合), `O2` (纯fp16/bf16)。(`str`, 可选, 默认为 `O1`)

  --amp_master_grad
                        在 O2 模式下，是否使用 float32 存储梯度主权重以提高精度。(`bool`, 可选)

  --recompute
                        是否启用重计算（Gradient Checkpointing）以节省显存。(`bool`, 可选)

  --recompute_granularity
                        指定重计算的激活函数。(`str`, 可选)

  --recompute_method
                        指定哪些 Transformer 层需要被重计算。(`str`, 可选)

  --recompute_num_layers
                        当 recompute_method 为 uniform 时，recompute_num_layers 表示每个均匀划分的重新计算单元中的 Transformer 层数。
                        当 recompute_method 为 block 时，recompute_num_layers 表示每个流水线阶段内需要重新计算的 Transformer 层数。
                        (`int`, 可选)

  --recompute_modules
                        精细化重计算配置字符串。
                        支持控制的模块包括: `attention_column_ln`, `attention_row_ln`, `flash_attn`,
                        `mlp_column_ln`, `mlp_row_ln`, `global`。
                        格式示例: `{"attention_column_ln":0,"flash_attn":2}` (0表示不重计算，大于0表示进行重计算的数量)。
                        (`dict`, 可选)

  --tensorwise_offload_optimizer
                        是否开启逐张量优化器状态卸载 (Tensor-wise Offload)。
                        将优化器状态逐个卸载至 CPU，仅在更新时加载回 GPU。
                        相比 Sharding Offload，此选项可在非 Sharding 场景或配合 Sharding 进一步降低显存峰值。
                        注意 此选项目前不支持数据并行（Data Parallel）模式。
                        (`bool`, 可选)

  --bf16_full_eval / --fp16_full_eval
                        是否在评估过程中完全使用 bfloat16/float16 计算，而不是默认的 32-bit。
                        默认为 `False`。(`bool`, 可选)

  --scale_loss
                        FP16 训练的初始 Loss Scaling 值。
                        默认为 32768。(`float`, 可选)

  --amp_custom_black_list
                        自定义 AMP 的黑名单 OP 列表（强制使用 FP32）。(`List[str]`, 可选)

  --amp_custom_white_list
                        自定义 AMP 的白名单 OP 列表（强制使用 FP16/BF16）。(`List[str]`, 可选)

  --skip_profile_timer
                        是否跳过框架层面的计时器（Profile Timer），减少性能开销。
                        默认为 `True`。(`bool`, 可选)

  --flatten_param_grads
                        是否在优化器中使用扁平化梯度（仅限 NPU 设备）。
                        默认为 `False`。(`bool`, 可选)

  --eval_accumulation_steps
                        在将预测结果从 GPU 移动到 CPU 之前，累积的预测步数。
                        如果未设置，将累积整个预测结果后一次性移动（速度快但需大量显存）。
                        设置此值可减少评估时的显存峰值。(`int`, 可选)

  --minimum_eval_times
                        确保在整个训练过程中至少进行的评估次数。
                        如果根据 `eval_steps` 计算出的评估次数少于此值，将自动调整 `eval_steps`。(`int`, 可选)

  --skip_memory_metrics
                        是否跳过内存使用情况的监控报告。
                        默认为 `True`。跳过监控可以减少一定的性能开销。(`bool`, 可选)

  --pre_alloc_memory
                        预分配显存大小 (GB)。默认为 0。
                        用于在训练开始前预先占用一部分显存，减少碎片化或用于特定优化。(`int`, 可选)

  --num_nextn_predict_layers
                        NextN (Multi-Token Prediction) 预测层的数量。
                        用于支持 DeepSeek V3/R1 等包含多 Token 预测头的模型架构。默认为 0。(`int`, 可选)

  --skip_data_intervals
                        指定需要跳过的数据区间列表。
                        格式为 `[[start_step, end_step], ...]`，用于在特定步数跳过坏数据或进行调试。(`List[List[int]]`, 可选)

  --release_grads
                        是否在训练过程中尽早释放梯度以节省显存。
                        默认为 `False`。(`bool`, 可选)
```

# 5. 检查点管理

```shell
  --save_strategy
                        检查点保存策略 (`no`, `steps`, `epoch`)。(`str`, 可选, 默认为 `steps`)

  --save_steps
                        保存检查点的步数间隔。(`int`, 可选, 默认为 500)

  --save_total_limit
                        最多保留的检查点数量，旧的检查点将被删除。(`int`, 可选)

  --save_to_hf
                        是否以 HuggingFace Safetensors 格式保存模型权重。(`bool`, 可选, 默认为 `True`)

  --save_tokenizer
                        是否将 Tokenizer 保存到输出目录。(`bool`, 可选, 默认为 `True`)

  --save_checkpoint_format
                        检查点保存格式。支持:
                        - `unified_checkpoint`: 统一检查点格式，支持跨并行策略恢复。
                        - `flex_checkpoint`: 灵活检查点格式。
                        - `sharding_io`: 传统的 sharding 保存格式。
                        (`str`, 可选)

  --unified_checkpoint
                        是否开启统一检查点功能 (开关)。(`bool`, 可选, 默认为 `False`)

  --unified_checkpoint_config
                        Unified Checkpoint 的详细配置字符串。
                        支持:
                        - `async_save`: 启用异步保存，显著减少保存时的训练停顿。
                        - `master_weight_compatible`: 灵活处理主权重加载。
                        - `skip_save_model_weight`: 跳过模型权重保存（仅保存优化器）。
                        (`str`, 可选)

  --enable_zero_cost_checkpoint
                        是否启用零开销检查点（Flash Save），利用 RDMA 等技术加速保存。
                        配合参数:
                        - `zcc_workers_num`: 异步保存的进程数 (默认为 3)。
                        - `zcc_save_ema_coef`: 开启 EMA 权重保存及其衰减系数。
                        - `flash_device_save_steps`: Flash 设备上的保存频率。
                        (`bool`, 可选)

  --load_checkpoint_format
                        指定**加载**检查点的格式。选项同 `save_checkpoint_format`。
                        支持 `unified_checkpoint`, `sharding_io`, `flex_checkpoint`。(`str`, 可选)

  --save_hf_steps
                        每多少步保存一次 HuggingFace Safetensors 格式的检查点（独立于 `save_steps`）。
                        默认为 500。(`int`, 可选)
  --save_hf_total_limit
                        最多保留的 HuggingFace Safetensors 格式检查点数量，旧的检查点将被删除。
                        (独立于`save_total_limit`)(`int`, 可选)

  --save_rng_states
                        是否保存随机数种子状态，用于恢复训练的可复现性。
                        默认为 `True`。(`bool`, 可选)

  --resume_from_huggingface_ckpt
                        指向一个有效的 HF 格式检查点目录路径，用于恢复训练。(`str`, 可选)

  --ckpt_quant_stage
                        检查点量化等级，用于压缩保存体积。
                        支持：`O0` (关闭), `O1` (Int8), `O2` (Int4)。默认为 `O0`。(`str`, 可选)

  --aoa_config
                        FlexCheckpoint 的 AoA 配置字典或路径，用于描述权重映射关系。(`dict` 或 `str`, 可选)

  --load_via_cpu
                        是否先将检查点加载到 CPU 内存再按需传输到 GPU，用于缓解 GPU 显存不足。
                        默认为 `False`。(`bool`, 可选)

  --zcc_workers_num
                        Zero Cost Checkpoint (Flash Save) 的异步保存进程数。
                        默认为 3。(`int`, 可选)

  --zcc_ema_interval
                        ZCC 模式下 EMA 参数更新的步数间隔。
                        默认为 1。(`int`, 可选)

  --zcc_ema_loss_threshold
                        ZCC 模式下，仅当 Loss 小于此阈值时才进行 EMA 更新。(`float`, 可选)

  --ignore_save_lr_and_optim
                        保存检查点时，是否忽略优化器状态和学习率调度器状态（仅保存模型权重）。
                        (`bool`, 可选, 默认为 `False`)

  --ignore_load_lr_and_optim
                        恢复训练时，是否忽略加载优化器状态和学习率调度器状态。
                        (`bool`, 可选, 默认为 `False`)

  --output_signal_dir
                        用于存放异步保存信号文件的目录。如果不指定，默认为 `output_dir`。
                        (`str`, 可选)

  --use_async_save
                        是否使用 `AsyncSaver` 替代标准的 `paddle.save` 进行异步保存。
                        不同于 Unified Checkpoint 的异步，这是通用的异步保存实现。默认为 `False`。(`bool`, 可选)

  --optim_shard_num
                        将优化器状态切分为多少个分片进行保存。
                        默认为 1。(`int`, 可选)

  --save_sharding_stage1_model_include_freeze_params
                        在 Sharding Stage1 模式下保存模型时，是否包含被冻结（Freeze）的参数。
                        默认为 `False`。(`bool`, 可选)

  --pdc_download_ckpt
                        是否在 PaddleCloud 长任务环境中从远程存储下载检查点。
                        默认为 `False`。(`bool`, 可选)

  --pdc_download_timeout
                        从远程集群下载检查点的超时时间（秒）。
                        默认为 300。(`int`, 可选)

  --flash_device_save_steps
                        在使用 Flash Device（如 NVMe SSD）作为缓存层时，每多少步保存一次检查点。
                        通常配合 Zero Cost Checkpoint 使用。默认为 0（禁用）。(`int`, 可选)
```

# 6. 日志与评估

```shell
  --logging_steps
                        日志打印的步数间隔。(`int`, 可选, 默认为 500)

  --logging_first_step
                        是否在训练的第 1 步就打印日志。(`bool`, 可选, 默认为 `False`)

  --report_to
                        日志上报平台。支持 `visualdl`, `wandb`, `tensorboard`, `swanlab`。
                        可以使用 `all` 或 `none`。(`str` 或 `List[str]`, 可选)

  --load_best_model_at_end
                        训练结束后是否加载在验证集上表现最好的模型权重。需配合 `metric_for_best_model` 使用。
                        (`bool`, 可选, 默认为 `False`)

  --metric_for_best_model
                        用于判断模型好坏的指标名称（如 `loss`, `accuracy`）。(`str`, 可选)

  --greater_is_better
                        指定最优指标是越大越好（如准确率 `True`）还是越小越好（如 Loss `False`）。
                        如果不指定，将根据指标名称自动推断（例如 loss 默认为 False）。(`bool`, 可选)

  --count_trained_tokens
                        是否统计已训练的有效 Token 数量。
                        统计结果包含 `trained_effective_tokens` (不含 padding) 和 `trained_tokens`。
                        (`bool`, 可选)

  --wandb_api_key
                        Weights & Biases (WandB) 的 API Key，用于鉴权。
                        如果未设置，将尝试从环境变量中读取。(`str`, 可选)

  --wandb_http_proxy
                        连接 WandB 服务时使用的 HTTP 代理地址。(`str`, 可选)

  --metrics_output_path
                        训练指标（metrics）的保存路径（JSON 格式）。
                        如果设置，每个 rank 的指标将被转储到该目录下的文件中。(`str`, 可选)
```
