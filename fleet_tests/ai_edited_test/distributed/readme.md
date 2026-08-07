# Distributed Tests / 分布式模块测试

Unit tests for PaddleFleet distributed module including parallel state, model parallel config, and context parallel utilities.
PaddleFleet 分布式模块的单元测试，包括并行状态、模型并行配置和上下文并行工具。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_ai_context_parallel_utils.py` | Tests for mark_context_parallel_parameter_disable_scale_grad on layers / 测试层与参数的上下文并行缩放梯度禁用标记 |
| `test_ai_context_parallel_utils_extra.py` | Tests for scatter_balance and all_gather_balance with single rank / 单卡场景下的 scatter/gather balance 测试 |
| `test_ai_cp_flashmask.py` | Tests for FlashMaskContextParallel forward pass error handling / 测试 FlashMask 上下文并行前向传播错误处理 |
| `test_ai_cp_padding.py` | Tests for scatter_with_padding with various divisibility scenarios / 测试不同整除场景下的带 padding scatter 操作 |
| `test_ai_cp_scatter_gather_ops.py` | Tests for ContextParallelScatterOp and ContextParallelGatherOp PyLayers / 测试上下文并行的 scatter/gather PyLayer 算子 |
| `test_ai_distributed_extra.py` | Tests for distributed_model with AMP, pipeline parallel, and interleave / 测试分布式模型与 AMP、流水线并行设置 |
| `test_ai_distributed_init.py` | Tests for the distributed package import / 测试分布式包的导入 |
| `test_ai_flashmask_version_dispatch.py` | Tests for group and block_mask branch dispatch in flashmask backward / 测试 FlashMask 反向传播的分支分发逻辑 |
| `test_ai_fp8_extra.py` | Tests for get_quant_func with blockwise recipe and parameters / 测试 blockwise 量化函数获取 |
| `test_ai_model.py` | Tests for distributed_model function with PipelineLayer validation / 测试分布式模型函数与 PipelineLayer 校验 |
| `test_ai_model_parallel_config.py` | Tests for ModelParallelConfig dataclass defaults and constraints / 测试模型并行配置数据类的默认值与约束 |
| `test_ai_packed_seq_params.py` | Tests for PackedSeqParams dataclass default and custom values / 测试打包序列参数数据类 |
| `test_ai_paddlefleet_utils_extra.py` | Tests for make_viewless_tensor and MakeViewlessTensor PyLayer / 测试无视图张量创建与 PyLayer |
| `test_ai_parallel_state.py` | Tests for parallel_state initialize_model_parallel and group setup / 测试并行状态初始化与通信组设置 |
| `test_ai_parallel_state_extra2.py` | Tests for parallel_state getter functions when not initialized / 测试未初始化状态下的并行状态读取 |
| `test_ai_parallel_state_extra3.py` | Tests for set_virtual_pipeline_model_parallel_rank and deprecation / 测试虚拟流水线并行排序设置与弃用警告 |
| `test_ai_process_groups_config.py` | Tests for ProcessGroupCollection initialization / 测试进程组集合初始化 |
| `test_ai_recompute_utils.py` | Tests for need_recompute_in_block with various configurations / 测试不同配置下的块级重计算判断 |
