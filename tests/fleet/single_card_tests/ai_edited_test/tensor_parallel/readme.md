# Tensor Parallel Tests / 张量并行模块测试

Unit tests for PaddleFleet tensor parallel module including mappings, layers, random states, cross entropy, and data utilities.
PaddleFleet 张量并行模块的单元测试，包括映射、层、随机状态、交叉熵和数据工具。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_ai_cross_entropy.py` | Tests for VocabParallelCrossEntropy.calculate_logits_max / 测试词表并行交叉熵 logits 最大值计算 |
| `test_ai_cross_entropy_extra.py` | Tests for VocabParallelCrossEntropy.calculate_logits_max / 词表并行交叉熵 logits 最大值计算额外测试 |
| `test_ai_cross_entropy_extra2.py` | Tests for VocabParallelCrossEntropy.calculate_logits_max / 词表并行交叉熵 logits 最大值计算更多测试 |
| `test_ai_data.py` | Tests for _check_data_types / 测试数据类型检查函数 |
| `test_ai_data_extra.py` | Additional tests for _check_data_types / 数据类型检查额外测试 |
| `test_ai_data_extra2.py` | Tests for _MAX_DATA_DIM constant / 测试最大数据维度常量 |
| `test_ai_layers.py` | Tests for param_is_not_tensor_parallel_duplicate / 测试参数是否非张量并行副本 |
| `test_ai_layers_extra.py` | Tests for ColumnParallelLinear initialization / 测试列并行线性层初始化 |
| `test_ai_mappings.py` | Tests for _reduce helper function / 测试 reduce 辅助函数 |
| `test_ai_mappings_extra.py` | Tests for _AllToAll forward / 测试 AllToAll 前向传播 |
| `test_ai_mappings_extra2.py` | Tests for _reduce_scatter_along_first_dim with input_split_sizes / 测试带输入分割大小的 reduce scatter |
| `test_ai_mappings_extra3.py` | Tests for _split_along_first_dim with single GPU / 单 GPU 下沿首维分割测试 |
| `test_ai_mappings_extra4.py` | Tests for _reduce helper function / reduce 辅助函数额外测试 |
| `test_ai_mappings_extra5.py` | Tests for high-level helper functions with single GPU / 单 GPU 下高层辅助函数测试 |
| `test_ai_random.py` | Tests for CudaRNGStatesTracker initialization / 测试 CUDA 随机状态追踪器初始化 |
| `test_ai_random_extra.py` | Tests for get_expert_parallel_rng_tracker_name / 测试专家并行随机追踪器名称获取 |
| `test_ai_random_extra2.py` | Tests for initialize_rng_tracker / 测试随机追踪器初始化 |
| `test_ai_random_extra3.py` | Tests for CudaRNGStatesTracker initialization / CUDA 随机状态追踪器初始化额外测试 |
| `test_ai_random_extra4.py` | Tests for _get_cuda_rng_state / 测试 CUDA 随机状态获取 |
| `test_ai_random_extra5.py` | Tests for model_parallel_cuda_manual_seed / 测试模型并行 CUDA 随机种子 |
| `test_ai_tp_mappings_extra.py` | Tests for _AllToAll autograd function / 测试 AllToAll 自动微分函数 |
| `test_ai_tp_utils.py` | Tests for split_tensor_along_last_dim / 测试沿最后维度分割张量 |
| `test_ai_tp_utils_extra.py` | Tests for split_tensor_along_last_dim / 沿最后维度分割张量额外测试 |
| `test_ai_vocab_parallel_embedding.py` | Tests for VocabParallelEmbedding vocab range / 测试词表并行嵌入词表范围 |
