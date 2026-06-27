# Pipeline Parallel Tests / 流水线并行模块测试

Unit tests for PaddleFleet pipeline parallel communication, P2P operations, schedule nodes, and VPP simulator. / PaddleFleet 流水线并行通信、P2P 操作、调度节点和 VPP 模拟器的单元测试。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_ai_dw_overlap.py` | Tests for dw_p2p_overlap code paths in FusedGateDetachMatmul, TopKRouter, FP8OverlapProj / 测试数据权重重叠 P2P 代码路径 |
| `test_ai_forward_backward_overlap_extra.py` | Tests for ScheduleChunk initialization / 测试调度块初始化 |
| `test_ai_forward_backward_overlap_utils.py` | Tests for FakeClone in forward_backward_overlap_utils / 测试前向反向重叠工具中的 FakeClone |
| `test_ai_four_directions_p2p_communication.py` | Tests for XPU communication group management / 测试四方向 P2P 通信组管理 |
| `test_ai_four_directions_p2p_extra.py` | Tests for SendRecvMeta in four_directions_p2p / 测试四方向 P2P 的收发元数据 |
| `test_ai_four_dirs_p2p_helper.py` | Tests for P2pHelper recv_forward/recv_backward with sync_recv / 测试 P2P 辅助器的同步接收 |
| `test_ai_four_dirs_p2p_ops.py` | Tests for _p2p_helper with recv_prev and recv_next / 测试 P2P 辅助器的前后向接收 |
| `test_ai_four_dirs_p2p_sync.py` | Tests for _p2p_helper with _sync_send=True / 测试 P2P 辅助器的同步发送 |
| `test_ai_four_dirs_partial_ops.py` | Tests for SendRecvMeta with key messages / 测试关键消息的收发元数据 |
| `test_ai_overlap_edge_cases.py` | Edge case tests for detach_and_requires_grad / 测试 detach_and_requires_grad 边界情况 |
| `test_ai_overlap_noop_streams.py` | Tests for NoopScheduleNode in pipeline_parallel/utils / 测试空操作调度节点 |
| `test_ai_overlap_schedule_node_fb.py` | Tests for ScheduleNode.forward / 测试调度节点前向传播 |
| `test_ai_overlap_schedule_node_first.py` | Tests for ScheduleNode.first_forward / 测试调度节点首层前向传播 |
| `test_ai_overlap_schedule_node_recompute.py` | Tests for ScheduleNode with recompute / 测试带重计算的调度节点 |
| `test_ai_p2p_batched_ops.py` | Tests for _batched_p2p_ops / 测试批量 P2P 操作 |
| `test_ai_p2p_comm_meta.py` | Tests for SendRecvMeta recv_meta and send_meta / 测试收发元数据的广播 |
| `test_ai_p2p_comm_ops.py` | Tests for initialize_p2p_groups / 测试 P2P 通信组初始化 |
| `test_ai_p2p_comm_p2p_helper.py` | Tests for batch_send_recv_on_calc_stream / 测试计算流上的批量收发 |
| `test_ai_p2p_communication.py` | Tests for SendRecvMeta class / 测试收发元数据类 |
| `test_ai_p2p_communication_extra.py` | Tests for SendRecvMeta initialization / 收发元数据初始化测试 |
| `test_ai_p2p_dynamic_shape.py` | Tests for P2pHelper with dynamic_shape=True / 测试动态形状 P2P 辅助器 |
| `test_ai_p2p_helper_dynamic.py` | Tests for PADDLE_P2P_SYNC_SEND env variable / 测试 P2P 同步发送环境变量 |
| `test_ai_pipeline_hooks.py` | Unit tests for pipeline_hooks module / 测试流水线钩子模块 |
| `test_ai_pipeline_parallel.py` | Tests for get_action function / 测试流水线动作获取函数 |
| `test_ai_pipeline_parallel_extra.py` | Tests for _get_align_mode_scale function / 测试对齐模式缩放函数 |
| `test_ai_pipeline_parallel_withinterleave.py` | Tests for P2PAsyncHandle dataclass / 测试 P2P 异步句柄数据类 |
| `test_ai_pipeline_parallel_withinterleave_fthenb.py` | Tests for PipelineParallelWithInterleaveFthenB / 测试交错流水线前向后向并行 |
| `test_ai_pipeline_utils.py` | Tests for is_pp_first_stage / 测试流水线首阶段判断 |
| `test_ai_pipeline_utils_ranks.py` | Tests for get_pp_first_rank / 测试流水线首排序获取 |
| `test_ai_pipeline_utils_stage.py` | Tests for profile_pipeline_details / 测试流水线详情分析 |
| `test_ai_pp_layers.py` | Tests for ScheduleChunk / 测试调度块 |
| `test_ai_pp_utils.py` | Tests for paddle_2_number conversion / 测试 Paddle 数值转换 |
| `test_ai_pp_utils_extra.py` | Tests for paddle_2_number conversion / Paddle 数值转换额外测试 |
| `test_ai_vpp_balanced_memory.py` | Tests for OffloadQueue / 测试卸载队列 |
| `test_ai_vpp_simulator.py` | Tests for ChunkType enum / 测试块类型枚举 |
| `test_ai_vpp_simulator_extra.py` | Additional tests for Chunk in vpp_simulator / VPP 模拟器块额外测试 |
| `test_ai_vpp_simulator_schedule.py` | Tests for VPPSimulator schedule with various configurations / 不同配置下的 VPP 模拟器调度测试 |
