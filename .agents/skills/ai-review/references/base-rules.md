# PaddleFleet 基础评审规则

本文件是每次 PaddleFleet 评审都必须加载的基础规则。

## Release 分支评审

- 当 PR 的目标分支 `base_ref` 为 `release` 或匹配 `release/*` 时，适用本节；源分支名称不触发此规则。
- 不对该 release PR 的 diff 进行完整评审。必须在同一仓库定位唯一对应、目标分支为 `develop` 的 PR，并将该 develop PR 的 diff 与当前 release PR 的 diff 逐项比较。
- 优先依据标题、描述、提交记录或讨论中明确给出的 PR URL/编号建立对应关系；没有明确关联时，必须有唯一的源分支或等价变更证据，不能只按相似标题猜测。
- 仅评审 develop PR 相对于 release PR 新增、缺失或语义不一致的改动；release PR 中已存在且语义等价的改动不重复评审。
- release PR 的 diff 用作差异对照和为评论定位；所有评论和 Review Board 更新仍发布在 release PR 上，不在 develop PR 上发布评审结果。
- 找不到对应 develop PR 或对应关系存在歧义时，不回退为完整评审 release PR 的 diff，也不得批准；要求作者提供对应的 develop PR 后再继续。

## 功能正确性与兼容性

- 核对 PR 描述与实现是否一致，包括配置值、后端、支持的 shape/dtype 和测试范围。
- 检查空输入、零值、负值、上下限、重复输入、非法格式和异常路径，避免只覆盖正常流程。
- 公共 API、配置项、默认值、state dict 或序列化格式变化必须保持向后兼容，或提供明确迁移方案。
- 修改 `TransformerConfig`、`ModelParallelConfig` 时，同步检查 CLI/YAML 入口、默认值、所有消费方和配置测试。
- 跨模块变更必须检查接口两端，避免签名、返回值、dtype、shape、设备或生命周期约定不一致。

## 分布式训练

- 修改 `parallel_state.py`、`process_groups_config.py` 或通信逻辑时，检查进程组创建顺序、rank 成员、销毁和重新初始化。
- 集合通信必须使用正确的通信组，并保证各 rank 的调用次数、顺序、tensor shape 和 dtype 对称。
- 检查 TP、PP、EP、CP 组合下的局部 shape、序列切分、位置编码、随机数同步、共享权重和流水线边界。
- MoE 路径重点检查 top-k/归一化、token dispatch/combine 顺序、expert index、容量与丢弃策略、共享专家及梯度对称性。
- 重计算和 CUDA Graph 变更必须保持前向/反向一致、随机状态正确、地址稳定，并为动态 shape 保留安全回退路径。

## 算子、数值与性能

- 新增或修改 C++/CUDA 算子时，同步检查构建源文件、注册名、Python 封装、签名、版本要求和测试。
- 检查 shape/dtype 推导、设备与 stream、整数宽度、索引边界、内存生命周期和 kernel launch 错误处理。
- FP8、融合算子、Triton、TileLang 和 cuDNN 路径必须有明确的硬件/shape/dtype 保护，并与非融合实现保持数值一致或提供回退。
- 关注热路径中的主机同步和多余拷贝，如 `.cpu()`、`.numpy()`、不必要的 `to_tensor` 或 `contiguous()`。
- 检查低效循环、重复计算、内存泄漏及资源未释放；性能建议必须说明实际热点或复杂度影响。
- `packages/paddlefleet_ops/third_party/` 属于第三方代码，修改时应说明上游来源、修改原因和后续同步方式。

## 安全与错误处理

- 禁止硬编码密钥、令牌和凭据；外部输入必须经过校验，不能直接拼接到 shell、路径或反序列化操作。
- 检查权限、认证和敏感数据处理是否符合最小权限原则。
- 不使用 Python `assert` 承担运行时输入校验；应抛出明确异常并保留有效错误信息。
- 避免无处理的宽泛 `except Exception`，确认文件、显存、通信组和临时资源在失败路径中正确释放。

## 代码质量

- 函数职责应清晰，命名应表达真实语义；仅在影响理解、复用或正确性时提出可读性问题。
- 删除、重命名或移动代码时，检查导出、注册、调用点、文档和兼容别名是否同步。
- 新文件应包含仓库要求的版权声明；依赖和 lockfile 变更必须与实际需求一致。

## 测试质量

- 行为变化应放入最接近变更模块的现有测试目录，例如 `tests/single_card_tests/`、`tests/multi_card_tests/` 或 `auto_configurator/tests/`；分布式行为需要代表性的多卡覆盖。
- 测试必须覆盖关键边界和异常路径，并使用 `pytest.raises` 等机制验证预期错误。
- 测试之间不得相互依赖；外部网络、文件系统或服务应隔离，不能依赖不稳定的共享状态。
- 断言应验证核心结果，不接受吞掉异常、`assert True` 或通过 mock 掉被测函数来制造成功。
- 浮点结果应使用合理容差；重复场景优先参数化；异步测试必须正确 `await`。
- 修改 `ci/rules` 时同步检查 `ci/rule-tests` 和快照；新增 blacklist 不能替代回归测试。

## PR 信息与评论

- 检查 PR 标题是否清晰概括修改对象和目的；类别前缀可参考 PaddleFormers 仓库惯例，不强制单一格式。
- PR 描述至少说明为什么修改、解决什么问题，以及必要的验证方式；描述与实现不一致时指出具体差异。
- 评论必须具体、可执行并解释影响原因；意图不明确时先提出精确问题，不给出空泛总结。
