# Muon 正交切分规则

- 适用路径：`src/paddlefleet/transformer/muon_utils.py`、`src/paddlefleet/transformer/**` 下任何定义 `muon_slice_specs` 的模块（当前为 `attention.py`、`multi_latent_attention.py`、`csa_attention.py`、`dsa_attention.py`、`dsv4_hybrid_attention.py`、`mlp.py`、`moe/moe_expert.py`）、`tests/single_card_tests/transformer/test_muon_slice_specs.py`、`tests/single_card_tests/transformer/test_muon_hybrid_mla_grouping.py`
- 触发条件：新增或修改 `muon_slice_specs`；新增、重命名、删除、改变 layout 的可训练权重（含 `nn.Linear` 子层、裸 `create_parameter`、grouped-gemm 的 3D 权重）；新增带权重的 attention/MLP/MoE/indexer 变体；修改 `muon_utils.py` 的任何 helper；修改影响权重打包方式的配置开关（融合 QKV、`gated_linear_unit`、`gated_attention`、VHA premix/postmix、EP dispatcher、latent 吸收拆分）
- 规则来源：`muon_utils.py` 模块 docstring 声明的契约（融合权重必须逐个独立正交）；消费方 `PaddleFormers/paddleformers/trainer/trainer.py` 的 `_build_muon_slice_config` / `_build_muon_param_info_map`；`dsa_attention.py` 的 `DSAIndexer.muon_slice_specs` docstring 记录的真实缺陷（DSA indexer 曾漏声明，导致 `wq_b` 静默失去逐头正交）；`tests/single_card_tests/transformer/test_muon_hybrid_mla_grouping.py::TestIndexerMuonSliceSpecSymmetry` 固化的对称性约定
- 引用约定：只用符号名（类名、方法名、参数路径、helper 名）定位代码，不写行号
- 评审范围：只评审本 PR diff 中新增或修改的模块与权重。存量未声明 spec 的权重不因本规则报告，除非本 PR 正在改动它们、或本 PR 新增的权重与它们打包在同一张张量里

## 机制背景（判定所有规则的前提）

消费链路是**精确字符串匹配 + 静默降级**，这是本文件所有规则的成因：

1. `_build_muon_slice_config` 遍历 `model.named_sublayers()`，对每个子层取 `muon_slice_specs`，把返回的每个相对参数路径 `rel` 拼成全局 key `f"{name}.{rel}"`。
2. `_build_muon_param_info_map` 遍历 `model.named_parameters()`，只在 `pp_name in slice_config` 时取出 `(slice_fn, kwargs)` 并 `partial(slice_fn, **slice_kwargs)`；否则 `split_concat_func = None`。
3. `split_concat_func is None` 意味着 Muon 把整张融合张量当作一个矩阵做 Newton-Schulz。**不抛异常、不告警、loss 不炸**，只是更新方向错了。唯一的痕迹是 `_build_muon_param_info_map` 每个参数一行的 INFO 日志会打出 `split_concat_func: None`——它对正常参数也这么打，所以不是可用的告警信号。

因此：漏声明、key 拼错、参数被重命名、kwargs 算错 head 数——这四类问题的现场表现完全相同，都是"训得出来但不对"。评审必须在静态层面拦住，不能指望测试或线上指标发现。

另有两条消费路径需要区分：ErnieBot 在 `fleet_model/ernie5_v2/modeling.py` 自带 `build_muon_param_info_map`，会短路模块遍历。**不得以"生产走 ErnieBot 那条路"为理由省略 `muon_slice_specs`**：PaddleFleet 必须独立自洽，这正是 DSA indexer 缺陷的成因。

`muon_configs` 本身不是 PaddleFleet 的配置项：`transformer_config.py` 里没有任何 muon 字段，它来自消费方的 `model.config.muon_configs`。因此本文件只约束 `muon_slice_specs` 如何**读**这个 dict，`user-configuration-rules.md` 的开关评审规则不适用于这些键。

## 规则优先级

参数名匹配错误 ≈ 融合权重的覆盖性缺失 > kwargs 数值/轴向错误 > 并行分片正确性 > 开关守卫 > 测试缺失 > 注释与文档。

判定"覆盖性缺失"的严重程度时区分两种情况：漏切**真正融合在一张张量里的多个矩阵**（逐头打包、融合 QKV、融合 gate/up、group-major 2D、EP 分片）会静默产生错误的更新方向，属最高优先级；而漏给 3D stacked 权重挂 `ortho_stacked` 不改变数值结果——`ortho_stacked` 的实现只是 `ndim != 3` 的校验加一次 `ortho_fn(weight)`，Muon 本来就把 3D 的 leading 轴当 batch 处理，所以漏挂只是丢了一道形状断言。后者按提示级报告，不要写成"更新方向错误"。

## 应切尽切：覆盖性判定

新增或修改带权重的模块时，逐个参数回答"这张张量是不是一个矩阵"。**只要一张张量在数学上承载多个逻辑独立的矩阵，就必须切**。判定依据是该权重在 forward 中的实际用法，不是它的形状看起来像什么。

必须切的打包形态，及对应的现有先例：

- **逐头打包**：输出轴按 head 拼接，每个 head 是独立矩阵。用 `ortho_per_head` + `heads`。先例：`MLASelfAttention` 的 `q_b_proj.weight`（`head_sizes=[qk_nope, qk_rope]`）、`CSAIndexer` 的 `linear_wq_b.weight`、`DSAIndexer` 的 `wq_b.weight`、`SelfAttentionVHA` 的 `q_proj.weight` / `k_proj.weight` / `v_proj.weight`。
- **单头内多语义段**：一个 head 内部由若干语义不同的段拼成（nope/rope、k/v、latent/rope）。用 `head_sizes` 逐段列出，不能只按 head 数均分。先例：`kv_a_proj_with_mqa.weight` 的 `head_sizes=[kv_lora, qk_rope]`、`Compressor` 的 `linear_wkv.weight` / `linear_wgate.weight`。
- **融合 QKV**：按 layout 选 helper。交错布局（group 内 Q/[Gate]/K/V 连续）用 `ortho_qkv_interleaved` + `groups` / `role_sizes` / `heads_per_group`；连续布局（`[all_Q | all_K | all_V]`）用 `ortho_qkv_contiguous` + `heads` / `groups` / `head_dim` / `v_head_dim`。先例：`SelfAttention.muon_slice_specs` 按 `gpt_model_use_experimental_version` 分派这两种。
- **融合 gate/up**：`up_gate_proj.weight`、grouped expert 的 `weight1`。用 `ortho_gate_up`（切点由 shape 推导，TP 下自动按本地 shard 切）。
- **stacked / grouped 3D（只保形状断言）**：leading 轴枚举独立矩阵。用 `ortho_stacked`。先例：grouped expert 的 `weight2`、`vha_premix_weight`。如上节所述这一条不改变数值结果，按提示级报告。
- **group-major 打包的 2D**：形状是 2D 但逻辑上 leading 轴打包了多个矩阵（grouped gemm 前的 flatten）。用 `ortho_per_head` + `axis=-2`。先例：`DSv4HybridSelfAttention` 的 `linear_o_group_proj`（`[o_groups * o_lora_rank, d]`，实际按 `[g, r, d]` 用）。
- **依附于分组输出的 gate**：gate 权重的列布局跟随它所乘的对象。乘 per-head 输出就按 head 切，乘 group-major 输出就按 group 切。先例：`SelfAttentionVHA` 的 `gate_proj.weight` 按 `num_attention_heads` 切，而 `DSv4HybridSelfAttention` 的 `gate_proj.weight` 按 `o_local_groups` 切——**同名参数在不同模块的切法不同，不能照抄**。
- **形状随配置变化的权重**：同一个参数在不同开关下可能是 2D 也可能是 3D，必须按每种拓扑分别判定。典型是 DSv4 的 `vha_postmix_U` / `vha_postmix_V`：`vha_postmix_grouped=False` 时是 `[num_attention_heads, rank]` 普通 2D，`=True` 时是 `[o_groups, group_heads, rank]`，且 forward 用 `einsum("btgjd,gjr->btgrd", ...)` 按 `g` 逐组消费，leading 轴是独立矩阵。`DSv4HybridSelfAttention.muon_slice_specs` 的 docstring 只写了"postmix 是普通 2D 所以不标记"，对 grouped 拓扑并不成立——评审新增此类形状可变权重时，要求作者对每种取值分别说明，不要沿用这句 docstring。
- **EP 分片的 intermediate**：见"并行与分片"一节。

不需要切、也不应报告的形态：

- 1D 参数（`k_norm`、各类 norm weight/bias、scale）——`_default_should_use_muon` 按 `len(shape) not in (2, 3)` 直接排除，根本到不了 Muon。
- 真正的单矩阵：`DSAIndexer` 的 `wk`（单个共享 head）、`weights_proj`（输出维**就是** head 数，本身不是打包）。这些留给 Muon 直接处理是正确的，`dsa_attention.py` 的 docstring 已显式说明"故意不标记"。
- 在当前模式下拿不到梯度的权重：`mqa_latent_split_kv_b` 打开时的 `kv_b_proj`，声明 spec 只会白烧一次 Newton-Schulz。这类"故意不切"必须在注释里写明理由，评审时要求补注释而不是补 spec。

评审动作：判定依据是**权重张量自身的列/行是否由多个独立矩阵拼成**，不是它的输出激活是否被 reshape。几乎所有 attention 投影的输出都会 reshape 成 `[b, s, h, d]`，`wk` / `weights_proj` 也在 reshape 之后被消费，但它们是单矩阵。正确的检查方式：找到该权重的构造处，看它的 out-features 是不是由 `heads * head_dim`、`groups * rank`、`2 * intermediate` 这类乘积或多段和构成；再找到 forward 里它是否被 `paddle.split` / 分组 einsum 按那个因子拆开使用。两处都成立才报告覆盖性缺失，并给出应使用的 helper 与 kwargs。

新增带权重的模块类不定义 `muon_slice_specs` 时，按提示级报告，并要求作者给出结论：该模块的权重是否全是单矩阵（结论可写在类 docstring 或 PR 描述里）。仓库现状是"只有已经评估过的模块才定义这个方法"，没有任何模块用返回 `{}` 的空实现做占位，因此不要把"必须写一个 `{}` 桩"当成既有约定来要求。同时注意"没定义"不等于"不需要切"：`gated_delta_net.py` 与 `kimi_delta_attention.py` 的 `in_proj` 都是多角色融合投影，属于上面"融合 QKV"一条要求切的形态，只是尚未评估，两者的实际通道边界见下一节；`block_attn_res.py` 的 `proj_weight` 形状是 `[1, hidden_size]`，本身就是单矩阵。存量缺口按本文件"评审范围"一节不报告，但本 PR 若正在改动这些模块的权重，就要一并处理。

### 线性注意力 `in_proj` 的通道边界（GatedDeltaNet 与 KDA 不同）

这两个模块的 `in_proj` 都是融合投影，但**角色顺序、角色集合和是否存在都不一样**，不能相互套用。切分必须按各自 forward 里 `paddle.split` 的实际 `split_sizes` 给边界，任何一段错位都会把 beta / gate 的列混进 q/k/v 的正交块里。以下宽度都是构造时的全局值，forward 里每段再除以 `tp_size`。

`GatedDeltaNet`（`gated_delta_net.py`）：

- `in_proj_dim = qk_dim * 2 + v_dim * 2 + num_value_heads * 2`，其中 `qk_dim = key_head_dim * num_key_heads`、`v_dim = value_head_dim * num_value_heads`。
- 通道顺序 `[q | k | v | gate | beta | alpha]`。forward 先按 `[qk_dim * 2 + v_dim, v_dim, num_value_heads, num_value_heads]` 切成 `qkv, gate, beta, alpha`，`qkv` 过完 depthwise conv 后再按 `[qk_dim, qk_dim, v_dim]` 切成 q/k/v。
- 切分要求：`q`、`k` 各按 `num_key_heads` 逐头切（每块 `key_head_dim` 宽），`v`、`gate` 各按 `num_value_heads` 逐头切（每块 `value_head_dim` 宽）。`beta`、`alpha` 每段宽度就等于 `num_value_heads`，即每个 head 只占 1 列，逐头切退化成 1 宽的块、正交化没有意义，应作为整段各自保留一块，**但不能和 q/k/v 合并成一块**。

`KimiDeltaAttention`（`kimi_delta_attention.py`）：

- `in_proj_dim = qk_dim * 2 + v_dim + num_value_heads`，再在 `use_full_rank_gate=True`（默认，也是 Kimi 实际用的配置）时 `+= v_dim`。**没有 alpha 段**：forget gate 走独立的低秩对 `f_a_proj` / `f_b_proj`。
- 通道顺序 `[q | k | v | beta | gate]`——beta 在 gate **之前**，与 GatedDeltaNet 的 `gate` 在 `beta` 之前正好相反。forward 按 `[conv_dim_local_tp, num_value_heads / tp, (v_dim / tp)]` 切成 `qkv, beta[, gate]`，其中 `conv_dim = qk_dim * 2 + v_dim`。
- `use_full_rank_gate=False` 时 `in_proj` 只有 `[q | k | v | beta]` 四段，输出门变成独立的低秩对 `g_a_proj` / `g_b_proj`，`in_proj` 里**没有 gate 段**。给这种配置套用 full-rank 的边界会把 beta 的列当成 gate 处理。
- 切分要求：q/k 按 `num_key_heads`、v 按 `num_value_heads` 逐头切；`beta` 同上是每头 1 列，整段保留；full-rank 的 `gate` 段按 `num_value_heads` 逐头切（每块 `value_head_dim` 宽，KDA 强制 `key_head_dim == value_head_dim`）。低秩配置下另需评估 `f_b_proj` / `g_b_proj`：它们的输出维是 `v_dim`，同样是按 value head 打包的，属于逐头打包形态。

评审要点：为这两个模块新增 spec 时，`role_sizes` / `sizes` 必须逐条对着该文件 forward 里的 `split_sizes` 核对，并覆盖 `use_full_rank_gate` 的两个取值。`ortho_qkv_interleaved` / `ortho_qkv_contiguous` 都不适用——前者假设 group 内 Q/[Gate]/K/V 交错，后者假设 `[all_Q | all_K | all_V]` 三段且无 gate 角色，而这里是"多角色单段连续、且每个角色的 head 数与 head 宽都不同"。应使用 `ortho_blocks` 并显式给出完整的 `sizes` 列表，或新增一个专用 helper；无论哪种，都要按上面的角色顺序把每个角色再展开成逐头的块宽。

## 参数名匹配

key 是**相对该子层的参数路径**，拼接后必须与 `named_parameters()` 的名字逐字符相同。

- 子层里的 `nn.Linear` 写 `"<attr>.weight"`；直接 `create_parameter` 挂在本层的裸参数写 `"<attr>"`，**不带 `.weight` 后缀**。对照：`linear_o_group_proj`、`k_b_proj`、`v_b_proj`、`vha_premix_weight`、grouped expert 的 `weight1` / `weight2` 都是裸参数；`q_b_proj.weight`、`gate_proj.weight` 是子层。写错后缀等于 spec 不生效，且没有任何运行时信号。
- 重命名或移动权重属性时，必须在同一 PR 内同步 `muon_slice_specs` 的 key。评审时对 diff 中每个被重命名的权重，用 `rg <old_name>` 确认 spec 里没有残留旧名。
- 可选权重必须用 `hasattr` / `getattr(self, ..., None) is not None` 守卫，只为本 rank 上真实存在的参数产生 spec。参考 `SelfAttentionVHA.muon_slice_specs` 对 `shared_kv_proj` / `k_proj` / `v_proj` / `gate_proj` / `vha_premix_weight` 的逐个守卫。无守卫地声明可选权重会往 `slice_config` 里塞永不匹配的死 key，掩盖真正的漏配。
- 互斥权重必须写成 `if/else`，不能同时声明。参考 DSv4 按 `use_vha_premix` 在 `vha_premix_weight` 与 `linear_q_up_proj.weight` 之间二选一，以及 MLA 按 `mqa_latent_split_kv_b` 在 `k_b_proj`/`v_b_proj` 与 `kv_b_proj.weight` 之间二选一。
- 父子类都定义 `muon_slice_specs` 时，注意模块遍历会分别调用两者；子类覆写必须覆盖父类全部条目，或显式调用父类实现再补充，不能让父类条目丢失。

## helper 选择与 kwargs 正确性

kwargs 算错不会报错，只会把矩阵切在错误的位置上，因此每个数值都要对着权重的构造代码核对。

- `heads` / `groups` 取哪个属性由**该权重在本 rank 上是否按该轴分片**决定，不能一律套 `_per_partition`，也不能一律用全局值。TP 切了该轴就必须用 per-partition 值（`num_attention_heads_per_partition`、`num_query_groups_per_partition`），否则 `paddle.split` 会切出错误块数或直接抛维度错误。TP 未切该轴时用全局配置值才是对的：`o_local_groups` 来自 `config.o_groups`，`CSAIndexer.index_n_heads` 与 `DSAIndexer.n_heads` 都来自 `config.dsa_index_n_heads`，indexer 的 `wq_b` 在构造处注明是 TP 复制（duplicated）而非分片，所以全局 head 数正确。同理 `SelfAttentionVHA` 的 `gate_proj.weight` 用 `num_attention_heads`，因为它的列布局跟随 premix 展开后的全 head 输出。评审时到权重构造处核对该轴的实际本地宽度，不要只看属性名里有没有 `_per_partition`。
- `head_sizes` 是**单个 head 内部**的段宽列表，`ortho_per_head` 内部按 `head_sizes * heads` 展开。写成整个权重的段宽、或与 `heads` 语义重复，都会切错。`heads` 缺省为 1，只给 `head_sizes` 表示整张权重就是一个 head 的多段结构（如 `kv_a_proj_with_mqa.weight`、`Compressor` 的两个权重）。
- `axis` 默认 `-1`。切 leading 轴时用 `-2` 而不是 `0`：Muon 可能把多个同形状参数 batch 成 3D 一起送进来，写 `0` 会切到 batch 轴上。理由记录在 `k_b_proj` / `v_b_proj` 的注释里，新增的 leading 轴切分必须沿用 `-2`。注意 `linear_o_group_proj` 的注释写的是"must be split along axis 0"而 kwargs 传的是 `axis=-2`——注释与实参不一致，以 `-2` 为准；评审时也要按这个例子检查新增 spec 的注释与 kwargs 是否自相矛盾。
- `transposed=True` 只用于"该块相对 Muon 缩放调优时的朝向是转置存放的"。理由见 `_transposing` 的 docstring 与 `v_b_proj` 的注释：`muon_version` 1/2 的 `_scaling_fn` 按 `dout / din` 缩放，对两个矩阵维不对称，转置存放的块会被乘上倒数比例；只有 `muon_version=3` 的 `max(dout, din) ** 0.5` 对称。判断依据是同族权重之间的朝向差异（`v_b_proj` 是 `[v_head_dim, kv_lora_rank]`，而 `k_b_proj` 是 `[kv_lora_rank, qk_nope_head_dim]`），这一点可以在仓库内静态核对；`_scaling_fn` 与 `muon_version` 本身在 `paddle.optimizer.muon` 里、两个仓库都看不到，所以不要断言具体的缩放系数，只要求作者说明朝向结论。朝向相反却没加 `transposed=True`、或无朝向差异却加了，都要报告。
- `ortho_stacked` 对非 3D 抛 `ValueError`。`ortho_gate_up` 只用 `assert weight.ndim in (2, 3)`（AssertionError，且 `python -O` 下会被剥掉），并直接取 `weight.shape[-1] // 2` 作切点，**没有偶数校验**：末轴为奇数时会静默截断再在 `paddle.split` 里失败。唯一显式的偶数校验在 `ortho_ep_full_intermediate` 的 `gate_up=True` 分支。给形状不符的权重挂这两个 helper 属于要到首个 optimizer step 才暴露的错误，评审时按权重的实际 `ndim` 与末轴宽度核对。
- `ortho_qkv_interleaved` 的 `role_sizes` 长度决定是否含 Gate：3 项为无门控，4 项为带门控，且内部固定假设"最后两项是 K、V，各算一个 head，其余为 Q/Gate 各占 `heads_per_group` 个 head"。新增第五种 role 或改变 role 顺序时，必须同步改 helper 的 `heads_by_role` 推导，不能只改调用方。
- `per_head=False`（即 `muon_qkv_update_mode == "split_qkv"`）走的是"按 role 跨 group 聚合成一个矩阵再散回"，与 `split_head` 的数学含义不同。新增模式或改变 `per_head` 的推导时，要确认两条路径都被覆盖，且散回的 group 顺序与原 layout 一致。
- 修改 `muon_utils.py` 的任何 helper 时，用 `rg <helper_name>` 列出全部调用方，确认新增/改名的 kwarg 在每个调用点都已更新，且默认值不会静默改变既有调用的行为。helper 签名属于跨模块契约，改动必须两端同时检查。

## 并行与分片正确性

- 切点应尽量由**本地 shape** 推导而非全局配置。`ortho_gate_up` 从 `weight.shape[-1] // 2` 取切点，正是为了让 TP 下每个 rank 切自己的 shard；新增 helper 时优先沿用这一做法，避免把全局 `ffn_hidden_size` 之类写进 kwargs。
- **EP 分片的 intermediate 必须先重分布再正交**。`moe_token_dispatcher_type == "allgather"` 且 EP > 1 时，一个 rank 持有全部 expert 但每个只有 `moe_intermediate_size // EP`，本地 Newton-Schulz 会对着一个薄片做正交，与非 EP 运行的更新不等价。此时必须用 `ortho_ep_full_intermediate`，并按被分片的轴给对 `shard_axis`：fc1 权重 `[E, H, 2I/EP]` 用 `-1`，fc2 权重 `[E, I/EP, H]` 用 `-2`。参考 `GroupedMLPExpert.muon_slice_specs` 对 `intermediate_ep_sharded` 的分支。
- `ortho_ep_full_intermediate` 内含两次 `dist.alltoall`，属于集合通信：所有 rank 必须无条件、同顺序、同 shape 地进入这条路径。因此决定是否走 EP 分支的条件（此处是 `intermediate_ep_sharded`）必须是各 rank 一致的配置量，不能依赖 rank 本地状态、参数是否有梯度、或某个 rank 独有的分支。评审新增的通信型 helper 时，明确检查这一点，并确认 `ep_group` 非 None、expert 轴能被 `ep_size` 整除这两个前置校验仍在。
- 新增分片形态（新的 dispatcher、新的并行维度切权重）时，检查它是否让某张权重在本 rank 上不再是完整矩阵。是则需要类似的重分布包装，仅调整 `heads` 是不够的。
- 需要跨 rank 通信的 spec 只在多卡下有意义，`tests/multi_card_tests/` 需要代表性覆盖；纯本地切分的 spec 放单卡测试即可。

## 开关守卫

- 读 `muon_configs` 一律用 `.get(key, default)`，默认值必须与既有模块一致：`muon_qkv_update_mode` 默认 `"split_head"`，`muon_ffn_split` 默认 `False`。改变默认值等于静默改变所有既有配置的更新方向，按基础规则的兼容性条款报告。
- 不支持的模式必须早返回 `{}`，而不是返回一份"凑合能用"的 spec。attention 类模块的既有约定：仅 `split_head`（部分模块另加 `split_qkv`）时返回 spec，其余模式返回 `{}`。
- **禁止无条件 `return {...}`**：无守卫的 spec 会在用户显式关闭切分时仍然切分，且会让针对守卫写的测试全部空过。评审时确认守卫条件与该模块实际支持的模式一致——例如 `SelfAttention` 的实验版 layout 只支持 `split_head`，即使非实验版同时支持 `split_qkv`。
- 守卫条件不得依赖 rank 本地状态或运行期张量，只能读配置与模块结构属性；`muon_slice_specs` 在建 optimizer 时被调用一次，此时不应触发任何计算。
- 新增 `muon_configs` 键时，同一 PR 内必须给出该键在消费方（`model.config.muon_configs` 的产出处）的声明与默认值，不要只在 `muon_slice_specs` 里就地 `.get` 一个 PaddleFleet 侧无从校验的键。这类跨仓键的命名与默认值评审不走 `user-configuration-rules.md`（该文件只约束 `transformer_config.py` / `model_parallel_config.py` 里的字段），但仍要求默认值保持既有行为。

## 测试要求

新增或修改 spec 必须补测试。仓库现有覆盖分布在两个文件，报告时要指向正确的那个：

- `tests/single_card_tests/transformer/test_muon_slice_specs.py`：逐模块的 spec 内容与形状执行，以及 `muon_utils.py` 各 helper 的直接测试。新增/修改某个模块的 spec 默认加在这里。
- `tests/single_card_tests/transformer/test_muon_hybrid_mla_grouping.py`：跨模块的对称性与存在性约束（`TestIndexerMuonSliceSpecSymmetry`），以及 helper "真的逐块生效"的验证。

`test_muon_slice_specs.py` 的既有做法是两段式：

1. `muon_slice_specs` 未绑定调用 + `SimpleNamespace`（或 `_fake()` 工厂）提供它读取的少量属性，断言返回的 key 集合与 `(helper, kwargs)` 内容。
2. 把 spec 交给 `_run_specs`，用与真实权重同构的合成张量执行一遍。`_run_specs` 内部用 `_OrthoRecorder` 记录每个被送进 `ortho_fn` 的块形状，并断言输出形状等于输入形状。

必须覆盖的点：

- **块形状而非仅块数量**。断言 `_run_specs` 返回的 recorder 的 `.shapes` 等于预期形状列表，例如 `linear_o_group_proj` 期望 `[(o_lora_rank, HIDDEN)] * o_groups`。只断言 `len(shapes)` 无法区分"切对了"和"切成等宽但错位的块"。
- **反向用例（anti-vacuity）**。至少一条断言在守卫关闭时返回 `{}`，一条断言可选权重缺席时对应 key 不出现。参考 `test_muon_hybrid_mla_grouping.py` 里 `TestIndexerMuonSliceSpecSymmetry.test_both_opt_out_when_mode_is_not_split_head` 的注释：无此用例，一个无条件 `return {...}` 会通过全部正向断言。
- **helper 真的逐块生效**。新增 helper 时，用返回块索引的 marker `ortho_fn` 验证各输出块彼此不同且形状保持，参考 `test_muon_hybrid_mla_grouping.py::test_ortho_per_head_is_actually_per_head`。
- **同族模块对称性**。承担同一角色的模块必须给出同形状的 spec，用一条测试固化，参考 `test_muon_hybrid_mla_grouping.py::TestIndexerMuonSliceSpecSymmetry::test_same_shape_of_spec_as_csa_indexer` 对 `DSAIndexer` / `CSAIndexer` 的 helper 与 kwargs 相等断言（只有参数名因属性命名不同而不同）。
- 新增模块类时补一条"该类暴露 `muon_slice_specs`"的存在性断言，参考同一文件的 `test_both_classes_expose_the_hook`。这是唯一能在静态层面拦住"漏声明"的手段。
- 涉及 `dist.alltoall` 的 EP 路径需要 `tests/multi_card_tests/` 下的覆盖，单卡测试只能验证形状契约。

只改了 `muon_slice_specs` 而没有任何测试变更的 PR，必须报告测试缺失，并指出应加在上述哪一类断言里。

## 注释与可追溯性

每条非显然的 spec 都要在旁边写清"这张权重为什么这样切"，即打包形态与 forward 用法的对应关系。既有代码已建立这一约定（`linear_o_group_proj` 说明 `[g, r, d]` 的 grouped gemm 用法、`gate_proj.weight` 说明列是 group-major、`v_b_proj` 说明转置与 `muon_version` 缩放的关系、`kv_b_proj` 说明无梯度所以故意不切）。新增 spec 缺少这类说明时报告：切分依据无法从形状反推，缺注释会让后续修改 layout 的人无法判断 spec 是否仍然成立。故意不切的权重同样要写明理由。
