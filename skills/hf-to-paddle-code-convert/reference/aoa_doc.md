FlexCheckpoint用户文档

本文档共分为8节，组织如下：

1. Checkpoint介绍。该部分介绍大模型Checkpoint的基本概念，与HuggingFace开源权重格式。
2. FlexCheckpoint主要解决的问题。该部分介绍Checkpoint面临的痛点问题。
3. FlexCheckpoint的原理与设计思路。
4. 用户接口——ShardedStateDict与AOA标记。
5. 如何写ShardedStateDict。
6. 如何写AOA标记。
7. Showcase。
8. 测试。
9. 开关配置。
10. 常见问题与注意事项。

## Checkpoint介绍
Checkpoint系统负责对模型参数、优化器、数据流、随机状态以及训练所需配置信息进行持久化保存，以便在训练任务出现故障中断后可以重新恢复状态（续训）。

### **Checkpoint存储的内容**
* 模型权重（**weights**）：网络各层的参数，**bfloat16**是其最常见的数据类型。
* 优化器状态（optimizer state），对大模型训练而言，主要面向AdamW优化器，主要包括：
    * **moment1 和 moment2**，分别称为一阶动量和二阶动量，数据类型通常为 **float32**。
        * moment1（一阶动量）：记录梯度的指数加权平均值，用于加速收敛和减少振荡。
        * moment2（二阶动量）：记录梯度平方的指数加权平均值，用于自适应调整学习率。

    * **master_weights（主权重）**，数据类型通常为 **float32**。
        * 在混合精度训练（如使用 float16/bfloat16 进行前向和反向计算）时，为了保证数值稳定性，优化器内部会维护一份 float32 精度的权重副本（master weights）。每次参数更新实际作用在 master_weights 上，然后再同步回模型参数。
        * 这是大模型训练（尤其在分布式、混合精度场景下）中非常重要的状态。

    * **bias（偏置）**，数据类型通常为 **float32**。：
        * 在某些实现中，优化器还会维护与偏置相关的状态（如 Adam 算法中的 β1、β2 的偏置校正系数），用于对动量估计做无偏校正。


* 其他信息，如：
    * LR_Schduler状态；
    * 训练步数；
    * rng状态等。


特别的，如果使用bfloat16训练，可以从master_weights中直接cast出模型权重。

在模型训练过程中，系统会按照固定步数周期性地保存上述内容。用户只需指定Checkpoint路径，即可恢复到对应的训练状态，实现断点续训。

### **HuggingFace Checkpoint格式**
以baidu/ERNIE-4.5-21B-A3B-Thinking模型开源权重为例，开源权重格式内容如下：

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=85fb64def6b94afcba0c5b04c0ec8a80&docGuid=F5ky4KwD3V7o_Z)
图一 HuggingFace开源权重内容

**HuggingFace开源权重格式具有如下特点：**

1. 采用 safetensor 作为权重文件的存储格式。
2. 每个 checkpoint 文件中的权重均为未切分的完整参数。
3. 参数未经过融合处理，保持原始结构。
4. Paddle 中 Linear 层的权重与 HuggingFace 格式中的 Linear 层权重存在转置关系。
5. model.safetensors.index.json 文件用于存储 checkpoint 的元信息，记录每个完整 Tensor 所在的具体文件。
6. 权重文件通常采用 model-XXXXX-of-XXXXX.safetensors 的命名方式。



## FlexCheckpoint解决的问题
### **背景**
下面的表格总结了大模型训练各阶段中Checkpoint重切分具体场景：

|**阶段**|**Checkpoint重切分原因**|**具体场景**|
|-|-|-|
|预训练|扩缩卡、训练策略变化导致权重与优化器状态分布变化|热启/恢复训练时每个GPU需获取对应权重和状态；分片保存，张量并行下参数被切分或融合|
|预训练|权重开源需要切换到通用格式|需将分片Checkpoint转为HuggingFace等通用格式，便于下游使用|
|后训练|GPU资源变化、训练任务变化等|微调、指令微调等需按当前GPU重切分Checkpoint|
|后训练|加载开源权重|加载HuggingFace等外部权重时，也需重新切分以适配当前训练环境|
|推理评估|Checkpoint与预训练阶段不同|推理评估所需Checkpoint需按推理环境进行格式和分布调整|

除此之外，在强化学习的Weight Sync阶段，需要将Actor模型中的权重同步给Rollout模型，该阶段虽然不涉及Checkpoint，但是本质上仍是模型状态的重切分。

总结上述场景，Checkpoint 重切分主要有两类原因：一是分布式策略的变化，二是**参数组织方式调整**。

* **分布式策略变化**指同一参数在不同并行策略下的切分方式发生改变，例如训练策略从 TP2 切换到 TP4 时，原本被切分为两份的参数需要重新切分为四份。
* **参数组织方式调整**则指为了提升计算效率或对齐开源标准，对参数进行融合、切分、转置等操作，从而改变了原有的参数组织形式。

如图一所示，在预训练中，通常会将Q,K,V三个权重融合成一个大权重进行运算。

[流程图]
图二 参数组织方式调整举例

### **挑战**
**分布式切分信息无法直接获取**。目前，在Paddle动态图手动并行模式下，每个rank仅关注本地的数据与计算，缺乏全局视角。各 rank 通过通信算子和通信组进行必要的数据交换，实现协同完成分布式训练任务。以张量并行的基础组网类为例：

```python
class ColumnParallelLinear(paddle.nn.Layer):
    def __init__(
        self,
        in_features,
        out_features,
        mp_group=None,
        .......
        ):
        # 直接计算切分后权重的shape
        self.output_size_per_partition = out_features // self.world_size
        # 根据切分后权重的shape创建权重
        self.weight = self.create_parameter(
                shape=[in_features, self.output_size_per_partition],
                attr=self._weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
        .......
    def forward(self, x):
        # 利用本地权重直接计算
        output_parallel = self.linear(
            input_parallel, self.weight, self.bias, name=self._name
        )
        # 和本通信组中的其他rank做通信，得到全局结果，保证计算数学等价
        output = mp_ops._c_concat(
            output_parallel, group=self.model_parallel_group
        )
```
由于每个 rank 只维护本地信息，因此在保存 Checkpoint 时，无法直接确定当前 Checkpoint 中的权重在全局权重中的 offset，这给 Checkpoint 的重切分带来了极大的不便。

**参数组织方式调整。**在 Checkpoint 重切分过程中，常常需要对部分参数进行特殊处理，例如将当前层所有专家参数融合以支持推理、重新排布本层专家参数，或对融合后的 QKV 权重进行拆分等。由于参数重组方式多种多样，Checkpoint 系统难以在代码层难以通用化，因此在大多数实际场景下，通常需要根据具体需求手动编写或定制切分脚本。

## FlexCheckpoint的原理与设计思路
**为解决分布式模型状态切分信息难以直接获取的问题**，FlexCheckpoint引入了ShardedWeight对象，其在Paddle Tensor基础上增加了切分信息属性，便于Checkpoint的重切分。同时，FlexCheckpoint在nn.Layer中新增了sharded_state_dict函数，其返回结果为dict，其中的键和state_dict返回的dict相同。但sharded_state_dict的值为包含切分信息的ShardedWeight对象，而state_dict的值为切分后的权重张量。两者的唯一区别即在于值的类型。获取到**ShardedStateDict**后，模型权重便携带了切分信息。FlexCheckpoint会将这些切分信息与模型权重一同存储，为后续的Checkpoint重切分提供支持。

**为解决参数组织方式调整带来的问题**，FlexCheckpoint设计了一套专用于描述参数组织调整的标记机制，称为**AOA（All in One Arrow）**标记。在参数组织方式调整过程中，Checkpoint的重切分通常依赖人工编写的脚本，这些脚本需要先将分散在不同Checkpoint文件中的所有参数切片拼接还原成完整参数，再进行相关操作。因此，人工脚本逻辑复杂、难以维护且难以通用。AOA标记允许用户以全局视角为参数添加标记，而无需关心参数在多个Checkpoint中如何被切分。在Checkpoint重切分时，FlexCheckpoint会解析用户提供的AOA标记，自动完成参数的映射和重组，简化了参数调整流程。

## 用户接口——ShardedStateDict与AOA标记
### **ShardedStateDict**
如第三节所述，ShardedStateDict中的值类型为ShardedWeight。因此首先介绍ShardedWeight。

ShardedWeight的定义如下所示：

```python
class ShardedWeight:
    """
    Represents a local shard of a distributed tensor parameter.

    Args:
        key (str): The name of the parameter.
        local_tensor (Tensor): The local shard of the parameter.
        local_shape (Tuple[int, ...]): The shape of the local shard.
        global_shape (Tuple[int, ...]): The global logical shape of the parameter.
        global_offset (Tuple[int, ...]): The offset of the local shard in the global parameter.
        is_flattened (bool, optional): Whether the parameter has been flattened (used in sharding_v2 scenarios). Default is False.
        flattened_range (slice, optional): If the parameter is flattened, this indicates the index range of the actual local shard within the local_tensor.
    """

    def __init__(
        self,
        key: str,
        local_tensor: Tensor,
        local_shape: tuple[int, ...],
        global_shape: tuple[int, ...],
        global_offset: tuple[int, ...],
        is_flattened: bool = False,
        flattened_range: slice | None = None,
    ) -> None:
        self.key = key
        self.local_tensor = local_tensor
        self.local_shape = local_shape
        self.global_shape = global_shape
        self.global_offset = global_offset
        self.is_flattened = is_flattened
        self.flattened_range = flattened_range
```
ShardedWeight各属性含义：

* key（str）：参数名称
* local_tensor（Tensor）：本地参数切片
* local_shape（Tuple[int]）：本地切片的shape
* global_shape（Tuple[int]）：参数在全局逻辑上的shape
* global_offset（Tuple[int]）：本地参数切片在全局逻辑上的偏移量
* is_flattened（bool=False）：参数是否被展平，用于sharding_v2场景
* flattened_range（slice）: 参数被展平时，本地实际切片可能小于local_shape大小，flattened_range标记切片展平后在local_tensor中的索引范围

[流程图]
图四 ShardedWeight示意图

图三展示了 ShardedWeight 各属性的具体含义。global_tensor 与 flattened_local_tensor 并非独立存储的实体，而是可以根据 ShardedWeight 的属性（如全局形状、全局偏移等）在逻辑上重构或拼接出来的视图。 local_tensor 表示 ShardedWeight 实际存储的张量数据。  

* **当 is_flattened 为 False 时**，local_shape 和 global_offset 用于确定 local_tensor 在全局张量（global_tensor）中的具体位置。  
* **当 is_flattened 为 True 时**，local_tensor 的形状（shape）不一定等于 local_shape。此时，local_tensor 表示：首先根据 local_shape 和 global_offset 在 global_tensor 上定位到对应的张量切片，然后将该切片展平成一维，再根据 flattened_range 指定的范围，从展平后的tensor中切取出 local_tensor。

sharded_state_dict 函数返回一个以 ShardedWeight 为 value 的字典，其主要作用是在模型权重的基础上附加切分属性。paddle.nn.Layer 作为 Paddle所有神经网络层的基类，默认实现了 sharded_state_dict 方法。该方法首先通过调用 state_dict 获取当前层的权重字典，并将每个权重视为未切分的完整 Tensor，构造相应的 ShardedWeight。随后，会递归调用各个子层的 sharded_state_dict 方法，并合并所有子层返回的 ShardedStateDict。如果用户在自定义 Layer 时对部分权重进行了特殊切分，可以通过重载 sharded_state_dict 方法，自定义这些权重的切分信息。此时，在模型顶层仅需调用 model.sharded_state_dict()，即可获取所有带有切分标记的权重。

```python
def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ) -> ShardedStateDict:
        """Recursively builds a sharded state dictionary for the model and its sub-layers.

        Args:
            structured_name_prefix: Prefix to prepend to all tensor names for hierarchical naming.

        Returns:
            Dictionary mapping tensor names to ShardedWeight.
            The dictionary contains both the current layer's parameters and all sub-layer parameters.
        """
        sharded_state_dict = {}
        # Get current layer's state dict (without sub-layers)
        state_dict = self.state_dict(
            structured_name_prefix="",  # We handle prefixing ourselves
            include_sublayers=False,
        )

        # Convert to sharded state dict
        current_sharded_dict = build_sharded_state_dict(
            state_dict=state_dict,
            shard_rules=None,  # No tensor parallelism rules by default
            prefix=structured_name_prefix,
        )
        sharded_state_dict.update(current_sharded_dict)

        # Recursively process sub-layers
        for layer_name, layer_item in self._sub_layers.items():
            if layer_item is not None:
                sub_sharded = layer_item.sharded_state_dict(
                    structured_name_prefix=f"{structured_name_prefix}{layer_name}.",
                )
                sharded_state_dict.update(sub_sharded)

        return sharded_state_dict

```
需要指出的是，is_flattened 和 flattened_range 这两个属性主要是为 Sharding Stage1 V2 优化器状态设计的。目前，FlexCheckpoint 能够根据 model_state_dict 的切分方式推导出优化器状态的切分信息，自动构造出优化器的ShardedStateDict。因此，在重载 sharded_state_dict 方法、需要构造 ShardedWeight 时，这两个属性可以直接省略，无需显式指定。

### **AOA标记**
AOA 标记主要用于描述参数重组方式，实现从 source sharded_state_dict 到 target sharded_state_dict 的参数映射。

其中：

* source sharded_state_dict 通常指待加载 checkpoint 中存储的参数切分状态；
* target sharded_state_dict 表示当前模型加载 checkpoint 时采用的参数切分状态。

在大多数情况下，source sharded_state_dict 与 target sharded_state_dict 的参数名称是对应且一致的，此时可以省略 AOA 标记，直接按 key 关联加载参数。

AOA 解析模块在处理 AOA 标记时，遵循如下约定：

* **箭头左侧**的变量名如果同时存在于 source sharded_state_dict 和 target sharded_state_dict，则默认操作 source sharded_state_dict 中的 tensor；
* **箭头右侧**的变量名如果同时存在于两者中，则默认操作 target sharded_state_dict 中的 tensor。

为保证操作的明确性，禁止将 target sharded_state_dict 中的 tensor 作为源操作数。

#### 4.2.1 AOA基本操作
AOA标记目前支持**拆分(SPLIT)**、**合并(MERGE)**、**重命名(RENAME)**、**增加(ADD)**、**删除(DELETE)**、**转置(TRANSPOSE)**、**转型(CAST)**基本操作。下面表格分别介绍这几种操作的使用规则。

|**操作**|**格式示例**|**说明与使用规则**|
|-|-|-|
|**拆分**SPLIT|`A -> B1, B2, ..., axis=k`|*  将张量 `A`沿指定轴`axis=k`切分，结果分别赋值给右侧多个 sharded_weight*  左侧只能有一个 sharded_weight，右侧可有多个* `A`在 axis 轴上的长度需能被右侧数量整除* 不在 source/target sharded_state_dict 的名称视为中间变量|
|**合并**MERGE|`A1, A2, ... -> B, axis=k`|* 将多个张量 `A1, A2, ...`沿 axis 轴合并为 `B`* 左侧可有多个，右侧仅有一个 sharded_weight* 除 axis 轴外 shape 必须一致* 不在 source/target sharded_state_dict 的名称视为中间变量|
|**重命名**RENAME|`A -> B`|* 将`A`赋值给`B`两者 global_shape和dtype应相同* 左右两端名称若不在 dict 中，视为中间变量|
|**增加**ADD|`_ -> A`|* 保留 `target sharded_state_dict`中`A`的原始值，不被source同名张量覆盖|
|**删除**DELETE|`A -> _`|* 从`source sharded_state_dict`中删除`A`，不会加载到target中|
|**转置**TRANSPOSE|`A^T -> B`|* 将 `A`转置后赋值给`B`* 左右两侧各只能有一个 sharded_weight* 转置后 global_shape 必须匹配* “^T” 只允许出现在箭头左侧（不允许右侧）|
|**转型**CAST|`A -> B, dtype="float32"`|* 将 `A`转型（目前仅支持 float16、bfloat16、float32 互转后赋值给`B`* 左右两侧各只能有一个sharded_weight* 转型后dtype必须匹配* dtype 目前支持 float16、bfloat16、float32|

**注：**

* 左侧 sharded_weight 名称若存在于 source sharded_state_dict 中，则表示实际的权重；否则表示中间变量。
* 右侧 sharded_weight 名称若存在于 target sharded_state_dict 中，则表示实际的权重；否则表示中间变量。
* 中间变量仅在转换过程中临时使用，不会写入最终的 target sharded_state_dict。

---

#### 4.2.2 AOA宏
在实际使用 AOA 时，常会遇到许多固定的操作模式（Pattern）。为简化操作，FlexCheckpoint 将这些常见模式实现为宏（Macro），能够自动将复杂的操作展开为上表中的基本原语。这一机制大大减少了用户手动编写 AOA 标记的工作量，提高了使用效率和易用性。每个 Macro 都对应一个特殊字符，用于标识该宏。

**AOA宏只会对source sharded_state_dict 中的key进行匹配，然后展开。无法应用到自定义的中间变量。**

 假设在`source sharded_state_dict`中有如下权重：

* `layer.0.moe.expert.0.fused_ffn.weight`
* `layer.1.moe.expert.0.fused_ffn.weight`
* `layer.2.moe.expert.0.fused_ffn.weight`
* `layer.3.moe.expert.0.fused_ffn.weight`

layer.$LAYER_ID.moe.expert.0.fused_ffn.weight -> layer.$LAYER_ID.moe.expert.0.fused_ffn.weight_RENAMED

**会被展开为**

layer.0.moe.expert.0.fused_ffn.weight -> layer.0.moe.expert.0.fused_ffn.weight_RENAMED

layer.1.moe.expert.0.fused_ffn.weight -> layer.1.moe.expert.0.fused_ffn.weight_RENAMED

layer.2.moe.expert.0.fused_ffn.weight -> layer.2.moe.expert.0.fused_ffn.weight_RENAMED

layer.3.moe.expert.0.fused_ffn.weight -> layer.3.moe.expert.0.fused_ffn.weight_RENAMED

**而假如后面有下面的AOA标记**

layer.$LAYER_ID.moe.expert.0.fused_ffn.weight_RENAMED -> layer.$LAYER_ID.moe.ffn

则不会展开，因为layer.$LAYER_ID.moe.expert.0.fused_ffn.weight_RENAMED是中间变量

目前，FlexCheckpoint支持的macro如下：

---

* **star_macro**

**标识符**：`*`

`star_macro`用于简化对具有数字序列命名的变量的匹配和展开操作。通过使用`*`，可以一次性选中所有符合命名模式的变量，无需逐一列出

使用说明

* 当`*`出现在箭头左侧时，会自动匹配`source sharded_state_dict`中所有符合模式的变量（如数字递增的权重名），并按照数字顺序展开。
* 当`*`出现在箭头右侧时，则会对`target sharded_state_dict`匹配到的变量进行展开。

示例

假设在`source sharded_state_dict`中有如下四个专家权重：

* `layer.0.moe.expert.0.fc1.weight`
* `layer.0.moe.expert.1.fc1.weight`
* `layer.0.moe.expert.2.fc1.weight`
* `layer.0.moe.expert.3.fc1.weight`

如果希望将这四个权重在`axis=1`轴上融合，赋值给`layer.0.moe.expert.fc1.fused_weight`，可以采用如下两种方式：

**方式一：逐个列出所有变量**

```
layer.0.moe.expert.0.fc1.weight, layer.0.moe.expert.1.fc1.weight, layer.0.moe.expert.2.fc1.weight, layer.0.moe.expert.3.fc1.weight -> layer.0.moe.expert.fc1.fused_weight, axis = 1

```
**方式二：使用 star_macro 简化写法**

```
layer.0.moe.expert.*.fc1.weight -> layer.0.moe.expert.fc1.fused_weight, axis = 1
```
两种写法在效果上完全等价，但使用`star_macro`可以简化操作。

---

* **fused_qkv_old_macro**

**标识符**：`fused_qkv_old`

`fused_qkv_old_macro`是一个用于简化 fused_qkv 权重排列转换的功能。当模型采用不同的张量并行（Tensor Parallel, TP）策略时，fused_qkv 权重在全局视角下的排列方式会发生变化，尤其是在采用 GQA（分组查询注意力）结构的模型中。手动处理这些排列变化往往十分繁琐，容易出错。fused_qkv_old_macro 提供了一种自动化的方式，帮助用户高效地完成这一过程。
具体来说，假设模型的注意力头（num_heads）数量为 8，并且存在 4 个 key/value 分组（num_key_value_groups=4, fused_qkv这个权重可以拆分出4个key）。在这种情况下，fused_qkv 权重本质上融合了所有的 query（q）、key（k）、value（v）参数，排列顺序会因 TP 数量不同而不同。例如：
当 TP=1 时，所有 query、key、value 通常按如下顺序排列：

```
fused_qkv.weight -> q0,q1,q2,q3,q4,q5,q6,q7,k0,k1,k2,k3,v0,v1,v2,v3, axis = 1
```
在TP=2下，在全局视角下，fused_qkv可以被拆分为：

```
fused_qkv.weight -> q0,q1,q2,q3,k0,k1,v0,v1,q4,q5,q6,q7,k2,k3,v2,v3, axis = 1
```
因此，当分布式策略从TP1转到TP2时，需要配置以下AOA标记。

```
fused_qkv.weight -> q0,q1,q2,q3,q4,q5,q6,q7,k0,k1,k2,k3,v0,v1,v2,v3, axis = 1
q0,q1,q2,q3,k0,k1,v0,v1,q4,q5,q6,q7,k2,k3,v2,v3 -> fused_qkv.weight, axis = 1
```
`fused_qkv_old_macro`对该流程进行了简化，

```
fused_qkv.weight -> fused_qkv.weight, fused_qkv_old, num_heads=8,num_key_value_groups=4
```
这里num_key_value_groups的含义是fused_qkv这个权重可以拆出几个k或者几个v。在执行时，fused_qkv_old_macro 会自动识别源权重的排列方式，并根据目标的张量并行策略，自动完成权重的重排，无需手动指定详细的排列次序。
同时`fused_qkv_old_macro`支持fused_qkv权重和非fused的权重互转。用法如下所示：

```
q,k,v -> fused_qkv.weight, fused_qkv_old, num_heads=8,num_key_value_groups=4
```
```
fused_qkv.weight -> q,k,v, fused_qkv_old, num_heads=8,num_key_value_groups=4
```
`fused_qkv_old_macro`还支持传入axis这个属性，指定下述操作的axis:

```
fused_qkv.weight -> q0,q1,q2,q3,q4,q5,q6,q7,k0,k1,k2,k3,v0,v1,v2,v3, axis = 1
q0,q1,q2,q3,k0,k1,v0,v1,q4,q5,q6,q7,k2,k3,v2,v3 -> fused_qkv.weight, axis = 1
```
一般地，query（q）、key（k）、value（v）投影矩阵参数，axis=1此时可以直接缺省。如果有bias，则对应的bias需要设置axis=0。

---

* **fused_qkv_macro**

**标识符**：`fused_qkv`

`fused_qkv_macro`也是用于实现简化 fused_qkv权重排列转换的功能，用于描述fused_qkv权重排布的另一个pattern。

同样假设模型的注意力头（num_heads）数量为 8，并且存在 4 个 key/value 分组（num_key_value_groups=4, fused_qkv这个权重可以拆分出4个key）。在该场景下，fused_qkv 权重融合了所有的 query（q）、key（k）、value（v）参数，但是排列顺序不会因 TP 数量不同而不同。例如：
当 TP=1 时，所有 query、key、value 通常按如下顺序排列：

```
fused_qkv.weight -> q0,q1,k0,v0,q2,q3,k1,v1,q4,q5,k2,v2,q6,q7,k3,v3, axis = 1
```
在TP=2时，所有query、key、value同样按照如下顺序排列：

```
fused_qkv.weight -> q0,q1,k0,v0,q2,q3,k1,v1,q4,q5,k2,v2,q6,q7,k3,v3, axis = 1
```
因此，假如fused_qkv权重按照上述方式排布，无需为该权重配置AOA标记，因为此排列模式和TP切分相洽。该macro一般用于fused和非fused互转的场景。

```
q,k,v -> fused_qkv.weight, fused_qkv, num_heads=8,num_key_value_groups=4
```
```
fused_qkv.weight -> q,k,v, fused_qkv, num_heads=8,num_key_value_groups=4
```
同样，该macro支持属性axis，表示在axis维度上做权重切分重排。

---

* **array_macro**

**标识符**：`[ ]`

`array_macro`的功能类似于静态数组，可以使用数组与索引描述多个具有相似名称的sharded_weight。例如，

```
fused_qkv.weight -> q0,q1,q2,q3,q4,q5,q6,q7,k0,k1,k2,k3,v0,v1,v2,v3, axis = 1
q0,q1,q2,q3,k0,k1,v0,v1,q4,q5,q6,q7,k2,k3,v2,v3 -> fused_qkv.weight, axis = 1
```
可以利用array_macro写成

```
fused_qkv.weight -> q[0:8],k[0:4],v[0:4], axis = 1
q[0:4],k[0:2]1,v[0:2],q[4:8],k[2:4],v[2:4] -> fused_qkv.weight, axis = 1
```
---

* **fused_ffn_macro**

**标识符**：`fused_ffn`

`fused_ffn_macro`也是用于实现简化 fused_ffn权重排列转换的功能。与fused_qkv类似，将mlp中的gate和up权重融合在一起之后，当模型采用不同的张量并行（Tensor Parallel, TP）策略时，fused_ffn 权重在全局视角下的排列方式会发生变化，fused_ffn_macro提供了在变TP时自动重新切分fused_ffn权重能力。

例如，当TP=1时，fused_ffn中gate，up权重按照如下方式排序：

```
fused_ffn.weight -> gate, up, axis = 1
```
在TP=2下，在全局视角下，fused_ffn可以被拆分为：

```
fused_ffn.weight -> gate0, up0, gate1, up1,  axis = 1
```
因此，当分布式策略从TP1转到TP2时，需要配置以下AOA标记。

```
fused_ffn.weight -> gate0, gate1, up0, up1, axis = 1
gate0, up0, gate1, up1 -> fused_ffn.weight, axis = 1
```
`fused_ffn_macro`对该流程进行了简化，

```
fused_ffn.weight -> fused_ffn.weight, fused_ffn
```
同时`fused_ffn_macro`支持fused权重和非fused的权重互转。用法如下所示：

```
gate, up -> fused_ffn.weight, fused_ffn
```
```
fused_ffn.weight -> gate, up, fused_ffn
```
同样，该macro支持属性axis，表示在axis维度上做权重切分重排。

---

* **layer_id**

**标识符**：`$LAYER_ID`

`layer_id_macro` 支持用 `$LAYER_ID` 占位符，将一条 AOA 标记**批量推广**到 `source sharded_state_dict` 中所有模型层，无需手动为每一层编写重复的标记语句。

使用说明

* 在需要匹配多层结构的权重名称（如 layer.0.*, layer.1.* 等）时，将层号直接替换为 $LAYER_ID。
* AOAEngine会自动将带有 $LAYER_ID 的模板，批量应用到所有实际存在的层号。

示例:

假设在`source sharded_state_dict`中有如下权重：

* `layer.0.moe.expert.0.fused_ffn.weight`
* `layer.1.moe.expert.0.fused_ffn.weight`
* `layer.2.moe.expert.0.fused_ffn.weight`
* `layer.3.moe.expert.0.fused_ffn.weight`

希望给上面四个权重标记fused_ffn。AOA标记示例如下：

```
layer.0.moe.expert.0.fused_ffn.weight -> layer.0.moe.expert.0.fused_ffn.weight, fused_ffn
layer.1.moe.expert.0.fused_ffn.weight -> layer.1.moe.expert.0.fused_ffn.weight, fused_ffn
layer.2.moe.expert.0.fused_ffn.weight -> layer.2.moe.expert.0.fused_ffn.weight, fused_ffn
layer.3.moe.expert.0.fused_ffn.weight -> layer.3.moe.expert.0.fused_ffn.weight, fused_ffn
```
$LAYER_ID简化了这一流程：

```
layer.$LAYER_ID.moe.expert.0.fused_ffn.weight -> layer.$LAYER_ID.moe.expert.0.fused_ffn.weight, fused_ffn
```
只需要将表示layer的数字替换成标识符$LAYER_ID即可。

需要注意的是，如果某个 sharded_weight 的 name 在标记中使用了 $LAYER_ID，那么建议该 AOA 标记中涉及到的所有 sharded_weight 也都使用 $LAYER_ID。

---

* **expert_id**

**标识符**：`$EXPERT_ID`

`expert_id_macro` 支持用 `$EXPERT_ID` 占位符，将一条 AOA 标记**批量推广**到 `source sharded_state_dict` 中某一模型层的所有专家，无需手动为每一个专家编写重复的标记语句。

示例:

假设在`source sharded_state_dict`中有如下权重：

* `layer.0.moe.expert.0.fused_ffn.weight`
* `layer.0.moe.expert.1.fused_ffn.weight`
* `layer.0.moe.expert.2.fused_ffn.weight`
* `layer.0.moe.expert.3.fused_ffn.weight`

希望给上面四个权重标记fused_ffn。AOA标记示例如下：

```
layer.0.moe.expert.0.fused_ffn.weight -> layer.0.moe.expert.0.fused_ffn.weight, fused_ffn
layer.0.moe.expert.1.fused_ffn.weight -> layer.0.moe.expert.1.fused_ffn.weight, fused_ffn
layer.0.moe.expert.2.fused_ffn.weight -> layer.0.moe.expert.2.fused_ffn.weight, fused_ffn
layer.0.moe.expert.3.fused_ffn.weight -> layer.0.moe.expert.3.fused_ffn.weight, fused_ffn
```
$EXPERT_ID简化了这一流程：

```
layer.0.moe.expert.$EXPERT_ID.fused_ffn.weight -> layer.0.moe.expert.$EXPERT_ID.fused_ffn.weight, fused_ffn
```
只需要将表示expert的数字替换成标识符$EXPERT_ID即可。

---



## 如何写ShardedStateDict
目前，在 Paddle 分布式组网的基类中，已经实现了 sharded_state_dict 函数。对于优化器状态的切分信息，用户无需手动提供，FlexCheckpoint 会根据 model_state 的切分信息，自动推断出 optimizer_state 中各权重的切分方式。目前，已实现 sharded_state_dict 函数的组网基类如下：

* ColumnSequenceParallelLinear
* RowSequenceParallelLinear
* RowParallelLinear
* ColumnParallelLinear
* VocabParallelEmbedding

下面以两个Case介绍并总结sharded_state_dict的实现方式。

### **Case 1**
下面以ColumnParallelLinear中实现的sharded_state_dict函数为例，介绍实现细节。

```python
    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):  
        ## 获取state_dict，state_dict的值为未被添加切分标记的类型
        state_dict = self.state_dict(structured_name_prefix="")
        ## 调用paddle.distributed.flex_checkpoint.dcp.sharded_weight.build_sharded_state_dict函数
        ## 构造出 ShardedStateDict
        return build_sharded_state_dict(
            state_dict, {"weight": 1, "bias": 0}, structured_name_prefix
        )
```
build_sharded_state_dict 函数一共接受三个参数：

* state_dict (dict[str, Tensor])：必选参数，原始的参数字典，包含需要切分的张量。
* shard_rules (dict[str, int]，可选)：张量名称到其切分轴的映射关系。如果为 None，则表示所有张量都不进行并行切分。
* prefix (str，可选)：为所有张量的 key 添加的前缀，默认为空字符串。

该函数根据 shard_rules 中的规则，将 state_dict 转换为切分后的参数字典。未在 shard_rules 中指定的张量将被视为非并行切分。返回值是一个新的字典，其中每个张量都被包装为 ShardedWeight 或保持为原始张量。

因为ColumnParallelLinear中，参数weight和bias分别被沿着axis=1和axis=0做了切分，所以shard_relus参数传入了{"weight":1, "bias":0}。

build_sharded_state_dict会根据用户提供的切分规则（shard_rules），自动将参数字典中的每个张量包装为包含切分信息的 ShardedWeight。对于需要切分的参数，函数会首先通过 paddle.distributed.fleet.get_hybrid_communicate_group()`获取默认的 tensor parallel 通信组（tp 通信组），并确定当前进程的 rank 和并行组的总进程数。随后，函数会根据切分轴（如 0 轴），将本地张量视为全局张量在该轴上的一段，并计算本地张量的形状（local_shape）、全局张量的完整形状（global_shape），以及当前进程负责片段的起始位置（global_offset）。所有这些切分信息会被封装进 ShardedWeight，从而准确描述每个张量在分布式环境下的切分方式(该步骤调用shard_weight函数实现)。未在规则中指定切分的参数，则以复制方式（所有进程持有完整张量）进行封装。

```python
def shard_weight(
    key: str,
    weight: Tensor,
    axis: int,
    group: Group,
) -> ShardedWeight:
    """Creates a ShardedWeight by splitting the input tensor along a specified axis.

    Args:
        key: Unique identifier for the tensor.
        weight: The input tensor to be sharded.
        axis: The axis along which to shard the tensor.
        group: The process group used for distributed communication.

    Returns:
        A ShardedWeight representing the local portion of the global tensor.
    """
    if axis < 0 or axis >= len(weight.shape):
        raise ValueError(
            f"Shard axis {axis} is invalid for tensor with shape {weight.shape}"
        )

    # Get hybrid communication group and rank information
    current_rank = group.rank
    world_size = group.nranks

    # Calculate shapes and offsets
    local_shape = weight.shape
    global_shape = deepcopy(local_shape)
    global_shape[axis] = local_shape[axis] * world_size
    global_shape = tuple(global_shape)
    local_shape = tuple(local_shape)
    global_offset = [0] * len(global_shape)
    if world_size > 1:
        global_offset[axis] = current_rank * local_shape[axis]
    global_offset = tuple(global_offset)

    return ShardedWeight(
        key=key,
        local_tensor=weight,
        local_shape=local_shape,
        global_shape=global_shape,
        global_offset=global_offset,
    )


```
**因此，根据用户自定义组网场景，sharded_state_dict函数实现方法可以总结如下，**

|**组网场景**|**切分方式和通信组**|**推荐实现方式（sharded_state_dict）**|**详细说明**|
|-|-|-|-|
|权重未做任何切分|无|**无需实现**|权重数据未做切分，sharded_state_dict 已在基类实现，无需重写。|
|权重已在子层（SubLayer）中完成切分 |复用SubLayer中的切分逻辑|**无需实现**|权重的切分已在各 SubLayer 内部完成，自动复用子层的切分信息，无需在当前Layer中重新实现 sharded_state_dict。|
|权重在非TP通信组上按某一 axis 切分|非TP通信组，指定 axis|建议复用 shard_weight函数，直接构建出 ShardedWeight|传入本地 local_tensor 和通信组，直接生成 ShardedWeight。适用于自定义分组下的权重切分。|
|权重在 TP 通信组上按某一 axis 切分|TP通信组，指定 axis|建议复用 build_sharded_state_dict函数，直接构建出ShardedStateDict|直接调用 build_sharded_state_dict 方法，shard_rules字典，生成 ShardedStateDict。|

### **Case 2**
在混合专家并行（MoE）场景下，Paddle动态图会在每个 Rank 上对本地的专家权重从 0 开始编号。这样会导致从全局视角来看，不同 Rank 上的专家权重 key 出现重复，无法唯一标识全局的每个专家权重。解决方法是可以在sharded_state_dict函数中为专家重新命名。

PaddleFormers/paddleformers/transformers/model_utils.py中的PretrainedModel是预训练模型的基类，该基类中对pp切分的Layer中的权重在LayerID维度上进行了统一命名。代码如下：

```python
    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)

        if self._single_to_pp_mapping is None:
            self._set_pipeline_name_mapping()
        assert len(self._single_to_pp_mapping) > 0, "The pipeline stage must have parameters!"

        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            state_dict[self._pp_to_single_mapping[k]] = v

        return state_dict
```
sharded_state_dict也需要实现同样的重命名逻辑。并且可以在应用_pp_to_single_mapping之后，进行专家重命名。

```python
    def sharded_state_dict(self, *args, **kwargs):
        ## 获取父类的sharded_state_dict
        sharded_state_dict = super().sharded_state_dict(*args, **kwargs)
        ## 应用_set_pipeline_name_mapping 进行全局layer_id重命名
        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()
        for k in list(sharded_state_dict.keys()):
            v = sharded_state_dict.pop(k)
            v.key = self._pp_to_single_mapping[k]
            sharded_state_dict[self._pp_to_single_mapping[k]] = v

        def increment_expert_number(s, increment):
            def replace(match):
                original_number = int(match.group(0))
                new_number = original_number + increment
                return str(new_number)
            return re.sub(r"(?<=experts\.)\d+", replace, s)
        renamed_sharded_state_dict = {}
        ## 获取ShardedWeight中的
        ## 为每个专家重新命名global_expert_id_offset属性，如果存在则为专家重命名
        for k, v in sharded_state_dict.items():
            global_expert_id_offset = getattr(v, "global_expert_id_offset", None)
            if global_expert_id_offset is not None:
                new_key = increment_expert_number(k, global_expert_id_offset)
                v.key = new_key
                delattr(v, "global_expert_id_offset")
                renamed_sharded_state_dict[new_key] = v
            else:
                renamed_sharded_state_dict[k] = v
        return renamed_sharded_state_dict
```
**global_expert_id_offset如何获得？**

例如，在ERNIE模型组网中，examples/pre-training/models/moe/moe_layer.py中的‎MOELayer组网类实现MoE的核心逻辑，可以为其实现sharded_state_dict，在该函数中添加global_expert_id_offset属性。

```python
    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
        ):
        sharded_state_dict = super().sharded_state_dict(structured_name_prefix)
        global_expert_id_offset = self.group.rank * self.num_local_experts
        for k,v in sharded_state_dict.items():
            v.global_expert_id_offset = global_expert_id_offset
            sharded_state_dict[k] = v
        return sharded_state_dict
```
## 如何写AOA标记
### **预训练场景**
在为预训练模型配置AOA时，应重点关注分布式训练策略的调整是否会引起参数切分方式的变化。如果参数切分方式发生变化，应进一步评估在全局视角下是否会导致参数重组。若确实发生参数重组，则需要相应地配置AOA。此外，针对自定义需求，也可以通过组合AOA提供的基本操作和 Macro 实现灵活定制。

在模型训练过程中，如果对部分参数进行了融合操作，那么当张量并行度发生变化时，这些参数在全局视角下的组合方式也会随之改变。这是因为参数融合一般是在每个 rank 上单独进行的。当张量并行度（tensor parallelism degree）发生变化时，参数被切分的份数也随之调整，导致每个 rank 上用于融合的参数切片大小发生变化，进而引发全局范围内参数分片的重组。

[流程图]
图五 变TP时全局视角下的参数分片排布

在预训练场景下，通常支持两种参数融合模式：fused_qkv 和 fused_ffn。如果在训练配置中启用了这两种融合策略，则需要为相应的参数添加 AOA 标记。针对这两种参数重排方式，AOA 都已提供了相应的 Macro，可直接使用，无需自行实现。

需要注意的是，fused_qkv 有两种不同的参数融合方式（详见第四节）。其中一种融合方式与张量并行（TP）的切分方式兼容，此时不需要额外添加 AOA 标记；只有在另一种与 TP 切分不兼容的融合模式下，才需要添加 AOA 标记以确保参数的正确重组。

### **后训练场景**
在后训练（微调）场景下，主要关注如何从 HuggingFace 格式的 checkpoint 加载权重，完成模型微调，并在训练结束后将模型重新保存为 HuggingFace 格式（详见第一节），以便发布和共享。目前，FlexCheckpoint 已支持直接加载 HuggingFace 开源权重。但由于 HuggingFace checkpoint 格式与实际训练过程中模型权重的组织方式存在差异，因此需要额外添加 AOA 标记以确保权重正确匹配。

在后训练完成后，同样需要通过配置 AOA，对参数的组织方式进行调整，将权重保存为 HuggingFace 开源格式（后续将支持直接根据加载 HuggingFace checkpoint 时的 AOA 标记自动推导保存方式，只要提供加载的AOA标记即可）。

**在此场景下，有三个场景需要重点考虑：**

|**需要关注的问题**|**解决方案/对应 AOA 功能**|
|-|-|
|Linear 权重需要转置|为 Linear 权重添加 Transpose AOA 标记，使用 AOA 的 `Transpose`操作|
|未融合的权重需要融合（如 fused_qkv 和 fused_ffn）|使用 AOA 提供的 `fused_qkv`和 `fused_ffn`两个宏|
|权重数据类型不一致需要转换（cast）|使用 AOA 提供的 `cast`功能|

## Showcase
本节以GLM-4.5-Air模型为例，介绍FlexCheckpoint使用的全流程。

### **sharded_state_dict**
在GLM-4.5-Air模型组网中，大部分权重的切分均是采用Paddle提供的分布式组网基类（如ColumnParallelLinear）实现的，这部分参数的切分信息已经在对应组网基类的sharded_state_dict函数中实现。因此无需做特殊处理。

唯一特例是LMHead，该组网类在TP组上自定义了切分逻辑。因此需要为其实现sharded_state_dict函数实现切分信息的补全。由于该参数是在TP通信组上完成的均匀切分，因此推荐可以直接使用build_sharded_state_dict函数，传入shard_rules直接构造出ShardedStateDict。

为LMHead组网类实现的sharded_state_dict实现方式如下：

```python
def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        state_dict = self.state_dict(structured_name_prefix="")
        return build_sharded_state_dict(state_dict, {"weight": 0, "bias": 0}, structured_name_prefix)
```
此外，尽管GLM-4.5-Air模型是MoE架构，但是专家名称是全局编号的，因此不用特殊处理。

### **AOA标记**
* **变分布式策略**

对于GLM-4.5-Air预训练，往往会开启fused_qkv和fused_ffn。因此，需要在json/yaml中配置下面的AOA标记。

```json
aoa_config:{
    "aoa_statements" : [
         "model.layers.$LAYER_ID.mlp.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn",
         "model.layers.$LAYER_ID.mlp.shared_experts.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.shared_experts.up_gate_proj.weight, fused_ffn",
    ]    
}
```
因为GLM-4.5-Air模型的qkv排布方式和TP切分契合，所以这里无需再为fused_qkv权重添加fused_qkv标记。

* **后训练加载/保存成huggingface格式权重**

GLM-4.5-Air后训练中主要涉及从HuggingFace加载权重和将模型保存HuggingFace权重格式的问题。HuggingFace权重格式在第一节已经详细介绍。由于涉及到参数重组，因此需要配置AOA标记。

直接加载HuggingFace格式权重，需要添加的AOA标记如下。

```python
    @classmethod
    def _gen_aoa_config(cls, config: Glm4MoeConfig):
        aoa_config = {
            "aoa_statements": [
                "model.layers.$LAYER_ID.mlp.gate.weight -> model.layers.$LAYER_ID.mlp.gate.weight, dtype='float32'",
                "model.layers.$LAYER_ID.mlp.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.down_proj.weight",
                "model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.layers.$LAYER_ID.self_attn.o_proj.weight",
                "model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight",
                "model.layers.$LAYER_ID.mlp.shared_experts.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.shared_experts.down_proj.weight",
            ]
        }

        # attention qkv
        if not config.fuse_attention_qkv:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> model.layers.$LAYER_ID.self_attn.{x}_proj.weight"
                for x in ("q", "k", "v")
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.self_attn.q_proj.weight^T, model.layers.$LAYER_ID.self_attn.k_proj.weight^T, model.layers.$LAYER_ID.self_attn.v_proj.weight^T -> model.layers.$LAYER_ID.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}",
                f"model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias -> model.layers.$LAYER_ID.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
            ]

        # FFN
        if not config.fuse_attention_ffn:
            aoa_config["aoa_statements"] += (
                [
                    f"model.layers.$LAYER_ID.mlp.{p}_proj.weight^T -> model.layers.$LAYER_ID.mlp.{p}_proj.weight"
                    for p in ("gate", "up")
                ]
                + [
                    f"model.layers.$LAYER_ID.mlp.shared_experts.{p}_proj.weight^T -> model.layers.$LAYER_ID.mlp.shared_experts.{p}_proj.weight"
                    for p in ("gate", "up")
                ]
                + [
                    f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{p}_proj.weight^T -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{p}_proj.weight"
                    for p in ("gate", "up")
                ]
            )
        else:
            aoa_config["aoa_statements"] += [
                "model.layers.$LAYER_ID.mlp.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.up_proj.weight^T -> model.layers.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn",
                "model.layers.$LAYER_ID.mlp.shared_experts.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.shared_experts.up_proj.weight^T -> model.layers.$LAYER_ID.mlp.shared_experts.up_gate_proj.weight, fused_ffn",
                "model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight^T -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_gate_proj.weight, fused_ffn",
            ]

        return aoa_config
```
目前，训练结束后将 checkpoint 直接以 Huggingface 格式（save_pretrained）存储时，需要手动配置 AOA 标记。后续将支持根据 from_pretrained 时配置的 AOA 参数，自动推导并设置相关标记。

```python
aoa_statements = [
            # do cast
            "model.layers.$LAYER_ID.mlp.gate.weight -> model.layers.$LAYER_ID.mlp.gate.weight, dtype='bfloat16'",
            # do transpose
            "model.layers.$LAYER_ID.mlp.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.down_proj.weight",
            "model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.layers.$LAYER_ID.self_attn.o_proj.weight",
            "model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight",
            "model.layers.$LAYER_ID.mlp.shared_experts.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.shared_experts.down_proj.weight",
        ]

        if not config_to_save.fuse_attention_qkv:
            aoa_statements += [
                f"model.layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> model.layers.$LAYER_ID.self_attn.{x}_proj.weight"
                for x in ("q", "k", "v")
            ]
        else:
            aoa_statements += [
                f"model.layers.$LAYER_ID.self_attn.qkv_proj.weight -> model.layers.$LAYER_ID.self_attn.q_proj.weight, model.layers.$LAYER_ID.self_attn.k_proj.weight, model.layers.$LAYER_ID.self_attn.v_proj.weight , fused_qkv, num_heads={config_to_save.num_attention_heads}, num_key_value_groups = {config_to_save.num_key_value_heads}",
                f"model.layers.$LAYER_ID.self_attn.qkv_proj.bias -> model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias , fused_qkv, num_heads={config_to_save.num_attention_heads}, num_key_value_groups = {config_to_save.num_key_value_heads}, axis = 0",
            ]
            aoa_statements += [
                f"model.layers.{layer_id}.self_attn.{x}_proj.weight^T -> model.layers.{layer_id}.self_attn.{x}_proj.weight"
                for layer_id in range(config_to_save.num_hidden_layers)
                for x in ("q", "k", "v")
            ]

        if not config_to_save.fuse_attention_ffn:
            aoa_statements += (
                [
                    f"model.layers.$LAYER_ID.mlp.{y}_proj.weight^T -> model.layers.$LAYER_ID.mlp.{y}_proj.weight"
                    for y in ("gate", "up")
                ]
                + [
                    f"model.layers.$LAYER_ID.mlp.shared_experts.{y}_proj.weight^T -> model.layers.$LAYER_ID.mlp.shared_experts.{y}_proj.weight"
                    for y in ("gate", "up")
                ]
                + [
                    f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{y}_proj.weight^T -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{y}_proj.weight"
                    for y in ("gate", "up")
                ]
            )
        else:
            aoa_statements += [
                "model.layers.0.mlp.up_gate_proj.weight -> model.layers.0.mlp.gate_proj.weight, model.layers.0.mlp.up_proj.weight, fused_ffn",
                "model.layers.0.mlp.gate_proj.weight^T -> model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight^T -> model.layers.0.mlp.up_proj.weight",
                "model.layers.$LAYER_ID.mlp.shared_experts.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.shared_experts.gate_proj.weight, model.layers.$LAYER_ID.mlp.shared_experts.up_proj.weight, fused_ffn",
                "model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight, model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight, fused_ffn",
            ]
            aoa_statements += (
                [
                    f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight^T -> model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight"
                    for layer_id in range(1, config_to_save.num_hidden_layers)
                ]
                + [
                    f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight^T -> model.layers.{layer_id}.mlp.shared_experts.up_proj.weight"
                    for layer_id in range(1, config_to_save.num_hidden_layers)
                ]
                + [
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight^T -> model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight"
                    for layer_id in range(1, config_to_save.num_hidden_layers)
                    for expert_id in range(config_to_save.n_routed_experts)
                ]
                + [
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight^T -> model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight"
                    for layer_id in range(1, config_to_save.num_hidden_layers)
                    for expert_id in range(config_to_save.n_routed_experts)
                ]
            )
        aoa_config = {"aoa_statements": aoa_statements}
```
### **用法对比**
#### 7.3.1 开关变化
FlexCheckpoint接入PaddleFormers之后，为了和UnifiedCheckpoint、ShardingIO兼容，引入两个新的flag。

```yaml
"save_checkpoint_format" : "unified_checkpoint , sharding_io , flex_checkpoint",
"load_checkpoint_format" : "unified_checkpoint , sharding_io , flex_checkpoint",
# 使用 flex_checkpoint
"save_checkpoint_format" : "flex_checkpoint",
"load_checkpoint_format" : "flex_checkpoint",
# 使用 unified checkpoint
"save_checkpoint_format" : "unified_checkpoint",
"load_checkpoint_format" : "unified_checkpoint",
```
另外，FlexCheckpoint自身有两个flag,

**"aoa_config"**，可以在训练配置json/yaml文件中，配置aoa_config。用于描述参数组织变化。

```yaml
aoa_config:{
    "aoa_statements" : [
         "model.layers.$LAYER_ID.mlp.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn",
         "model.layers.$LAYER_ID.mlp.shared_experts.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.shared_experts.up_gate_proj.weight, fused_ffn",
    ]    
}
```
 "**load_via_cpu**",由于FlexCheckpoint在加载时，显存中会同时存在两份数据，有OOM的风险，在OOM时，可以使用该flag，将数据先加载到CPU，需要通信时，按需加载进GPU。

#### 7.3.2 Checkpoint存储结构变化
//预训练

* UnifiedCheckpoint文件格式

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=f026c370ee144020bb54595ddebfaf9f&docGuid=F5ky4KwD3V7o_Z)
* FlexCheckpoint文件存储格式

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=ca94c81b99ed4e9585ead95764ae70f8&docGuid=F5ky4KwD3V7o_Z)
#### 7.3.3 切分标记变化
//以GLM模型为例

* 参数切分信息

Unified Checkpoint

```python
def _get_tensor_parallel_mappings(cls, config: Glm4MoeConfig, is_split=True):
        from ..conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        LAYER_COLWISE = [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
        ]
        FUSE_LAYER_COLWISE = [
            "self_attn.qkv_proj.weight",
        ]

        LAYER_ROWWISE = ["self_attn.o_proj.weight"]

        EXPERT_LAYER_COLWISE = [
            "up_proj.weight",
            "gate_proj.weight",
        ]
        FUSE_EXPERT_LAYER_COLWISE = [
            "up_gate_proj.weight",
        ]

        EXPERT_LAYER_ROWWISE = ["down_proj.weight"]

        BIAS_KEYS = [
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
        ]
        FUSE_BIAS_KEYS = [
            "self_attn.qkv_proj.bias",
        ]

        def make_base_actions():
            actions = {
                "lm_head.weight": partial(fn, is_column=False),
                "model.embed_tokens.weight": partial(fn, is_column=False),
            }
            for layer_idx in range(config.num_hidden_layers):
                if not config.fuse_attention_qkv:
                    actions.update(
                        {
                            f"{cls.base_model_prefix}.layers.{layer_idx}.{k}": partial(fn, is_column=True)
                            for k in LAYER_COLWISE
                        }
                    )
                else:
                    actions.update(
                        {
                            f"{cls.base_model_prefix}.layers.{layer_idx}.{k}": partial(fn, is_column=True)
                            for k in FUSE_LAYER_COLWISE
                        }
                    )

                actions.update(
                    {
                        f"{cls.base_model_prefix}.layers.{layer_idx}.{k}": partial(fn, is_column=False)
                        for k in LAYER_ROWWISE
                    }
                )
                try:
                    moe_group = fleet.get_hybrid_communicate_group().get_expert_parallel_group()
                except:
                    moe_group = None
                expert_parallel_degree = dist.get_world_size(moe_group) if moe_group is not None else 1
                # TODO: merge disable_ffn_model_parallel and expert_parallel_degree
                if expert_parallel_degree <= 1:
                    # # if disable_ffn_model_parallel is True, disable expert layer tp plan
                    # if not config.disable_ffn_model_parallel:
                    if not config.fuse_attention_ffn:
                        actions.update(
                            {
                                f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.experts.{e}.{k}": partial(
                                    fn, is_column=True
                                )
                                for e in range(config.n_routed_experts)
                                for k in EXPERT_LAYER_COLWISE
                            }
                        )
                    else:
                        actions.update(
                            {
                                f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.experts.{e}.{k}": partial(
                                    fn, is_column=True, is_naive_2fuse=True
                                )
                                for e in range(config.n_routed_experts)
                                for k in FUSE_EXPERT_LAYER_COLWISE
                            }
                        )
                    actions.update(
                        {
                            f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.experts.{e}.{k}": partial(
                                fn, is_column=False
                            )
                            for e in range(config.n_routed_experts)
                            for k in EXPERT_LAYER_ROWWISE
                        }
                    )
                actions.update(
                    {
                        f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.{k}": partial(fn, is_column=False)
                        for k in EXPERT_LAYER_ROWWISE
                    }
                )
                if not config.fuse_attention_ffn:
                    actions.update(
                        {
                            f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.{k}": partial(fn, is_column=True)
                            for k in EXPERT_LAYER_COLWISE
                        }
                    )
                else:
                    actions.update(
                        {
                            f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.{k}": partial(
                                fn, is_column=True, is_naive_2fuse=True
                            )
                            for k in FUSE_EXPERT_LAYER_COLWISE
                        }
                    )

                if not config.fuse_attention_ffn:
                    actions.update(
                        {
                            f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.shared_experts.{k}": partial(
                                fn, is_column=True
                            )
                            for k in EXPERT_LAYER_COLWISE
                        }
                    )
                else:
                    actions.update(
                        {
                            f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.shared_experts.{k}": partial(
                                fn, is_column=True, is_naive_2fuse=True
                            )
                            for k in FUSE_EXPERT_LAYER_COLWISE
                        }
                    )
                actions.update(
                    {
                        f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.shared_experts.{k}": partial(
                            fn, is_column=False
                        )
                        for k in EXPERT_LAYER_ROWWISE
                    }
                )
                # bias
                if config.attention_bias:
                    if not config.fuse_attention_qkv:
                        actions.update(
                            {
                                f"{cls.base_model_prefix}.layers.{layer_idx}.{b}": partial(fn, is_column=True)
                                for b in BIAS_KEYS
                            }
                        )
                    else:
                        actions.update(
                            {
                                f"{cls.base_model_prefix}.layers.{layer_idx}.{b}": partial(fn, is_column=True)
                                for b in FUSE_BIAS_KEYS
                            }
                        )
            return actions

        mappings = make_base_actions()
        return mappings
```
FlexCheckpoint

path: paddleformers/nn/lm_head.py

```python
def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        state_dict = self.state_dict(structured_name_prefix="")
        return build_sharded_state_dict(state_dict, {"weight": 0, "bias": 0}, structured_name_prefix)
```
* 参数融合信息

UnifiedCheckpoint

```python
 base_model_prefix = "model"
 _keep_in_fp32_modules = ["mlp.gate.weight", "e_score_correction_bias"]
 transpose_weight_keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
     @classmethod
    def _get_fuse_or_split_param_mappings(cls, config: Glm4MoeConfig, is_fuse=False):
        # return parameter fuse utils
        from ..conversion_utils import split_or_fuse_func

        fn = split_or_fuse_func(is_fuse=is_fuse)

        # last key is fused key, other keys are to be fused.
        fuse_qkv_keys = [
            (
                "layers.0.self_attn.q_proj.weight",
                "layers.0.self_attn.k_proj.weight",
                "layers.0.self_attn.v_proj.weight",
                "layers.0.self_attn.qkv_proj.weight",
            ),
            (
                "layers.0.self_attn.q_proj.bias",
                "layers.0.self_attn.k_proj.bias",
                "layers.0.self_attn.v_proj.bias",
                "layers.0.self_attn.qkv_proj.bias",
            ),
        ]
        fuse_gate_up_keys = [
            (
                "layers.0.mlp.gate_proj.weight",
                "layers.0.mlp.up_proj.weight",
                "layers.0.mlp.up_gate_proj.weight",
            ),
            (
                "layers.0.mlp.experts.0.gate_proj.weight",
                "layers.0.mlp.experts.0.up_proj.weight",
                "layers.0.mlp.experts.0.up_gate_proj.weight",
            ),
            (
                "layers.0.mlp.shared_experts.gate_proj.weight",
                "layers.0.mlp.shared_experts.up_proj.weight",
                "layers.0.mlp.shared_experts.up_gate_proj.weight",
            ),
        ]
        num_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)
        fuse_attention_qkv = getattr(config, "fuse_attention_qkv", False)
        fuse_attention_ffn = getattr(config, "fuse_attention_ffn", False)
        num_experts = getattr(config, "n_routed_experts", 128)

        final_actions = {}
        if is_fuse:
            if fuse_attention_qkv:
                for i in range(config.num_hidden_layers):
                    for fuse_keys in fuse_qkv_keys:
                        keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in fuse_keys])
                        final_actions[keys] = partial(
                            fn, is_qkv=True, num_heads=num_heads, num_key_value_heads=num_key_value_heads
                        )

            if fuse_attention_ffn:
                for i in range(config.num_hidden_layers):
                    for fuse_keys in fuse_gate_up_keys:
                        keys = [key.replace("layers.0.", f"layers.{i}.") for key in fuse_keys]
                        if "experts.0." in keys[0]:
                            for j in range(num_experts):
                                experts_keys = tuple([key.replace("experts.0.", f"experts.{j}.") for key in keys])
                                final_actions[experts_keys] = fn
                        else:
                            experts_keys = tuple(keys)
                            final_actions[experts_keys] = fn

        else:
            if not fuse_attention_qkv:
                for i in range(config.num_hidden_layers):
                    for fuse_keys in fuse_qkv_keys:
                        keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in fuse_keys])
                        final_actions[keys] = partial(
                            fn,
                            split_nums=3,
                            is_qkv=True,
                            num_heads=num_heads,
                            num_key_value_heads=num_key_value_heads,
                        )
            if not fuse_attention_ffn:
                for i in range(config.num_hidden_layers):
                    for fuse_keys in fuse_gate_up_keys:
                        keys = [key.replace("layers.0.", f"layers.{i}.") for key in fuse_keys]
                        if "experts.0." in keys[0]:
                            for j in range(num_experts):
                                experts_keys = tuple([key.replace("experts.0.", f"experts.{j}.") for key in keys])
                                final_actions[experts_keys] = partial(fn, split_nums=2)
                        else:
                            experts_keys = tuple(keys)
                            final_actions[experts_keys] = partial(fn, split_nums=2)
        return final_actions
```
FlexCheckpoint

path: paddleformers/transformers/glm4_moe/modeling.py

```python
    @classmethod
    def _gen_aoa_config(cls, config: Glm4MoeConfig):
        aoa_config = {
            "aoa_statements": [
                "model.layers.$LAYER_ID.mlp.gate.weight -> model.layers.$LAYER_ID.mlp.gate.weight, dtype='float32'",
                "model.layers.$LAYER_ID.mlp.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.down_proj.weight",
                "model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.layers.$LAYER_ID.self_attn.o_proj.weight",
                "model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight",
                "model.layers.$LAYER_ID.mlp.shared_experts.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.shared_experts.down_proj.weight",
            ]
        }

        # attention qkv
        if not config.fuse_attention_qkv:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> model.layers.$LAYER_ID.self_attn.{x}_proj.weight"
                for x in ("q", "k", "v")
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.self_attn.q_proj.weight^T, model.layers.$LAYER_ID.self_attn.k_proj.weight^T, model.layers.$LAYER_ID.self_attn.v_proj.weight^T -> model.layers.$LAYER_ID.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}",
                f"model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias -> model.layers.$LAYER_ID.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
            ]

        # FFN
        if not config.fuse_attention_ffn:
            aoa_config["aoa_statements"] += (
                [
                    f"model.layers.$LAYER_ID.mlp.{p}_proj.weight^T -> model.layers.$LAYER_ID.mlp.{p}_proj.weight"
                    for p in ("gate", "up")
                ]
                + [
                    f"model.layers.$LAYER_ID.mlp.shared_experts.{p}_proj.weight^T -> model.layers.$LAYER_ID.mlp.shared_experts.{p}_proj.weight"
                    for p in ("gate", "up")
                ]
                + [
                    f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{p}_proj.weight^T -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{p}_proj.weight"
                    for p in ("gate", "up")
                ]
            )
        else:
            aoa_config["aoa_statements"] += [
                "model.layers.$LAYER_ID.mlp.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.up_proj.weight^T -> model.layers.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn",
                "model.layers.$LAYER_ID.mlp.shared_experts.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.shared_experts.up_proj.weight^T -> model.layers.$LAYER_ID.mlp.shared_experts.up_gate_proj.weight, fused_ffn",
                "model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight^T -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_gate_proj.weight, fused_ffn",
            ]

        return aoa_config
```
## 测试
### **功能测试**
* **分布式策略转换**

//该项测试主要测试变分布式策略情形下（含fused策略），Flexcheckpoint自动重切分能力的正确性。

|**模型**||**dp**|**sharding（v1）**|**sharding（v2）**|**tp**|**pp**|**tp + pp**|**sd + tp**|
|-|-|-|-|-|-|-|-|-|
|**Llama2**|dp|✅|✅|✅|✅|✅|✅|✅|
||sharding（v1）|✅|✅|✅|✅|✅|✅|✅|
||sharding（v2）|✅|✅|✅|✅|✅|✅|✅|
||tp|✅|✅|✅|✅|✅|✅|✅|
||pp|✅|✅|✅|✅|✅|✅|✅|
||tp + pp|✅|✅|✅|✅|✅|✅|✅|
|**EB4.5 纯文**|dp|✅|✅|✅|✅|✅|✅|✅|
||sharding（v1）|✅|✅|✅|✅|✅|✅|✅|
||sharding（v2）|✅|✅|✅|✅|✅|✅|✅|
||tp|✅|✅|✅|✅|✅|✅|✅|
||pp|✅|✅|✅|✅|✅|✅|✅|
||sharding|✅|✅|✅|✅|✅|✅|✅|
||sharding + tp|✅|✅|✅|✅|✅|✅|✅|
||tp + pp|✅|✅|✅|✅|✅|✅|✅|

* **全流程验证**

//该项测试主要测试FlexCheckpoint加载与保存HuggingFace格式Checkpoint的能力，以及变分布式策略下，自动重切分能力。

|**模型**|**load from huggingface**|**reshard**|**save to huggingface**|**LoRA**|
|-|-|-|-|-|
|**GLM4.5-Air**|✅|✅|✅|⏳|

* **模型结构转换**

//该项测试主要测试使用AOA标记实现自定义Checkpoint映射能力

|**模型**|**转换方式**||
|-|-|-|
|**EB4.5纯文**|fused_qkv ↔ 非 fused / fused ffn ↔ 非 fused|✅|
||专家合并|✅|
||专家拆分|✅|
||参数重新命名|✅|
||层删除|✅|
||层添加|✅|

## 开关配置
|**配置**|**含义**|
|-|-|
|**resume_from_checkpoint**|指明待加载的ckpt路径|
|**load_from_hf**|热启saftensor权重，该功能需关闭same_data,开启ignore_load_lr_and_optim，并将load_checkpoint_format设置为flex_checkpoint将resume_from_checkpoint的路径指定为HuggingFace格式的权重路径|
|**load_checkpoint_format**|指明待加载的ckpt的格式，可以选择‘sharding_io’或‘flex_checkpoint’。指定‘flex_checkpoint’即可启用flexcheckpoint 加载|
|**save_checkpoint_format**|指明待存储的ckpt格式，可以选择‘sharding_io’或‘flex_checkpoint’。指定‘flex_checkpoint’即可启用flexcheckpoint 存储|
|**load_sharded_model: True**|使用Sharding IO 加载ckpt，等价于load_checkpoint_format设置为sharding_io，但该开关的优先级低于load_checkpoint_format|
|**save_sharded_model: True**|使用Sharding IO 存储ckpt，等价于save_checkpoint_format设置为sharding_io，但该开关的优先级低于save_checkpoint_format|
|**ignore_load_lr_and_optim: False**|加载ckpt是否要忽略优化器状态和学习率|
|**sharded_model_from_ema: False**|是否将存储的ema状态加载进模型|
|**save_hf_steps: -1**|每间隔多少步存储一次HuggingFace格式权重，该开关是否工作和用户指定使用的Checkpoint格式无关。在开启ema时，指定步数下，还会存储ema_huggingface权重|
|**aoa_config**|格式为dict，目前仅支持“aoa_statements”这一字段，aoa_statements字段下可以配置一个列表，每个列表项是一个aoa标记语句。例如：aoa_config:    aoa_statements: [        "_ -> lm.head",        "- -> embedding",    ]|

## 常见问题与注意事项
* **多组网模型适配**

一个模型包含多套组网，以GLM为例，在modeling文件中，向外暴露了以下组网，

```json
__all__ = ["Glm4MoeForCausalLMPipe", "Glm4MoeModel", "Glm4MoeForCausalLM"]
```
Glm4MoeModel是基础模型。Glm4MoeForCausalLM在基础模型的基础上添加了预测头，是完整的因果语言模型，一般在此模型上适配sharded_state_dict和AOA标记。Glm4MoeForCausalLMPipe是Glm4MoeForCausalLM的流水线实现版本，他的切分标记可以直接复用Glm4MoeForCausalLM，但是生成aoa的_gen_aoa_config和_gen_inv_aoa_config虽然和Glm4MoeForCausalLM相同，但均需手动添加。

```json
class Glm4MoeForCausalLMPipe(GeneralModelForCausalLMPipe):
    config_class = Glm4MoeConfig
    _decoder_layer_cls = Glm4MoeDecoderLayer
    _decoder_layer_pipe_cls = Glm4MoeDecoderLayerPipe
    _get_tensor_parallel_mappings = Glm4MoeModel._get_tensor_parallel_mappings
    _get_fuse_or_split_param_mappings = Glm4MoeModel._get_fuse_or_split_param_mappings
    _init_weights = Glm4MoeModel._init_weights
    _keep_in_fp32_modules = Glm4MoeModel._keep_in_fp32_modules
    _tied_weights_keys = ["lm_head.weight"]
    transpose_weight_keys = Glm4MoeModel.transpose_weight_keys
    _rotary_emb_cls = Glm4MoeRotaryEmbedding
    _gen_aoa_config = Glm4MoeForCausalLM._gen_aoa_config
    _gen_inv_aoa_config = Glm4MoeForCausalLM._gen_inv_aoa_config

```
* **版本依赖**

FlexCheckpoint版本依赖：**paddleformers v0.4以上； paddle v3.2.2以上**。

paddle版本过低常见报错有：

1. 执行from_pretrained函数时找不到.distcp文件

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=1ba69e4e5e214a34976c64915bfe3295&docGuid=F5ky4KwD3V7o_Z)
2. 执行from_pretrained函数时读去safetensors报错Place(gpu:XX)不合法

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=9cb1c0f761c0451288b95468356fbb95&docGuid=F5ky4KwD3V7o_Z)
* **AOA编写错误**

一般的，报错栈文件在flex_checkpoint/aoa文件夹下，一般是aoa标记添加错误导致。

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=04627043293f4cb9989c435fc5ab304f&docGuid=F5ky4KwD3V7o_Z)
* **fused_qkv macro在使用时为什么指定了axis=0？**

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=330d9451f11947f6b525897261f2b26a&docGuid=F5ky4KwD3V7o_Z)
如果不指定默认是1，对于qkv和ffn，他们的weight都是在axis=1上做重切分。但是对于bias而言，因为只有一个维度，就是axis=0，因此需要特殊传入。否则在aoa engine解析aoa标记时会直接报错。

* **num_key_value_groups = config.num_attention_heads // config.num_key_value_heads，是不是只需要传kv groups就行了，就可以算出q k v 的size？**

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=2f0a08146d814eab86f43294fc566a25&docGuid=F5ky4KwD3V7o_Z)
还需要传入注意力头数。aoa语句里面num_key_value_groups目前就是我们理解的config.num_key_value_heads。这么写是因为，之前适配模型时有的模型num_key_value_groups直接等于了config.num_key_value_heads，有的num_key_value_groups = config.num_attention_heads // config.num_key_value_heads。后面aoa会修改回num_key_value_heads。

* 考虑 tie_word_embeddings为true的情形么，如果为true时，权重没有lm head，model中有lm head，fc load时会报错么？

会报错。因为现在想要加载的ckpt权重里面没有lm_head，fc会认为权重缺失。解决方案可以将embedding权重转置后加载进lm_head，也可以利用AOA的ADD原语，  _ -> lm_head。

在save时，如果不想保存 lm_head可以使用AOA的删除原语 lm_head -> _。

key miss之后优化报错。优化报错信息。简化aoa，