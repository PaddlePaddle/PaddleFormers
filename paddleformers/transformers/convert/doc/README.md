

## 使用方法：

### 1构建modular__**.py文件

#### 1.1分析基础模型

​	在开始构建 `modular_xxx.py`文件之前，首先需要深入分析要基于的**基础模型**。这个基础模型通常是 paddle/ Transformers 库中已有的成熟模型。

##### 1.1.1 选择合适的基础模型

**选择标准：**

- **架构相似性**：新模型与基础模型的架构应尽可能相似
- **任务类型**：基础模型应支持相同的任务（如文本生成、分类等）
- **代码质量**：选择代码结构清晰、文档完善的模型

**常见基础模型选择：**

```
# 基于BERT架构的模型
基础模型：BertModel, RobertaModel, DebertaModel

# 基于GPT架构的模型  
基础模型：GPT2Model, LlamaModel, GPTNeoXModel

# 基于Encoder-Decoder架构的模型
基础模型：T5Model, BartModel, PegasusModel
```

##### 1.1.2 分析基础模型的关键组件

对于选定的基础模型，需要分析其核心组件：

###### **1. 配置文件 (`configuration_xxx.py`)**

```
# 分析配置参数
# 关注：hidden_size, num_attention_heads, num_hidden_layers, 
#       vocab_size, max_position_embeddings 等关键参数
```

###### **2. 模型架构 (`modeling_xxx.py`)**

```
# 分析模型类结构
import inspect
from transformers import BertModel

# 查看类的方法和属性
print(inspect.getmembers(BertModel, predicate=inspect.ismethod))
# 重点关注：__init__, forward, 以及其他关键方法
```

#####  1.1.3 识别需要修改的部分

基于分析结果，确定哪些部分需要自定义：

| 组件             | 是否需要修改 | 修改原因                   |
| :--------------- | :----------- | :------------------------- |
| **配置参数**     | ✅ 通常需要   | 调整模型尺寸、注意力头数等 |
| **前向传播逻辑** | ✅ 通常需要   | 适配新的架构变化           |
| **注意力机制**   | ⚠️ 可能需要   | 如果使用不同的注意力机制   |
| **位置编码**     | ⚠️ 可能需要   | 如果使用不同的位置编码方案 |
| **输出头**       | ✅ 通常需要   | 适配不同的任务需求         |
| **初始化方法**   | ⚠️ 可能需要   | 如果使用不同的初始化策略   |

#### 1.2编写modular文件结构

​	在完成基础模型分析后，您需要创建一个结构清晰、符合规范的 `modular_xxx.py`文件。这个文件是代码生成器的模板，其结构直接决定了最终输出的 `modeling_xxx.py`文件的质量。

##### 1.2.1 文件基本结构

一个标准的 `modular_xxx.py`文件应包含以下部分，按顺序排列：

```
# coding=utf-8
# 版权声明 (可选)
""" 新模型的简要文档字符串 (可选) """

# 1. 导入部分
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

# 从基础模型导入必要的组件
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
    LlamaDecoderLayer,
    # ... 其他需要继承或引用的组件
)
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.utils import logging


logger = logging.get_logger(__name__)

# 3. 注意力机制 (如果需要自定义)
class MyNewAttention(nn.Module):
    """自定义注意力机制"""
    def __init__(self, config: MyNewModelConfig):
        super().__init__()
        # 实现自定义注意力逻辑
        pass
    
    def forward(self, hidden_states, attention_mask=None):
        # 实现前向传播
        pass


# 4. 解码器层 (如果需要修改层结构)
class MyNewDecoderLayer(LlamaDecoderLayer):
    """
    自定义解码器层，继承自LlamaDecoderLayer
    重写需要修改的方法
    """
    def __init__(self, config: MyNewModelConfig):
        super().__init__(config)
        # 替换或修改注意力机制
        if config.use_custom_attention:
            self.self_attn = MyNewAttention(config)
    
    def forward(self, hidden_states, attention_mask=None):
        # 可以完全重写或部分修改父类逻辑
        if self.config.use_custom_attention:
            # 自定义逻辑
            return self._custom_forward(hidden_states, attention_mask)
        else:
            # 回退到父类逻辑
            return super().forward(hidden_states, attention_mask)
    
    def _custom_forward(self, hidden_states, attention_mask):
        """自定义前向传播实现"""
        pass


# 5. 主模型类
class MyNewModel(LlamaModel):
    """
    我的新模型主类，继承自LlamaModel
    通常需要重写 __init__ 和 forward 方法
    """
    def __init__(self, config: MyNewModelConfig):
        super().__init__(config)
        # 替换解码器层
        self.layers = nn.ModuleList([
            MyNewDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        # 其他自定义初始化
        self.custom_layer = nn.Linear(config.hidden_size, config.custom_param)
    
    def forward(self, input_ids, attention_mask=None):
        # 调用父类获取基础输出
        super().forward(input_ids, attention_mask)
        
        # 添加自定义处理
        hidden_states = outputs[0]
        custom_output = self.custom_layer(hidden_states)
        
        # 返回修改后的输出
        return (custom_output,) + outputs[1:]


# 6. 任务特定模型 (如用于因果语言建模)
class MyNewForCausalLM(LlamaForCausalLM):
    """
    用于因果语言建模的我的新模型
    """
    def __init__(self, config: MyNewModelConfig):
        super().__init__(config)
        # 替换主模型
        self.model = MyNewModel(config)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # 可以完全重写或扩展父类逻辑
        outputs = self.model(input_ids, attention_mask=attention_mask)
        
        # 计算损失等
        loss = None
        if labels is not None:
            # 计算损失逻辑
            pass
            
        return {"loss": loss, "logits": outputs[0]}


# 8. 更新 __all__ 列表，声明哪些类应该被导出
__all__ = [
    "MyNewModelConfig",
    "MyNewModel", 
    "MyNewForCausalLM",
    "MyNewDecoderLayer",
]
```

##### 1.2.2 关键编写原则

**清晰的继承关系**

```
# ✅ 正确：明确继承关系
class MyNewModel(LlamaModel):
    pass

# ❌ 避免：直接继承过于通用的基类
class MyNewModel(PreTrainedModel):
    pass  # 这会导致需要实现大量抽象方法
```

**最小化重写**

```
# ✅ 正确：只重写需要修改的方法
class MyNewDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config):
        super().__init__(config)  # 先调用父类初始化
        # 只修改需要定制的部分
        if config.use_custom_attention:
            self.self_attn = CustomAttention(config)

# ❌ 避免：完全重写整个类，除非必要
```

**保持接口一致性**：

```
def forward(self, input_ids, attention_mask=None, **kwargs):
    # 处理自定义逻辑
    result = custom_processing(input_ids)
    # 调用父类实现剩余逻辑
    super().forward(result, attention_mask, **kwargs)
```

**充分利用现有组件**：

```
# ✅ 正确：复用基础模型的组件
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)
```

### 2.**执行转换命令**

通过一个用户主脚本main来驱动整个流程。其标准使用方式如下：

```
#自动查找各个模型文件下的modular__**.py模块化构建代码，执行转换生成modeling__***.py文件
python main.py
```

###  **自动化处理流水线**

<img src="../images/process.png" style="zoom:33%;" />

### 最终输出

最终，在模型对应的目录下（如 `src/transformers/models/qwen2/`）会生成目标文件：

- **`modeling_qwen2.py`**：**这是唯一的输出文件，也是最终成果。** 它包含了：**模型架构**（如 `Qwen2Model`, `Qwen2ForCausalLM`）**内联的配置类**（如 `Qwen2Config`）**所有相关的函数、工具类和常量****正确的导入语句**（只导入标准库或 transformers 的通用组件）**文件顶部的警告注释**：明确告知开发者此文件为自动生成，不可手动编辑。