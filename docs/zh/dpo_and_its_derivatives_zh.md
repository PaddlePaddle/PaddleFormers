# DPO及其衍生算法
## 数据集
DPO所需的数据集格式为 $(x,y_c,y_r)$，其中 $x$ 表示模型输入，$y_c,y_r$ 分别表示符合人类偏好的chosen_response和不符合人类偏好的rejected_response。

数据集准备可以参考：  
中文：
- [数据集格式说明及 demo 数据下载](https://github.com/PaddlePaddle/PaddleFormers/blob/develop/docs/zh/datasets_format_zh.md)
- [数据流参数说明](https://github.com/PaddlePaddle/PaddleFormers/blob/develop/docs/zh/datasets_zh.md)

英文：
- [Dataset format description and demo data download](https://github.com/PaddlePaddle/PaddleFormers/blob/develop/docs/en/datasets_format.md)
- [Description of dataset parameters](https://github.com/PaddlePaddle/PaddleFormers/blob/develop/docs/en/datasets.md)

## DPO 
[论文链接arxiv](https://arxiv.org/abs/2305.18290)

DPO是一种直接从偏好数据中学习策略的方法，避免了复杂的强化学习训练过程，具有训练稳定、实现简单的优势。  
对同一上文，DPO训练最大化模型生成chosen数据和rejected数据的概率的差值，从而使模型的回答更符合人类偏好。

### 损失函数

$\mathcal{L}_{\mathrm{DPO}}=-\mathbb{E}_{\left(x, y_{c}, y_{r}\right) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_{\theta}\left(y_{c} \mid x\right)}{\pi_{r e f}\left(y_{c} \mid x\right)}-\beta \log \frac{\pi_{\theta}\left(y_{r} \mid x\right)}{\pi_{ref}\left(y_{r} \mid x\right)}\right)\right] $

### loss_type参数
- `sigmoid`（默认）：标准DPO损失函数，使用sigmoid函数建模偏好概率

### 主要超参数配置

#### 基础参数
- `beta`：KL正则化系数，控制策略模型与参考模型的偏离程度。值越大表示对偏离的惩罚越强。**默认值：0.1**
- `pref_loss_ratio`：偏好损失权重，用于控制DPO损失的比重。**默认值：1.0**
- `sft_loss_ratio`：SFT损失权重，可在训练中混合监督学习损失。**默认值：0.0**

#### 算法特定参数
- `offset_alpha`：基于分数的DPO损失的偏移系数，仅在`sigmoid`损失中生效。**默认值：0.0**

#### 模型配置
- `reference_free`：是否使用参考模型。特定算法会自动设置。**默认值：False**
- `ref_model_update_steps`：参考模型参数更新步数，-1表示不更新。**默认值：-1**
- `ignore_eos_token`：是否忽略EOS token进行计算。**默认值：False**
- `normalize_logps`：是否对log概率进行长度归一化。**默认值：False**
- `label_smoothing`：标签平滑系数，提升训练稳定性。**默认值：0.0**

### 训练配置文件参考

[文件夹位置](https://github.com/PaddlePaddle/PaddleFormers/tree/develop/examples/config/dpo)

## Hinge DPO
Hinge DPO是DPO的一个变种，使用hinge loss来替代原来的sigmoid损失函数。  
通过设置一个边界（margin）‌来确保模型对偏好信号的置信度，当模型对优劣回答的区分度足够大时，就停止优化这部分样本，否则进行线性惩罚。

### 损失函数

$\mathcal{L}_{\mathrm{Hinge DPO}} =\mathbb{E}_{\left(x, y_{c}, y_{r}\right) \sim \mathcal{D}}\left[\mathrm{ReLU}\left(1 - \beta\left(\log \frac{\pi_{\theta}\left(y_{c} \mid x\right)}{\pi_{r e f}\left(y_{c} \mid x\right)} - \log \frac{\pi_{\theta}\left(y_{r} \mid x\right)}{\pi_{r e f}\left(y_{r} \mid x\right)}\right)\right)\right] $

### loss_type参数
- `hinge`：使用hinge loss的变种，当奖励差值<1时进行惩罚

### 算法特点
- 损失函数更简洁，计算效率更高
- 通过hinge机制提供明确的边界约束
- 对较大奖励差值的样本不产生梯度，避免过拟合

## SimPO
[论文链接arxiv](https://arxiv.org/abs/2405.14734)

DPO是最大化偏好对之间的奖励差值，而不是直接最大化绝对奖励值。这可能导致模型找到一个“捷径”——通过极力压低rejected_response的奖励来拉大差距，而非真正提升chosen_response的质量。  
SimPO最具创新性的一点是目标奖励边际。它希望chosen_response的奖励不仅能超过rejected_response，还要超过一个固定的目标值$γ>0$。这直接要求chosen_response具有正的、足够高的绝对奖励，而不仅仅是比rejected_response好。

### 损失函数

$\mathcal{L}_{\mathrm{SimPO}} = -\mathbb{E}_{\left(x, y_{c}, y_{r}\right) \sim \mathcal{D}}\left[\log \sigma\left(\beta\log \frac{\pi_{\theta}(y_c|x)}{\pi_{\theta}(y_r|x)} - \gamma\right)\right] $

### loss_type参数
- `simpo`：SimPO损失函数，使用奖励差值与偏移$\gamma$的比较，仅支持无参考模型

### 关键超参数配置
- `simpo_gamma`：奖励偏移系数$\gamma$，控制奖励阈值。**默认值：0.5**
- `reference_free`：自动设为**True**，不使用参考模型

### 算法特点
- 无需参考模型，训练更高效
- 引入了$\gamma$参数控制奖励阈值
- 更简单的实现，减少了计算复杂度

## IPO
[论文链接arxiv](https://arxiv.org/abs/2310.12036)
 
DPO正则项的强度是随样本变化的、非固定的，缺乏一个强度固定且严格的正则化项来防止策略过度偏离初始模型，从而避免过优化。  
IPO建立在偏好学习的策略梯度理论之上。从损失函数看，优势比被直接约束去接近一个常数 $1/(2β)$。$β$ 是一个固定的超参数，直接明确地控制着策略 $π$ 被允许偏离参考策略 $π_{ref}$ 的程度——$β$ 越大，约束越强，策略越保守；$β$ 越小，约束越弱，策略优化空间越大。

### 损失函数

$\mathcal{L}_{\mathrm{IPO}} = \mathbb{E}_{\left(x, y_{c}, y_{r}\right) \sim \mathcal{D}}\left[\left(\log \frac{\pi_{\theta}(y_c|x)}{\pi_{ref}(y_c|x)} - \log \frac{\pi_{\theta}(y_r|x)}{\pi_{ref}(y_r|x)} - \frac{1}{2\beta}\right)^2\right]$

### loss_type参数
- `ipo`：使用平方损失的DPO变种

### 算法特点
- 提供了统计保证，防止过拟合
- 使用平方损失替代sigmoid损失
- 具有更好的理论性质和泛化能力

## DPOP
[论文链接arxiv](https://arxiv.org/abs/2402.13228)

DPO只强调“chosen > rejected”，却没有强制模型保持对 chosen 响应的生成能力。  
DPOP 在 DPO 的基础上，增加了一个正样本正则化项（Positive Regularization），用于鼓励模型保持对高质量（chosen）响应的生成概率。

### 损失函数

$\mathcal{L}_{\mathrm{DPOP}} = -\log \sigma\left(\beta\left(\log \frac{\pi_{\theta}(y_c|x)}{\pi_{ref}(y_c|x)} - \log \frac{\pi_{\theta}(y_r|x)}{\pi_{ref}(y_r|x)} - \lambda \cdot \mathrm{ReLU}\left(\log \frac{\pi_{ref}\left(y_{c} \mid x\right)}{\pi_{\theta}\left(y_{c} \mid x\right)}\right)\right)\right)$

### loss_type参数
- `dpop`：DPOP损失函数，添加偏好比例正则化

### 关键超参数配置
- `dpop_lambda`：正则化系数$\lambda$，控制正则化强度。**默认值：50**

### 算法特点
- 引入偏好比例正则化项
- 更好地控制策略模型的偏离程度
- 提高了训练的稳定性

## KTO_pair
[论文链接arxiv](https://arxiv.org/abs/2402.01306)

DPO依赖于成对的偏好数据。  
KTO 的核心在于摆脱了对成对数据的依赖，只需要单个样本的“好”或“差”的二元标签：
* 对于一个标记为 “好” 的样本 $(x, y_c)$，希望模型生成它的概率（相对于参考模型）越高越好。但如果它已经比“足够好”的阈值高很多，收益会递减。
* 对于一个标记为 “差” 的样本 $(x, y_r)$，希望模型生成它的概率越低越好。但如果它已经比“足够差”的阈值低很多，再降低它的收益（避免损失）也会递减。

KTO Pair‌，作为KTO的衍生方法，借鉴了DPO的成对偏好数据，但优化目标与标准KTO类似，旨在通过对比学习来对齐模型偏好。

### 损失函数

$\mathcal{L}_{\mathrm{KTO\_pair}} = \mathbb{E}_{\left(x, y_{c}, y_{r}\right) \sim \mathcal{D}}\left[\frac{1}{2}(1 - \sigma(\beta(A - \mathrm{KL}_r))) + \frac{1}{2}(1 - \sigma(\beta(\mathrm{KL}_c - B)))\right] $

其中$A = \log \frac{\pi_{\theta}(y_c|x)}{\pi_{ref}(y_c|x)}$，$B = \log \frac{\pi_{\theta}(y_r|x)}{\pi_{ref}(y_r|x)}$

### loss_type参数
- `kto_pair`：KTO_pair损失函数，基于前景理论

### 算法特点
- 基于人类决策心理学的理论依据
- 对称处理偏好和非偏好样本
- 避免了对单一类型样本的过度优化

## SPPO_hard
[论文链接arxiv](https://arxiv.org/abs/2405.00675)

传统的强化学习（RL）通常依赖于人工标注的偏好数据或奖励模型（Reward Model）来优化策略。  
SPPO 引入自博弈的思想，将语言模型对齐问题表述为一个两玩家博弈问题，其中一个玩家是当前策略模型，另一个玩家是参考策略或者之前的策略。算法通过迭代更新策略来逼近纳什均衡。  
这里直接使用DPO的成对偏好数据集。

### 损失函数

$\mathcal{L}_{\mathrm{SPPO\_hard}} = \mathbb{E}_{\left(x, y_{c}, y_{r}\right) \sim \mathcal{D}}\left[(\log \frac{\pi_{\theta}(y_c|x)}{\pi_{ref}(y_c|x)} - \frac{1}{2\beta})^2 + (\log \frac{\pi_{\theta}(y_r|x)}{\pi_{ref}(y_r|x)} + \frac{1}{2\beta})^2\right]$

### loss_type参数
- `sppo_hard`：简化版SPPO算法的hard版本

### 算法特点
- 基于自博弈思想改进DPO
- 使用两个平方项分别优化偏好和非偏好样本
- 提供了更强的正则化效果

## NCA_pair
NCA pair版本通过类间对比优化特征空间。

### 损失函数

$\mathcal{L}_{\mathrm{NCA\_pair}} = -\mathbb{E}_{\left(x, y_{c}, y_{r}\right) \sim \mathcal{D}}\left[\log \sigma(\beta\log \frac{\pi_{\theta}(y_c|x)}{\pi_{ref}(y_c|x)}) - \frac{1}{2}\log \sigma(-\beta\log \frac{\pi_{\theta}(y_c|x)}{\pi_{ref}(y_c|x)}) - \frac{1}{2}\log \sigma(-\beta\log \frac{\pi_{\theta}(y_r|x)}{\pi_{ref}(y_r|x)})\right] $

### loss_type参数
- `nca_pair`：NCA_pair损失函数，基于对比估计的配对损失

### 算法特点
- 结合了对比学习和偏好优化
- 通过三个损失项综合考虑不同情况
- 增强了模型的判别能力

## ORPO
[论文链接arxiv](https://arxiv.org/abs/2403.07691)

ORPO将指令遵循能力训练和偏好对齐这两个目标合二为一，通过odds ratio来优化偏好学习，无需参考模型，从而简化了训练架构。  

### loss_type参数
- `orpo`：自动映射为 `or` 损失函数，不使用参考模型
- `or`：基于odds ratio的偏好优化损失函数，不使用参考模型

### 损失函数

$\mathcal{L}_{\mathrm{ORPO}} = \mathbb{E}_{\left(x, y_{c}, y_{r}\right) \sim \mathcal{D}}\left[\mathcal{L}_{\mathrm{SFT}} + \lambda \cdot \mathcal{L}_{\mathrm{OR}}\right] $

$\mathcal{L}_{\mathrm{OR}} = -\log \sigma\left(\log \frac{\pi_{\theta}(y_c|x)}{\pi_{\theta}(y_r|x)} - \log \frac{1 - \pi_{\theta}(y_c|x)}{1 - \pi_{\theta}(y_r|x)}\right) $

### 关键超参数配置
- `sft_loss_ratio`：SFT损失权重，`loss_type = orpo`时自动设为**1.0**。
- `reference_free`：自动设为**True**，不使用参考模型

### 算法特点
- 基于odds ratio进行偏好优化
- 无需参考模型，训练效率高
- 提供了概率解释性