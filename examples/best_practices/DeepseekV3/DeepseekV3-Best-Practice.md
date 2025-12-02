# DeepSeek-V3 全参数微调实践

近期，我们成功组织并完成了 DeepSeek-V3（671B）模型的全参数微调实验。本次实践旨在验证超大规模模型在特定业务场景下的可控性与实际落地能力，同时系统探索全参数微调在性能优化、训练效率提升及资源调度等方面的关键技术路径。以下为我们构建的整体解决方案以及在实践过程中积累的经验与教训总结。

### 项目亮点
* 参考 Hugging Face Transformers 等主流训练框架，补全了训练流程中的全部关键逻辑，包括 Multi-Token Prediction 与 MOE 训练组件，完成整体 modeling 组网的编写。
* 实现了融合 Sharding 并行、PP（流水线并行）、SP（序列并行）、TP（张量并行）与 EP（专家并行）的混合并行训练方案，并引入 subbatch 处理、优化器 offload 等优化技术，在16台机器环境下支持 DeepSeek-V3模型的全量微调，同时可处理128K 长度的长文本输入。

### 快速开始
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

为充分发挥 EP（专家并行）训练优势，推荐使用基于 Hopper 架构的 GPU。

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
| 上下文长度 | 配置文件 |
|------------|----------|
| 4K   | [`dsv3_4k_config.yaml`](./dsv3_4k_config.yaml) |
| 32K  | [`dsv3_32k_config.yaml`](./dsv3_32k_config.yaml) |
| 128K | [`dsv3_128k_config.yaml`](./dsv3_128k_config.yaml) |

##### 启动脚本
| 上下文长度 | 脚本文件 |
|------------|----------|
| 4K   | [`run_dsv3_4k.sh`](./run_dsv3_4k.sh) |
| 32K  | [`run_dsv3_32k.sh`](./run_dsv3_32k.sh) |
| 128K | [`run_dsv3_128k.sh`](./run_dsv3_128k.sh) |

##### 启动命令
```
# 推荐使用mpirun进行多机启动，以4k为例
mpirun bash run_dsv3_4k.sh
```

### 实验效果
##### 实验配置
|机器数|seq_len|sharding|tp|sp|pp|ep|tokens/s/card|数据来源|
|-|-|-|-|-|-|-|-|-|
|16机|4K|16|1|fasle|8|16|203|自测|
|16机|32K|2|8|true|8|16|182|自测|
|16机|128K|2|8|true|8|16|124|自测|

##### 收敛效果：
* 在4K 长度的上下文场景下，100个 step ，loss 收敛效果
<img width="1200" height="750" alt="image" src="https://github.com/user-attachments/assets/29ffe118-2c37-46e0-baeb-9bf45b7097bd" />
* 在32K 长度的上下文场景下，100个 step， loss 收敛效果
<img width="1200" height="750" alt="image" src="https://github.com/user-attachments/assets/ab1e9dc7-1fd5-4824-a27b-5a6edae2c579" />
* 在128K 长度的上下文场景下，100个 step，loss 收敛效果
<img width="1200" height="750" alt="image" src="https://github.com/user-attachments/assets/8470631c-bf76-406c-b462-4c2943eecf08" />


### 实验总结
* 在大规模参数场景下，优化器状态往往无法完全驻留于 GPU 显存，因此需采用 Offload 技术，以内存空间换取显存容量，确保训练任务持续执行。
* 面对长序列输入时，前向计算过程中的激活值峰值随 token 数量急剧上升，极易耗尽显存。此时可引入 Subbatch 方法，通过分段计算以时间换空间，保障训练流程的稳定推进。
* 在 MoE 模型中，专家间负载不均衡也可能引发 OOM 错误。为此，合理引入 AuxLoss 及其无辅助损失机制至关重要。以下是实验过程中总结的关键注意事项：
    * Gate 计算隔离：e_score_correction_bias 应仅用于门控权重计算，避免传递至后续 FFN 模块。
    * AuxLoss 计算适配：在 SP 或 Subbatch 等并行策略下，需注意 seq_len 的实际取值，确保损失计算正确。
    * 配置调整：Hugging Face 所提供的部分配置（如 aux_loss_alpha）需结合具体训练场景进行针对性调优。
