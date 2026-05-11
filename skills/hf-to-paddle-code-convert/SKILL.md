---
  name: hf-to-paddle-code-convert
  description: 将 HuggingFace Transformers 中的模型组网代码转换成 PaddleFormers，参考 API 映射，替换相关 API，冷启跑通
  allowed-tools: Bash(mkdir *)
---

  ## Your Task

  将 HuggingFace Transformers 中，$0 的组网代码 `$0/modeling.py`（如果该文件不存在，则扫描transformers库中的组网代码`transformers/src/transformers/$0/modeling_$0.py`） 转换为 PaddleFormers 实现。

  1. 阅读 Transformers 中的代码，了解相关的依赖和实现方式
  2. 参考 API 映射表，将 Transformers 中组网代码的 PyTorch 相关实现，转换为基于 PaddlePaddle 的实现，生成 PaddleFormers 的组网代码，放入`paddleformers/transformers/$0/modeling.py` 中，如目录不存在，可以使用`mkdir -p` 创建
  3. 基于模型配置文件`$0/config.json` ，生成模型配置类`configuration.py`，同样放到模型组网目录中
  4. 如有需要，可生成其他的相关文件
  5. 生成模型及配置类后，注意生成`__init__.py` ，方便后续调用
  6. 在 PaddleFormers 中注册`auto`，方便调用
  7. PaddleFormers 实现中，`$0PretrainedModel`中，需要有两个方法`_gen_aoa_config`和`_gen_inv_aoa_config`，其中：

     1. `_gen_aoa_config`是从safetensors的key到Paddleformers组网key的映射规则
     2. `_gen_inv_aoa_config`是从PaddleFormers组网key到safetensors的key的映射规则

     需要注意的是，`paddle.nn.Linear`和`torch.nn.Linear`算子的`weight`维度是不同的，需要做转置
  8. 生成组网测试代码，能够使用`from_config`实例化模型，配置文件在`$0/config.json`中，生成简单的样例数据，完成模型前向推理

  ## Note

  1. PaddleFormers 中有一些通用接口实现，在`paddleformers.nn`中，例如 `paddleformers.nn.Linear` 等，转换代码的时候注意充分使用它，可以参考 `paddleformers.transformers`中其他模型中的用法
  2. 注意使用 PaddleFormers 中相关的基础接口，如`model_utils`、`configuration_utils`等，可以参考其他组网实现
  3. 充分使用 PaddleFormers 中与 transformers 对应的算法实现，如SDPA、eagar之类的
  4. 要以最终能跑通为标准，如果测试的时候有问题需要及时调整
  5. PaddleFormers 模型里面，attention 和 mlp 需要有 fuse 实现
  7. 测试代码非正式测试用例，生成在当前目录即可

  ## Additional resources

  - PaddlePaddle 和 PyTorch API 的映射规则，可参考[PyTorch 最新 release 与 Paddle develop API 映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html)
  - PaddleFormers 组网的代码风格，以及fuse算子怎么实现，可参考`paddleformers/transformers/`中其他模型的实现方式
  - AOA配置的写法可以参考文档[Flex Checkpoints 用户文档](references/aoa_doc.md)
  - 模型模板说明可见[Chat Template 说明](https://github.com/PaddlePaddle/PaddleFormers/blob/develop/docs/zh/chat_template_guide.md)
