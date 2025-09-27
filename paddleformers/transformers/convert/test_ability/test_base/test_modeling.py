"""模型组网正确性验证
【基本流程】

定义原模型，加载权重，固定seed，基于numpy生成随机数，转换为PyTorch可以处理的tensor，送入网络，获取输出。

定义模块化转换后modeling模型，加载权重，固定seed，基于numpy生成随机数，转换为PaddlePaddle可以处理的tensor，送入网络，获取输出。

排查diff，小于阈值，即可完成自测。
"""
import numpy as np
import paddle
from paddleformers.transformers.qwen2 import Qwen2Config
from paddleformers.transformers.qwen2.modeling import Qwen2ForCausalLM
from paddleformers.transformers import Qwen2Config as Qwen2Config_hf
from paddleformers.transformers import Qwen2ForCausalLM as Qwen2ForCausalLM_hf
#from paddleformers.transformers.qwen2.test_model_expanded import Qwen2ForCausalLM as Qwen2ForCausalLM_hf



def eval_model_convert():
    paddle_input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
    torch_input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])

    # paddle model
    paddle_ckpt_path = "Qwen/Qwen2-0.5B"
    config_paddle = Qwen2Config.from_pretrained(paddle_ckpt_path)
    model_paddle = Qwen2ForCausalLM.from_pretrained(paddle_ckpt_path, config=config_paddle, dtype="float32")

    # torch model
    
    torch_ckpt_path = "Qwen/Qwen2-0.5B"
    config_torch = Qwen2Config_hf.from_pretrained(torch_ckpt_path)
    config_torch.dtype = "float32"
    model_torch = Qwen2ForCausalLM_hf.from_pretrained(torch_ckpt_path, config=config_torch, dtype="float32")

    model_paddle.eval()
    model_torch.eval()

    out_paddle = model_paddle(paddle_input_ids)[0]
    out_torch = model_torch(torch_input_ids, return_dict=False)[0]
    print(out_paddle)
    print(out_torch)
    assert np.allclose(out_paddle.numpy(), out_torch.detach().numpy(), rtol=1e-5, atol=1e-3)

eval_model_convert()