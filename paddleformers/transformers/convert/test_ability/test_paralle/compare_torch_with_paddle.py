import numpy as np
import paddle
from paddle.distributed import fleet
from paddleformers.transformers.qwen2 import Qwen2Config
from paddleformers.transformers.qwen2.modeling import Qwen2ForCausalLM
from paddleformers.transformers import Qwen2Config as Qwen2Config_hf
from paddleformers.transformers import Qwen2ForCausalLM as Qwen2ForCausalLM_hf

def eval_model_convert_parallel(mp_degree=1):
    paddle_input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
    torch_input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])

    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": mp_degree,
        "pp_degree": 1,
        "sharding_degree": 1,
    }
    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()

    # paddle model
    paddle_ckpt_path = "Qwen/Qwen2-0.5B"
    config_paddle = Qwen2Config.from_pretrained(paddle_ckpt_path)
    config_paddle.tensor_parallel_degree = hcg.get_model_parallel_world_size()
    config_paddle.tensor_parallel_rank = hcg.get_model_parallel_rank()
    config_paddle.tensor_parallel_output = False
    model_paddle = Qwen2ForCausalLM.from_pretrained(paddle_ckpt_path, config=config_paddle, dtype="float32")

    # torch model
    torch_ckpt_path = "Qwen/Qwen2-0.5B"
    config_torch = Qwen2Config_hf.from_pretrained(torch_ckpt_path)
    config_torch = Qwen2Config.from_pretrained(paddle_ckpt_path)
    config_torch.tensor_parallel_degree = hcg.get_model_parallel_world_size()
    config_torch.tensor_parallel_rank = hcg.get_model_parallel_rank()
    config_torch.tensor_parallel_output = False
    model_torch = Qwen2ForCausalLM_hf.from_pretrained(torch_ckpt_path, config=config_torch,  dtype="float32")

    model_paddle.eval()
    model_torch.eval()

    # 手动验证
    out_paddle = model_paddle(paddle_input_ids)[0]
    out_torch = model_torch(torch_input_ids)[0]
    print(out_paddle)
    print(out_torch)
    assert np.allclose(out_paddle.numpy(), out_torch.detach().numpy(), rtol=1e-5, atol=1e-4)

eval_model_convert_parallel(mp_degree=2)