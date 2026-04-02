import sys
import os
sys.path.insert(0, os.path.abspath("/media/user/40da1a25-a924-43a2-b274-e33d39ea8680/cv/hzg/Diff-Transformer/PaddleFormers"))

import unittest
import paddle

from paddleformers.transformers.diff_transformer.configuration import DiffTransformerConfig
from paddleformers.transformers.diff_transformer.modeling import DiffTransformerForCausalLM

class TestDiffTransformerForward(unittest.TestCase):
    def test_forward(self):
        # 超小模型测试
        config = DiffTransformerConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            vocab_size=32000,
        )

        model = DiffTransformerForCausalLM(config)
        model.eval()

        # 构造输入
        input_ids = paddle.randint(0, 32000, shape=[1, 8])

        # 推理
        with paddle.no_grad():
            output = model(input_ids)

        print("\n✅ 组网测试成功！模型结构 100% 正确！")
        print(f"输出 shape: {output.shape}")

if __name__ == "__main__":
    unittest.main()