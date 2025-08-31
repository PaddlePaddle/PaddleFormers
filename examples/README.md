## 精调

### 数据准备

我们支持的精调数据格式是每行包含一个字典的 json 文件，每个字典包含以下字段：

- `src` : `str, List(str)`, 模型的输入指令（instruction）、提示（prompt），模型应该执行的任务。
- `tgt` : `str, List(str)`, 模型的输出。

样例数据：

```text
{"src": "Give three tips for staying healthy.", "tgt": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."}
...
```

为了方便测试，我们也提供了[tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)demo 数据集可以直接使用：

```shell
wget https://bj.bcebos.com/paddlenlp/datasets/examples/alpaca_demo.gz
tar -xvf alpaca_demo.gz
```

### 全参精调：SFT

单卡
```bash
# 需要12G显存左右
python -u run_finetune.py ./config/qwen/sft_argument_qwen2_0p5b.json
```

多卡
```bash
python -u -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" run_finetune.py ./config/qwen/sft_argument_qwen2_0p5b.json
```

### LoRA

LoRA 启动命令参考
```bash
# 需要9G左右显存
python -u run_finetune.py ./config/qwen/lora_argument_qwen2_0p5b.json
```
