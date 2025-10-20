## 权重转换描述

gpt-oss 官方权重包含 MXFP4数据类型，在进行后训练时需要先将权重转换为 bf16，可以通过 change_wieght_dtype.py 中的 fp4_to_bf16方法进行转换，同样的训练后得到 bf16权重也可以通过 bf16_to_fp4方法进行转换。


示例：
```python
tempdir = "./models/gpt-oss"

# fp4_to_bf16
load_path = os.path.join(tempdir, "gpt-oss-test-fp4")
save_path = os.path.join(tempdir, "gpt-oss-test-new-bf16")
fp4_to_bf16(load_path, save_path)

# bf16_to_fp4
load_path = os.path.join(tempdir, "gpt-oss-test-bf16")
save_path = os.path.join(tempdir, "gpt-oss-test-new-fp4")
bf16_to_fp4(load_path, save_path)

```
### 依赖
PaddleFormers 跟目录下执行
```bash
pip install -r requirements.txt
```

### 使用说明

当前转换脚本只支持单卡转换，按照原始文件个数依次转换。
例如：
load_path 下有3个 model-0000x-of-00003.safetensors.
save_path 下会得到3个 model-0000x-of-00003.safetensors 和一个 model.safetensors.index.json

修改 change_weight_dtype.py 中的 load_path 和 save_path 后，执行

```bash
python change_wieght_dtype.py
```
即可
