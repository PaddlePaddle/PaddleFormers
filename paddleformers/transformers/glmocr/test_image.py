from .image_processor import Glm46VImageProcessor
from PIL import Image

processor = Glm46VImageProcessor()
img = Image.open("/home/work/zkx_test/glmocr/handwritten.png")
result = processor.preprocess(img, return_tensors="pd")
print(result["pixel_values"].shape)   # [N_patches, C*tp*ps*ps]
print(result["image_grid_thw"].shape) # [1, 3]

import os
import paddle
import numpy as np
def _to_np_pd(v):
    if v is None:
        return None
    if isinstance(v, paddle.Tensor):
        v = v.detach()
        if v.dtype == paddle.bfloat16:
            v = v.astype("float32")
        return v.cpu().numpy()
    return v  # 可能是 numpy 或 python 标量

def save_any_pd(save_dir: str, name: str, v):
    os.makedirs(save_dir, exist_ok=True)

    # 1) list/tuple：逐个保存
    if isinstance(v, (list, tuple)):
        # 额外保存一个“长度”，方便对齐
        np.save(os.path.join(save_dir, f"{name}_len.npy"), np.array([len(v)], dtype=np.int64))
        for i, item in enumerate(v):
            arr = _to_np_pd(item)
            if arr is None:
                continue
            np.save(os.path.join(save_dir, f"{name}_{i:03d}.npy"), arr)
        return

    # 2) 单个 tensor/array
    arr = _to_np_pd(v)
    if arr is None:
        # 保存一个标记文件，避免你以为没跑到
        np.save(os.path.join(save_dir, f"{name}_is_none.npy"), np.array([1], dtype=np.int8))
        return
    np.save(os.path.join(save_dir, f"{name}.npy"), arr)

save_any_pd("/home/work/zkx_test/glmocr_debug", "pd_image_grid_thw", result["image_grid_thw"])
save_any_pd("/home/work/zkx_test/glmocr_debug", "pd_pixel_values", result["pixel_values"])




