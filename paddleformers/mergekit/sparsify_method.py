# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import paddle


class SparsifyMethod:
    def __init__(self, merge_config):
        self.merge_config = merge_config

    def sparsify(self, tensor):
        if self.merge_config.sparsify_type is None:
            return tensor
        if self.merge_config.sparsify_type == "dare":
            return self.dare(tensor)
        elif self.merge_config.sparsify_type == "magprune":
            return self.magprune(tensor)
        elif self.merge_config.sparsify_type == "trim":
            return self.trim(tensor)
        else:
            raise ValueError(f"Unknown sparsify method: {self.merge_config.sparsify_type}")

    def dare(self, tensor):
        if self.merge_config.tensor_type == "np":
            tensor *= (np.random.rand(*tensor.shape) < self.merge_config.reserve_p).astype(tensor.dtype)
            if self.merge_config.rescale:
                tensor /= self.merge_config.reserve_p
            return tensor
        elif self.merge_config.tensor_type == "pd":
            mode = "upscale_in_train" if self.merge_config.rescale else "downscale_in_infer"
            tensor = paddle.nn.functional.dropout(tensor, p=1 - self.merge_config.reserve_p, mode=mode, training=True)
            return tensor
        else:
            raise ValueError(f"Unknown tensor type {self.merge_config.tensor_type}")

    def magprune(self, tensor):
        if self.merge_config.tensor_type == "np":
            if not np.any(tensor != 0):
                return tensor
            drop_p = 1 - self.merge_config.reserve_p
            # 1: ranking(descending)
            abs_tensor = np.abs(tensor)
            sorted_indices = np.argsort(-abs_tensor.flatten())

            # 2: caclculate drop rate p_i
            probs = np.empty_like(sorted_indices)
            probs[sorted_indices] = np.arange(tensor.size).astype(tensor.dtype)
            probs = probs.reshape(tensor.shape)  # r_i ∈ {0，1，... ,n}
            probs = probs * self.merge_config.epsilon / tensor.size  # Δ_i =  ε/n * r_i
            p_min = drop_p - self.merge_config.epsilon / 2  # minimal drop rate
            probs += p_min  # p_i for each parameter

            # 3: drop parameters according to their probabilities
            mask = np.random.binomial(1, probs)
            tensor *= (1 - mask).astype(tensor.dtype)
            if self.merge_config.rescale:
                tensor /= 1 - probs
            return tensor
        elif self.merge_config.tensor_type == "pd":
            if not paddle.any(tensor != 0):
                return tensor
            drop_p = 1 - self.merge_config.reserve_p
            abs_tensor = paddle.abs(tensor)
            sorted_indices = paddle.argsort(-abs_tensor.flatten())

            probs = paddle.zeros_like(sorted_indices, dtype="float32")
            probs = paddle.scatter(probs, sorted_indices, paddle.arange(tensor.numel(), dtype="float32"))
            probs = probs.reshape(tensor.shape)
            probs = probs * self.merge_config.epsilon / tensor.numel()
            p_min = drop_p - self.merge_config.epsilon / 2
            probs += p_min
            mask = paddle.bernoulli(1 - probs).astype(tensor.dtype)
            tensor *= mask
            if self.merge_config.rescale:
                tensor /= 1 - probs
            return tensor
        else:
            raise ValueError(f"Unknown tensor type {self.merge_config.tensor_type}")

    def trim(self, tensor):
        if self.merge_config.tensor_type == "np":
            shape = tensor.shape
            tensor = tensor.flatten()
            abs_tensor = np.abs(tensor)
            threshold = np.quantile(abs_tensor, 1 - self.merge_config.reserve_p)

            if self.merge_config.rescale:
                org_sum = np.sum(np.abs(tensor))
                tensor[abs_tensor < threshold] = 0
                new_sum = np.sum(np.abs(tensor))
                if org_sum >= 1e-8 and new_sum >= 1e-8:
                    tensor *= org_sum / new_sum
            else:
                tensor[abs_tensor < threshold] = 0
            return tensor.reshape(shape)
        elif self.merge_config.tensor_type == "pd":
            abs_tensor = paddle.abs(tensor)
            threshold = paddle.quantile(abs_tensor, 1 - self.merge_config.reserve_p)
            tensor = paddle.where(abs_tensor < threshold, paddle.zeros_like(tensor), tensor)
            if self.merge_config.rescale:
                org_sum = paddle.sum(abs_tensor)
                new_sum = paddle.sum(paddle.abs(tensor))
                if org_sum >= 1e-8 and new_sum >= 1e-8:
                    tensor *= org_sum / new_sum
            return tensor
        else:
            raise ValueError(f"Unknown tensor type {self.merge_config.tensor_type}")
