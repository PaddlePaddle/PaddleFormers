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


class MergeMethod:
    def __init__(self, merge_config, sparsify_method=None):
        self.merge_config = merge_config
        self.sparsify_method = sparsify_method

    def merge(self, tensor_list):
        if self.sparsify_method is not None:
            tensor_list = [self.sparsify_method.sparsify(tensor) for tensor in tensor_list]
        if self.merge_config.merge_type == "linear":
            return self.linear(tensor_list)
        elif self.merge_config.merge_type == "slerp":
            return self.slerp(tensor_list)
        elif self.merge_config.merge_type == "ties":
            return self.ties(tensor_list)
        else:
            raise NotImplementedError(f"{self.merge_config.merge_type} is not supported yet.")

    def linear(self, tensor_list):
        """
        Linear interpolation between multiple values.
        """
        # init weight
        weight_list = self.merge_config.weight_list
        if self.merge_config.normalize:
            weight_sum = sum(weight_list)
            weight_list = [weight / weight_sum for weight in weight_list]

        # merge
        if self.merge_config.tensor_type == "np":
            tensor_output = sum(weight * tensor for weight, tensor in zip(weight_list, tensor_list))
            return tensor_output
        elif self.merge_config.tensor_type == "pd":
            tensor_output = paddle.zeros_like(tensor_list[0])
            for i, tensor in enumerate(tensor_list):
                tensor_output += tensor * weight_list[i]
            return tensor_output
        else:
            raise ValueError(f"Unknown tensor type {self.merge_config.tensor_type}")

    def slerp(self, tensor_list):
        """
        Spherical linear interpolation
        """
        # check tensor_list length
        if len(tensor_list) != 2:
            raise ValueError("Slerp only support two tensors merge.")

        if self.merge_config.tensor_type == "np":
            t0, t1 = tensor_list
            # Copy the vectors to reuse them later
            t0_copy = np.copy(t0)
            t1_copy = np.copy(t1)

            # Normalize the vectors to get the directions and angles
            t0 = self.normalize(t0)
            t1 = self.normalize(t1)

            # Dot product with the normalized vectors (can't use np.dot in W)
            dot = np.sum(t0 * t1)
            # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
            if np.abs(dot) > self.merge_config.slerp_dot_threshold:
                return (1 - self.merge_config.slerp_alpha) * t0_copy + self.merge_config.slerp_alpha * t1_copy

            # Calculate initial angle between t0 and t1
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)

            # Angle at timestep t
            theta_t = theta_0 * self.merge_config.slerp_alpha
            sin_theta_t = np.sin(theta_t)

            # Finish the slerp algorithm
            s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0

            return s0 * t0_copy + s1 * t1_copy
        elif self.merge_config.tensor_type == "pd":
            t0, t1 = tensor_list
            # Copy the tensors to reuse them later
            t0_copy = t0.clone()
            t1_copy = t1.clone()

            # Normalize the tensors to get the directions and angles
            t0 = self.normalize(t0)
            t1 = self.normalize(t1)

            # Dot product with the normalized tensors
            dot = paddle.sum(t0 * t1)
            # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
            if paddle.abs(dot) > self.merge_config.slerp_dot_threshold:
                return (1 - self.merge_config.slerp_alpha) * t0_copy + self.merge_config.slerp_alpha * t1_copy

            # Calculate initial angle between t0 and t1
            theta_0 = paddle.acos(dot)
            sin_theta_0 = paddle.sin(theta_0)

            # Angle at timestep t
            theta_t = theta_0 * self.merge_config.slerp_alpha
            sin_theta_t = paddle.sin(theta_t)

            # Finish the slerp algorithm
            s0 = paddle.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0

            return s0 * t0_copy + s1 * t1_copy
        else:
            raise ValueError(f"Unknown tensor type {self.merge_config.tensor_type}")

    def ties(self, tensor_list):
        if self.merge_config.tensor_type == "np":
            # Get weight tensor
            mask_dtype = tensor_list[0].dtype
            weight_list = self.merge_config.weight_list
            tensor_list = [weight * tensor for (weight, tensor) in zip(weight_list, tensor_list)]
            # Elect majority sign
            sign_tensor_list = [np.sign(tensor).astype(mask_dtype) for tensor in tensor_list]
            if self.merge_config.ties_elect_type == "sum":
                majority_sign = (np.sum(tensor_list, axis=0) >= 0).astype(mask_dtype) * 2 - 1
            elif self.merge_config.ties_elect_type == "count":
                majority_sign = (np.sum(sign_tensor_list, axis=0) >= 0).astype(mask_dtype) * 2 - 1
            else:
                raise NotImplementedError(f"ties_elect_type: {self.merge_config.ties_elect_type} is unknown.")

            # Merge
            mask_list = [sign_tensor == majority_sign for sign_tensor in sign_tensor_list]
            tensor_list = [mask * tensor for mask, tensor in zip(mask_list, tensor_list)]
            merge_tensor = np.sum(tensor_list, axis=0)

            # Normalize
            if self.merge_config.normalize:
                weight_mask = [mask * weight for mask, weight in zip(mask_list, weight_list)]
                divisor = np.sum(weight_mask, axis=0)
                divisor[np.abs(divisor) < 1e-8] = 1
                merge_tensor /= divisor
            return merge_tensor

        elif self.merge_config.tensor_type == "pd":
            mask_dtype = tensor_list[0].dtype

            # Elect majority sign
            majority_sign = paddle.zeros_like(tensor_list[0])
            for i, tensor in enumerate(tensor_list):
                if self.merge_config.ties_elect_type == "sum":
                    majority_sign += tensor * self.merge_config.weight_list[i]
                elif self.merge_config.ties_elect_type == "count":
                    majority_sign += tensor.sign()
                else:
                    raise NotImplementedError(f"ties_elect_type: {self.merge_config.ties_elect_type} is unknown.")
            majority_sign = (majority_sign >= 0).astype(mask_dtype) * 2 - 1

            # Merge
            merge_tensor = paddle.zeros_like(tensor_list[0])
            if self.merge_config.normalize:
                divisor = paddle.zeros_like(tensor_list[0])
            for i, tensor in enumerate(tensor_list):
                if self.merge_config.normalize:
                    mask = (tensor.sign() == majority_sign).astype(mask_dtype) * self.merge_config.weight_list[i]
                    divisor += mask
                    merge_tensor += mask * tensor
                else:
                    merge_tensor += (
                        (tensor.sign() == majority_sign).astype(mask_dtype) * tensor * self.merge_config.weight_list[i]
                    )

            # Normalize
            if self.merge_config.normalize:
                divisor = paddle.where(paddle.abs(divisor) < 1e-8, paddle.ones_like(divisor), divisor)
                merge_tensor /= divisor

            return merge_tensor
        else:
            raise ValueError(f"Unknown tensor type {self.merge_config.tensor_type}")

    def normalize(self, t):
        """
        Normalize a vector by its L2 norm.
        """
        if self.merge_config.tensor_type == "np":
            norm_t = np.linalg.norm(t)
            if norm_t > self.merge_config.slerp_normalize_eps:
                t = t / norm_t
            return t
        elif self.merge_config.tensor_type == "pd":
            norm_t = paddle.norm(t, p=2)
            if norm_t > self.merge_config.slerp_normalize_eps:
                t = t / norm_t
            return t
        else:
            raise ValueError(f"Unknown tensor type {self.merge_config.tensor_type}")
