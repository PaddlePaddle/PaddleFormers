// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda.h>         // NOLINT
#include <cuda_runtime.h> // NOLINT

#include <limits> // NOLINT
#include <vector> // NOLINT

#include <cub/cub.cuh>        // NOLINT
#include <paddle/extension.h> // NOLINT

__global__ void count_valid_kernel(const int64_t *indices, int *valid_count,
                                   const int64_t total_elements) {
  for (int64_t i =
           static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
           static_cast<int64_t>(threadIdx.x);
       i < total_elements; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    if (indices[i] != -1) {
      atomicAdd(valid_count, 1);
    }
  }
}

__global__ void create_mask_kernel(const int64_t *indices, int *mask,
                                   const int64_t total_elements) {
  for (int64_t i =
           static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
           static_cast<int64_t>(threadIdx.x);
       i < total_elements; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    mask[i] = (indices[i] != -1) ? 1 : 0;
  }
}

template <typename scalar_t>
__global__ void
scatter_scores_kernel(const scalar_t *probs, const int64_t *indices,
                      const int *write_indices, scalar_t *output_scores,
                      const int64_t total_elements) {
  for (int64_t i =
           static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
           static_cast<int64_t>(threadIdx.x);
       i < total_elements; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    if (indices[i] != -1) {
      int write_idx = write_indices[i];
      output_scores[write_idx] = probs[i];
    }
  }
}

template <typename scalar_t>
__global__ void
scatter_grad_kernel(const scalar_t *grad_topk, const int64_t *indices,
                    const int *write_indices, scalar_t *grad_probs,
                    const int64_t total_elements) {
  for (int64_t i =
           static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
           static_cast<int64_t>(threadIdx.x);
       i < total_elements; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    if (indices[i] != -1) {
      int write_idx = write_indices[i];
      grad_probs[i] = grad_topk[write_idx];
    }
  }
}

void count_valid_cuda_launcher(const int64_t *indices, int *valid_count,
                               const int64_t total_elements,
                               cudaStream_t stream) {
  if (total_elements == 0)
    return;
  int block_size = 256;
  int64_t grid_size64 = (total_elements + block_size - 1) / block_size;
  int grid_size = static_cast<int>(std::min<int64_t>(grid_size64, 4096));
  count_valid_kernel<<<grid_size, block_size, 0, stream>>>(indices, valid_count,
                                                           total_elements);
}

std::vector<paddle::Tensor> FilterScoresGPU(const paddle::Tensor &probs,
                                            const paddle::Tensor &indices) {
  PD_CHECK(probs.place().GetType() == phi::AllocationType::GPU,
           "probs must be a GPU tensor.");
  PD_CHECK(indices.place().GetType() == phi::AllocationType::GPU,
           "indices must be a GPU tensor.");
  PD_CHECK(probs.shape() == indices.shape(),
           "probs and indices must have the same shape.");
  PD_CHECK(indices.dtype() == paddle::DataType::INT64,
           "indices must be of type int64.");
  cudaStream_t stream = probs.stream();
  const int64_t total_elements = probs.numel();
  if (total_elements == 0) {
    return {paddle::empty({0}, probs.dtype(), probs.place())};
  }
  PD_CHECK(total_elements <=
               static_cast<int64_t>(std::numeric_limits<int>::max()),
           "total_elements exceeds INT_MAX in filter_scores.");
  auto valid_count_tensor =
      paddle::full({1}, 0, paddle::DataType::INT32, probs.place());
  count_valid_cuda_launcher(indices.data<int64_t>(),
                            valid_count_tensor.data<int>(), total_elements,
                            stream);
  int total_valid = 0;
  PD_CHECK(cudaMemcpyAsync(&total_valid, valid_count_tensor.data<int>(),
                           sizeof(int), cudaMemcpyDeviceToHost,
                           stream) == cudaSuccess,
           "cudaMemcpyAsync total_valid_experts failed.");
  PD_CHECK(cudaStreamSynchronize(stream) == cudaSuccess,
           "cudaStreamSynchronize failed for total_valid_experts.");
  if (total_valid == 0) {
    return {paddle::empty({0}, probs.dtype(), probs.place())};
  }
  auto topk_scores = paddle::empty({total_valid}, probs.dtype(), probs.place());
  auto mask =
      paddle::empty({total_elements}, paddle::DataType::INT32, probs.place());
  auto write_indices =
      paddle::empty({total_elements}, paddle::DataType::INT32, probs.place());
  int block_size = 256;
  int64_t grid_size64 = (total_elements + block_size - 1) / block_size;
  int grid_size = static_cast<int>(std::min<int64_t>(grid_size64, 4096));
  create_mask_kernel<<<grid_size, block_size, 0, stream>>>(
      indices.data<int64_t>(), mask.data<int>(), total_elements);
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                mask.data<int>(), write_indices.data<int>(),
                                total_elements);
  auto temp_storage = paddle::empty({static_cast<int64_t>(temp_storage_bytes)},
                                    paddle::DataType::UINT8, probs.place());
  d_temp_storage = temp_storage.data<uint8_t>();
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                mask.data<int>(), write_indices.data<int>(),
                                total_elements, stream);

  PD_DISPATCH_FLOATING_TYPES(
      probs.dtype(), "scatter_scores_kernel", ([&] {
        scatter_scores_kernel<data_t><<<grid_size, block_size, 0, stream>>>(
            probs.data<data_t>(), indices.data<int64_t>(),
            write_indices.data<int>(), topk_scores.data<data_t>(),
            total_elements);
      }));
  return {topk_scores};
}

std::vector<paddle::Tensor>
FilterScoresGradGPU(const paddle::Tensor &indices,
                    const paddle::Tensor &grad_topk_scores) {
  PD_CHECK(indices.place().GetType() == phi::AllocationType::GPU,
           "indices must be a GPU tensor.");
  PD_CHECK(grad_topk_scores.place().GetType() == phi::AllocationType::GPU,
           "grad_topk_scores must be a GPU tensor.");
  PD_CHECK(indices.dtype() == paddle::DataType::INT64,
           "indices must be of type int64.");
  cudaStream_t stream = indices.stream();
  const int64_t total_elements = indices.numel();
  PD_CHECK(total_elements <=
               static_cast<int64_t>(std::numeric_limits<int>::max()),
           "total_elements exceeds INT_MAX in filter_scores_grad.");
  auto grad_probs = paddle::full(indices.shape(), 0, grad_topk_scores.dtype(),
                                 grad_topk_scores.place());
  const int64_t total_valid = grad_topk_scores.numel();
  PD_CHECK(total_valid <= static_cast<int64_t>(std::numeric_limits<int>::max()),
           "total_valid exceeds INT_MAX in filter_scores_grad.");
  if (total_elements == 0 || total_valid == 0) {
    return {grad_probs};
  }
  auto mask = paddle::empty({total_elements}, paddle::DataType::INT32,
                            grad_topk_scores.place());
  auto write_indices = paddle::empty({total_elements}, paddle::DataType::INT32,
                                     grad_topk_scores.place());
  int block_size = 256;
  int64_t grid_size64 = (total_elements + block_size - 1) / block_size;
  int grid_size = static_cast<int>(std::min<int64_t>(grid_size64, 4096));
  create_mask_kernel<<<grid_size, block_size, 0, stream>>>(
      indices.data<int64_t>(), mask.data<int>(), total_elements);
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                mask.data<int>(), write_indices.data<int>(),
                                total_elements);
  auto temp_storage =
      paddle::empty({static_cast<int64_t>(temp_storage_bytes)},
                    paddle::DataType::UINT8, grad_topk_scores.place());
  d_temp_storage = temp_storage.data<uint8_t>();
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                mask.data<int>(), write_indices.data<int>(),
                                total_elements, stream);
  PD_DISPATCH_FLOATING_TYPES(
      grad_topk_scores.dtype(), "scatter_grad_kernel", ([&] {
        scatter_grad_kernel<data_t><<<grid_size, block_size, 0, stream>>>(
            grad_topk_scores.data<data_t>(), indices.data<int64_t>(),
            write_indices.data<int>(), grad_probs.data<data_t>(),
            total_elements);
      }));
  return {grad_probs};
}

PD_BUILD_OP(filter_scores)
    .Inputs({"Probs", "Indices"})
    .Outputs({"TopkScores"})
    .SetKernelFn(PD_KERNEL(FilterScoresGPU));

PD_BUILD_OP(filter_scores_grad)
    .Inputs({"Indices", "TopkScoresGrad"})
    .Outputs({"ProbsGrad"})
    .SetKernelFn(PD_KERNEL(FilterScoresGradGPU));
