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

#include <cuda_runtime.h>
#include <paddle/extension.h>
#include <cub/cub.cuh>
#include <limits>

__global__ void simple_arange_kernel(int* output, int64_t N) {
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < N;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    output[idx] = static_cast<int>(idx);
  }
}

template <typename T>
__global__ void CountValidExpertsKernel(const T* topk_router_indices,
                                        int* num_activated_per_token,
                                        int num_tokens,
                                        int K) {
  int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (token_idx < num_tokens) {
    int count = 0;
    for (int k = 0; k < K; ++k) {
      if (topk_router_indices[token_idx * K + k] > -1) {
        count++;
      }
    }
    num_activated_per_token[token_idx] = count;
  }
}

template <typename T>
__global__ void PopulateValidInfoKernel(const T* topk_router_indices,
                                        const int* num_activated_offset,
                                        T* valid_expert_ids,
                                        int* original_token_indices,
                                        int num_tokens,
                                        int K) {
  int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (token_idx < num_tokens) {
    int base_offset = num_activated_offset[token_idx];
    int current_offset = 0;
    for (int k = 0; k < K; ++k) {
      T expert_id = topk_router_indices[token_idx * K + k];
      if (expert_id > -1) {
        int flat_idx = base_offset + current_offset;
        valid_expert_ids[flat_idx] = expert_id;
        original_token_indices[flat_idx] = token_idx;
        current_offset++;
      }
    }
  }
}

__global__ void GenerateReverseScatterKernel(const int* s_scatter_idx_valid,
                                             int* s_reverse_scatter_idx_valid,
                                             int total_valid_experts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_valid_experts) {
    s_reverse_scatter_idx_valid[s_scatter_idx_valid[idx]] = idx;
  }
}

__global__ void ComputeXGatherIdxKernel(const int* s_scatter_idx_all,
                                        int* x_gather_idx_all,
                                        int K,
                                        int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    x_gather_idx_all[idx] = s_scatter_idx_all[idx] / K;
  }
}

cudaError_t launch_simple_arange(int* output,
                                 int64_t N,
                                 cudaStream_t stream = nullptr) {
  if (N <= 0) return cudaSuccess;
  const int threads_per_block = 256;
  const int max_blocks = 1024;
  const int needed_blocks =
      static_cast<int>((N + threads_per_block - 1) / threads_per_block);
  const int blocks = (needed_blocks < max_blocks) ? needed_blocks : max_blocks;

  simple_arange_kernel<<<blocks, threads_per_block, 0, stream>>>(output, N);
  return cudaGetLastError();
}

template <typename T>
std::vector<paddle::Tensor> RouterMetadataCuda(
    const paddle::Tensor& topk_router_indices,
    const paddle::Tensor& expert_frequency_offset,
    int K) {
  int64_t num_tokens = topk_router_indices.shape()[0];
  int64_t num_experts = expert_frequency_offset.shape()[0];
  PADDLE_ENFORCE_LE(
      num_tokens * K,
      static_cast<int64_t>(std::numeric_limits<int>::max()),
      common::errors::InvalidArgument(
          "num_tokens * K must be <= INT_MAX for router_metadata kernels."));

  const int total_elements = static_cast<int>(num_tokens * K);
  auto place = topk_router_indices.place();
  cudaStream_t stream = topk_router_indices.stream();

  auto padded_expert_frequency_offset = paddle::full(
      {num_experts + 1}, 0, expert_frequency_offset.dtype(), place);
  padded_expert_frequency_offset.slice(1, num_experts + 1)
      .copy_(expert_frequency_offset, place, false);

  if (topk_router_indices.numel() == 0) {
    auto empty_indices = paddle::empty({0}, paddle::DataType::INT32, place);
    auto num_activated_per_token_offset =
        paddle::full({num_tokens + 1}, 0, paddle::DataType::INT32, place);
    return {padded_expert_frequency_offset,
            empty_indices,
            empty_indices,
            empty_indices,
            num_activated_per_token_offset};
  }

  auto num_activated_per_token =
      paddle::empty({num_tokens}, paddle::DataType::INT32, place);
  auto num_activated_per_token_offset =
      paddle::empty({num_tokens + 1}, paddle::DataType::INT32, place);
  int threads = 256;
  int blocks = static_cast<int>((num_tokens + threads - 1) / threads);

  if (num_tokens > 0) {
    CountValidExpertsKernel<T>
        <<<blocks, threads, 0, stream>>>(topk_router_indices.template data<T>(),
                                         num_activated_per_token.data<int>(),
                                         num_tokens,
                                         K);
    PD_CHECK(cudaGetLastError() == cudaSuccess,
             "CountValidExpertsKernel failed.");
  }

  void* scan_temp_storage = nullptr;
  size_t scan_temp_storage_bytes = 0;
  if (num_tokens > 0) {
    cub::DeviceScan::InclusiveSum(
        nullptr,
        scan_temp_storage_bytes,
        num_activated_per_token.data<int>(),
        num_activated_per_token_offset.slice(1, num_tokens + 1).data<int>(),
        num_tokens,
        stream);

    auto scan_temp_storage_tensor =
        paddle::empty({static_cast<int64_t>(scan_temp_storage_bytes)},
                      paddle::DataType::UINT8,
                      place);
    scan_temp_storage = scan_temp_storage_tensor.data();

    PD_CHECK(cudaMemsetAsync(num_activated_per_token_offset.data<int>(),
                             0,
                             sizeof(int),
                             stream) == cudaSuccess,
             "cudaMemsetAsync failed.");

    cub::DeviceScan::InclusiveSum(
        scan_temp_storage,
        scan_temp_storage_bytes,
        num_activated_per_token.data<int>(),
        num_activated_per_token_offset.slice(1, num_tokens + 1).data<int>(),
        num_tokens,
        stream);
    PD_CHECK(cudaGetLastError() == cudaSuccess,
             "cub::DeviceScan::InclusiveSum failed.");
  }

  int total_valid_experts = 0;
  if (num_tokens > 0) {
    PD_CHECK(cudaMemcpyAsync(&total_valid_experts,
                             num_activated_per_token_offset
                                 .slice(num_tokens, num_tokens + 1)
                                 .data<int>(),
                             sizeof(int),
                             cudaMemcpyDeviceToHost,
                             stream) == cudaSuccess,
             "cudaMemcpyAsync total_valid_experts failed.");
  }
  PD_CHECK(cudaStreamSynchronize(stream) == cudaSuccess,
           "cudaStreamSynchronize failed for total_valid_experts.");

  if (total_valid_experts == 0) {
    auto empty_indices = paddle::empty({0}, paddle::DataType::INT32, place);
    return {padded_expert_frequency_offset,
            empty_indices,
            empty_indices,
            empty_indices,
            num_activated_per_token_offset};
  }

  auto valid_expert_ids =
      paddle::empty({total_valid_experts}, topk_router_indices.dtype(), place);
  auto original_token_indices =
      paddle::empty({total_valid_experts}, paddle::DataType::INT32, place);

  PopulateValidInfoKernel<T><<<blocks, threads, 0, stream>>>(
      topk_router_indices.template data<T>(),
      num_activated_per_token_offset.data<int>(),
      valid_expert_ids.template data<T>(),
      original_token_indices.data<int>(),
      num_tokens,
      K);
  PD_CHECK(cudaGetLastError() == cudaSuccess,
           "PopulateValidInfoKernel failed.");

  auto original_flat_indices_valid =
      paddle::empty({total_valid_experts}, paddle::DataType::INT32, place);
  PD_CHECK(launch_simple_arange(original_flat_indices_valid.data<int>(),
                                total_valid_experts,
                                stream) == cudaSuccess,
           "launch_simple_arange for valid failed.");
  auto sorted_valid_expert_ids = paddle::empty_like(valid_expert_ids);
  auto s_scatter_idx_valid =
      paddle::empty({total_valid_experts}, paddle::DataType::INT32, place);

  void* sort_temp_storage = nullptr;
  size_t sort_temp_bytes = 0;
  cub::DeviceRadixSort::SortPairs<T, int>(
      nullptr,
      sort_temp_bytes,
      valid_expert_ids.template data<T>(),
      sorted_valid_expert_ids.template data<T>(),
      original_flat_indices_valid.data<int>(),
      s_scatter_idx_valid.data<int>(),
      total_valid_experts,
      0,
      sizeof(T) * 8,
      stream);
  auto sort_temp_storage_tensor = paddle::empty(
      {static_cast<int64_t>(sort_temp_bytes)}, paddle::DataType::UINT8, place);
  sort_temp_storage = sort_temp_storage_tensor.data();
  cub::DeviceRadixSort::SortPairs<T, int>(
      sort_temp_storage,
      sort_temp_bytes,
      valid_expert_ids.template data<T>(),
      sorted_valid_expert_ids.template data<T>(),
      original_flat_indices_valid.data<int>(),
      s_scatter_idx_valid.data<int>(),
      total_valid_experts,
      0,
      sizeof(T) * 8,
      stream);
  PD_CHECK(cudaGetLastError() == cudaSuccess,
           "cub::DeviceRadixSort::SortPairs for valid failed.");

  auto s_reverse_scatter_idx_valid =
      paddle::empty({total_valid_experts}, paddle::DataType::INT32, place);
  int blocks_valid_out = (total_valid_experts + threads - 1) / threads;
  GenerateReverseScatterKernel<<<blocks_valid_out, threads, 0, stream>>>(
      s_scatter_idx_valid.data<int>(),
      s_reverse_scatter_idx_valid.data<int>(),
      total_valid_experts);
  PD_CHECK(cudaGetLastError() == cudaSuccess,
           "GenerateReverseScatterKernel failed.");

  auto original_full_indices =
      paddle::empty({total_elements}, paddle::DataType::INT32, place);
  PD_CHECK(launch_simple_arange(original_full_indices.data<int>(),
                                total_elements,
                                stream) == cudaSuccess,
           "launch_simple_arange for full failed.");
  auto s_scatter_idx_all =
      paddle::empty({total_elements}, paddle::DataType::INT32, place);
  auto topk_router_indices_flat =
      paddle::reshape(topk_router_indices, {total_elements});
  auto sorted_topk_indices_temp = paddle::empty_like(topk_router_indices_flat);

  size_t sort_all_bytes = 0;
  void* sort_all_temp_storage = nullptr;
  cub::DeviceRadixSort::SortPairs<T, int>(
      nullptr,
      sort_all_bytes,
      topk_router_indices_flat.template data<T>(),
      sorted_topk_indices_temp.template data<T>(),
      original_full_indices.data<int>(),
      s_scatter_idx_all.data<int>(),
      total_elements,
      0,
      sizeof(T) * 8,
      stream);
  auto sort_all_temp_storage_tensor = paddle::empty(
      {static_cast<int64_t>(sort_all_bytes)}, paddle::DataType::UINT8, place);
  sort_all_temp_storage = sort_all_temp_storage_tensor.data();
  cub::DeviceRadixSort::SortPairs<T, int>(
      sort_all_temp_storage,
      sort_all_bytes,
      topk_router_indices_flat.template data<T>(),
      sorted_topk_indices_temp.template data<T>(),
      original_full_indices.data<int>(),
      s_scatter_idx_all.data<int>(),
      total_elements,
      0,
      sizeof(T) * 8,
      stream);
  PD_CHECK(cudaGetLastError() == cudaSuccess,
           "cub::DeviceRadixSort::SortPairs for all failed.");

  auto x_gather_idx_all =
      paddle::empty({total_elements}, paddle::DataType::INT32, place);
  int blocks_all_out = (total_elements + threads - 1) / threads;
  ComputeXGatherIdxKernel<<<blocks_all_out, threads, 0, stream>>>(
      s_scatter_idx_all.data<int>(),
      x_gather_idx_all.data<int>(),
      K,
      total_elements);
  PD_CHECK(cudaGetLastError() == cudaSuccess,
           "ComputeXGatherIdxKernel failed.");

  int invalid_tokens = total_elements - total_valid_experts;
  auto x_gather_idx = x_gather_idx_all.slice(invalid_tokens, total_elements);

  return {padded_expert_frequency_offset,
          x_gather_idx,
          s_scatter_idx_valid,
          s_reverse_scatter_idx_valid,
          num_activated_per_token_offset};
}

std::vector<paddle::Tensor> RouterMetadataDispatch(
    const paddle::Tensor& topk_router_indices,
    const paddle::Tensor& expert_frequency_offset,
    int K) {
  if (topk_router_indices.dtype() == paddle::DataType::INT32) {
    return RouterMetadataCuda<int>(
        topk_router_indices, expert_frequency_offset, K);
  } else if (topk_router_indices.dtype() == paddle::DataType::INT64) {
    return RouterMetadataCuda<int64_t>(
        topk_router_indices, expert_frequency_offset, K);
  } else {
    PD_THROW(
        "Unsupported dtype for topk_router_indices. Must be int32 or int64.");
  }
}

PD_BUILD_OP(router_metadata)
    .Inputs({"TopkRouterIndices", "ExpertFrequencyOffset"})
    .Outputs({"PaddedExpertFrequencyOffset",
              "XGatherIdx",
              "SScatterIdxValid",
              "SReverseScatterIdxValid",
              "NumActivatedExpertPerTokenOffset"})
    .Attrs({"K: int"})
    .SetKernelFn(PD_KERNEL(RouterMetadataDispatch));
