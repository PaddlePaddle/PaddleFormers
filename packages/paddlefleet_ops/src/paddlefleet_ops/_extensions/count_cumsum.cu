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

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <paddle/extension.h>
#include <cub/cub.cuh>

#define MAX_ALLOWED_E 50000
#define BLOCK_SIZE 1024
#define WARP_SIZE 32
#define LOG_WARP_SIZE 5

namespace cg = cooperative_groups;

using BlockScan = cub::BlockScan<uint32_t, BLOCK_SIZE>;

template <typename T, typename IdxT>
inline __device__ void load_128_bits(const T* src, T* dst, IdxT idx) {
  constexpr int num_elements = 16 / sizeof(T);
  float4 vec = *reinterpret_cast<const float4*>(src + idx * num_elements);
  *reinterpret_cast<float4*>(dst) = vec;
}

template <typename T, typename IdxT>
inline __device__ void store_128_bits(const T* src, T* dst, IdxT idx) {
  constexpr int num_elements = 16 / sizeof(T);
  float4 vec = *reinterpret_cast<const float4*>(src);
  *reinterpret_cast<float4*>(dst + idx * num_elements) = vec;
}

template <typename scalar_t>
inline __device__ void _update_local_count(const scalar_t* x,
                                           int32_t* shared_memory,
                                           const int64_t& N,
                                           const uint32_t global_thread_id,
                                           const uint32_t grid_size) {
  constexpr uint32_t N_per_thread = 16 / sizeof(scalar_t);
  const int64_t N_vec = N / N_per_thread;
  const int64_t vec_covered = N_vec * N_per_thread;

  for (int64_t i = global_thread_id; i < N_vec; i += grid_size) {
    scalar_t x_vec[N_per_thread];
    load_128_bits<scalar_t>(x, x_vec, i);

    for (uint32_t j = 0; j < N_per_thread; j++) {
      atomicAdd(&shared_memory[x_vec[j]], 1);
    }
  }

  const int64_t i = vec_covered + global_thread_id;
  if (i < N) {
    atomicAdd(&shared_memory[x[i]], 1);
  }
}

struct BlockPrefixCallbackOp {
  uint32_t running_total;
  __device__ BlockPrefixCallbackOp(uint32_t running_total)
      : running_total(running_total) {}

  __device__ int operator()(uint32_t block_aggregate) {
    uint32_t old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};

inline __device__ void _compute_cumsum(
    typename BlockScan::TempStorage& temp_storage,
    int32_t* shared_memory,
    const uint32_t& E) {
  const uint32_t num_loops = (E + blockDim.x - 1) / blockDim.x;
  uint32_t i = threadIdx.x;

  BlockPrefixCallbackOp prefix_op(0);

  for (uint32_t j = 0; j < num_loops; j++) {
    const bool is_valid_i = i < E;
    const uint32_t count = is_valid_i ? shared_memory[i] : 0;

    __syncwarp();

    uint32_t scan_value;
    BlockScan(temp_storage).InclusiveSum(count, scan_value, prefix_op);

    __syncthreads();

    if (is_valid_i) {
      shared_memory[i] = scan_value;
    }

    i += blockDim.x;
  }
}

template <typename scalar_t, bool do_cumsum>
__global__ void count_cumsum_cuda_kernel(const scalar_t* x,
                                         int32_t* count_output,
                                         int32_t* cumsum_output,
                                         const int64_t N,
                                         const uint32_t E) {
  // NOTE: gridDim = num_SMs is small (<256), grid_size and global_thread_id
  // never exceed int range.
  const uint32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t grid_size = gridDim.x * blockDim.x;

  extern __shared__ int32_t shared_memory[];

  const uint32_t E4 = E >> 2;
  const int32_t init_value[4] = {0, 0, 0, 0};

  for (uint32_t i = threadIdx.x; i < E4; i += blockDim.x) {
    store_128_bits<int32_t>(init_value, shared_memory, i);
  }

  for (uint32_t i = global_thread_id; i < E4; i += grid_size) {
    store_128_bits<int32_t>(init_value, count_output, i);
  }

  cg::this_grid().sync();

  _update_local_count<scalar_t>(
      x, shared_memory, N, global_thread_id, grid_size);

  __syncthreads();

  for (uint32_t i = threadIdx.x; i < E; i += blockDim.x) {
    atomicAdd(&count_output[i], shared_memory[i]);
  }

  if constexpr (do_cumsum) {
    __shared__ typename BlockScan::TempStorage temp_storage;

    cg::this_grid().sync();

    for (uint32_t i = threadIdx.x; i < E4; i += blockDim.x) {
      int32_t output_vec[4];
      load_128_bits<int32_t>(count_output, output_vec, i);
      store_128_bits<int32_t>(output_vec, shared_memory, i);
    }

    __syncthreads();

    _compute_cumsum(temp_storage, shared_memory, E);

    __syncthreads();

    for (uint32_t i = global_thread_id; i < E4; i += grid_size) {
      int32_t output_vec[4];
      load_128_bits<int32_t>(shared_memory, output_vec, i);
      store_128_bits<int32_t>(output_vec, cumsum_output, i);
    }
  }
}

template <typename scalar_t>
void count_cumsum_cuda_impl(const paddle::Tensor& x,
                            paddle::Tensor& count_output,
                            paddle::Tensor& cumsum_output,
                            bool do_cumsum,
                            int64_t N,
                            uint32_t E,
                            cudaStream_t stream) {
  int device_id;
  cudaGetDevice(&device_id);

  int num_SMs;
  cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, device_id);

  const uint32_t block_reduce_smem_size =
      do_cumsum
          ? std::max(sizeof(uint32_t), sizeof(typename BlockScan::TempStorage))
          : 0;

  void (*kernel)(
      const scalar_t*, int32_t*, int32_t*, const int64_t, const uint32_t);
  if (do_cumsum) {
    kernel = count_cumsum_cuda_kernel<scalar_t, true>;
  } else {
    kernel = count_cumsum_cuda_kernel<scalar_t, false>;
  }

  cudaFuncSetAttribute(
      kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      MAX_ALLOWED_E * sizeof(uint32_t) + block_reduce_smem_size);

  cudaLaunchConfig_t launch_config = {0};
  launch_config.blockDim = BLOCK_SIZE;
  launch_config.gridDim = num_SMs;
  launch_config.dynamicSmemBytes =
      E * sizeof(uint32_t) + block_reduce_smem_size;

  cudaLaunchAttribute attributes[1];
  attributes[0].id = cudaLaunchAttributeCooperative;
  attributes[0].val.cooperative = 1;

  launch_config.attrs = attributes;
  launch_config.numAttrs = 1;

  int32_t* cumsum_ptr = do_cumsum ? cumsum_output.data<int32_t>() : nullptr;

  cudaLaunchKernelEx(&launch_config,
                     kernel,
                     x.data<scalar_t>(),
                     count_output.data<int32_t>(),
                     cumsum_ptr,
                     N,
                     E);

  PD_CHECK(cudaGetLastError() == cudaSuccess,
           "count_cumsum_cuda_kernel failed.");
}

std::vector<paddle::Tensor> CountCumsumCuda(const paddle::Tensor& x,
                                            int E,
                                            bool do_cumsum) {
  PD_CHECK(E <= MAX_ALLOWED_E, "E exceeds MAX_ALLOWED_E.");
  PD_CHECK(E % 4 == 0, "E must be divisible by 4.");

  // TODO(xingmingyyj): Refactor this kernel to remove the do_cumsum parameter
  PD_CHECK(do_cumsum, "do_cumsum must be true.");

  const int64_t N = x.numel();
  auto place = x.place();
  cudaStream_t stream = x.stream();

  if (N == 0) {
    auto count_output = paddle::full({E}, 0, paddle::DataType::INT32, place);
    paddle::Tensor cumsum_output =
        do_cumsum ? paddle::full({E}, 0, paddle::DataType::INT32, place)
                  : paddle::empty({0}, paddle::DataType::INT32, place);
    return {count_output, cumsum_output};
  }

  auto count_output = paddle::empty({E}, paddle::DataType::INT32, place);

  paddle::Tensor cumsum_output;
  if (do_cumsum) {
    cumsum_output = paddle::empty({E}, paddle::DataType::INT32, place);
  } else {
    cumsum_output = paddle::empty({0}, paddle::DataType::INT32, place);
  }

  if (x.dtype() == paddle::DataType::INT32) {
    count_cumsum_cuda_impl<int>(
        x, count_output, cumsum_output, do_cumsum, N, E, stream);
  } else if (x.dtype() == paddle::DataType::INT64) {
    count_cumsum_cuda_impl<int64_t>(
        x, count_output, cumsum_output, do_cumsum, N, E, stream);
  } else {
    PD_THROW("Unsupported dtype for x. Must be int32 or int64.");
  }
  return {count_output, cumsum_output};
}

// The implementation of this kernel is inspired by SonicMoE
PD_BUILD_OP(count_cumsum)
    .Inputs({"X"})
    .Outputs({"CountOutput", "CumsumOutput"})
    .Attrs({"E: int", "do_cumsum: bool"})
    .SetKernelFn(PD_KERNEL(CountCumsumCuda));
