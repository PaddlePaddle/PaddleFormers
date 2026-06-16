// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#ifdef __CUDACC__
#include <cuda.h>         // NOLINT
#include <cuda_bf16.h>    // NOLINT
#include <cuda_fp8.h>     // NOLINT
#include <cuda_runtime.h> // NOLINT
#endif

#include <iostream> // NOLINT
#include <limits>   // NOLINT

#include "paddle/extension.h"                // NOLINT
#include "paddle/phi/api/all.h"              // NOLINT
#include "paddle/phi/core/utils/data_type.h" // NOLINT
#ifdef __CUDACC__
#include "paddle/phi/kernels/funcs/math_cuda_utils.h" // NOLINT
#endif

template <paddle::DataType DType> struct TypeMap;
template <> struct TypeMap<paddle::DataType::BFLOAT16> {
  using type = phi::bfloat16;
};
template <> struct TypeMap<paddle::DataType::FLOAT16> {
  using type = phi::float16;
};
template <> struct TypeMap<paddle::DataType::FLOAT32> {
  using type = float;
};
template <> struct TypeMap<paddle::DataType::INT32> {
  using type = int;
};
template <> struct TypeMap<paddle::DataType::INT64> {
  using type = int64_t;
};

inline paddle::DataType TransToDataType(int64_t dtype) {
  // Caller must ensure dtype is a valid paddle::DataType enum value
  return static_cast<paddle::DataType>(dtype);
}

inline int LimitGridDim(int64_t n) {
  return static_cast<int>(std::min<int64_t>(n, 1024 * 1024));
}

inline int GetSwiGLURowGridSize(int64_t rows) { return LimitGridDim(rows); }

#ifdef __CUDACC__
template <typename T>
T **GetTensorDevicePtrs(const std::vector<paddle::Tensor> &tensors,
                        paddle::Tensor *ptr_tensor, cudaStream_t stream,
                        phi::Place place) {
  auto nbytes = tensors.size() * sizeof(T *);
  std::vector<const T *> cpu_ptrs(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    cpu_ptrs[i] = tensors[i].data<T>();
  }
  *ptr_tensor = paddle::empty({static_cast<int64_t>(nbytes)},
                              paddle::DataType::UINT8, place);
  auto *device_ptrs = reinterpret_cast<T **>(ptr_tensor->data());
  auto err = cudaMemcpyAsync(device_ptrs, cpu_ptrs.data(), nbytes,
                             cudaMemcpyHostToDevice, stream);
  PD_CHECK(err == cudaSuccess, "cudaMemcpyAsync error",
           cudaGetErrorString(err));
  err = cudaStreamSynchronize(stream);
  PD_CHECK(err == cudaSuccess, "cudaStreamSynchronize error",
           cudaGetErrorString(err));
  return device_ptrs;
}
#endif

template <typename T, int N> struct alignas(16) VectorType {
  T data[N];
};

#ifdef __CUDACC__
template <> struct alignas(16) VectorType<float, 4> {
  float4 data; // Built-in CUDA vector type  // NOLINT
};

template <> struct alignas(16) VectorType<__nv_bfloat16, 8> {
  __nv_bfloat16 data[8];
};

template <> struct alignas(16) VectorType<__nv_fp8_e4m3, 16> {
  __nv_fp8_e4m3 data[16];
};
#endif

template <> struct alignas(16) VectorType<uint8_t, 16> {
  uint8_t data[16];
};

#ifdef __CUDACC__
template <typename T>
__device__ __forceinline__ void unrolled_memcpy(const T *src, T *dst,
                                                const int64_t num_elements) {
#pragma unroll
  for (int64_t idx = threadIdx.x; idx < num_elements; idx += blockDim.x) {
    dst[idx] = src[idx];
  }
}

// Helper function to perform vectorized memory copy
template <typename T>
__device__ __forceinline__ void vectorized_memcpy(const T *src, T *dst,
                                                  const int64_t num_elements) {
  constexpr int vector_size_in_bytes = 16;
  const int64_t elements_per_vector = vector_size_in_bytes / sizeof(T);

  int64_t num_vectors = num_elements / elements_per_vector;
  int64_t remaining_elements = num_elements % elements_per_vector;

  using VecType = VectorType<T, elements_per_vector>;
  const VecType *src_vec = reinterpret_cast<const VecType *>(src);
  VecType *dst_vec = reinterpret_cast<VecType *>(dst);

#pragma unroll
  for (int64_t idx = threadIdx.x; idx < num_vectors; idx += blockDim.x) {
    dst_vec[idx] = src_vec[idx];
  }

  if (remaining_elements > 0) {
    int64_t offset = num_vectors * elements_per_vector;
    for (int64_t i = threadIdx.x; i < remaining_elements; i += blockDim.x) {
      dst[offset + i] = src[offset + i];
    }
  }
}

template <typename T>
__device__ __forceinline__ void
try_vectorized_memcpy(const T *src, T *dst, const int64_t num_elements) {
  bool is_aligned_128bit =
      ((uintptr_t)src & 0xF) == 0 && ((uintptr_t)dst & 0xF) == 0;
  if (is_aligned_128bit) {
    vectorized_memcpy(src, dst, num_elements);
  } else {
    unrolled_memcpy(src, dst, num_elements);
  }
}
#endif

#define PD_SWITCH_NUM_EXPERTS_IMPL(__num_expert, __max_num_experts, ...)       \
  if (__num_expert <= __max_num_experts) {                                     \
    constexpr auto MAX_NUM_EXPERTS_C = __max_num_experts;                      \
    do {                                                                       \
      __VA_ARGS__();                                                           \
    } while (0);                                                               \
    break;                                                                     \
  }

#define PD_SWITCH_NUM_EXPERTS(__num_experts_expr, ...)                         \
  do {                                                                         \
    auto __num_expert = (__num_experts_expr);                                  \
    PD_SWITCH_NUM_EXPERTS_IMPL(__num_expert, 8, __VA_ARGS__);                  \
    PD_SWITCH_NUM_EXPERTS_IMPL(__num_expert, 16, __VA_ARGS__);                 \
    PD_SWITCH_NUM_EXPERTS_IMPL(__num_expert, 32, __VA_ARGS__);                 \
    PD_SWITCH_NUM_EXPERTS_IMPL(__num_expert, 64, __VA_ARGS__);                 \
    PD_THROW("Unsupported expert number %d", static_cast<int>(__num_expert));  \
  } while (0)
