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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/fusion/gpu/fused_transpose_split_quant_kernel.h"
#include "paddle/phi/kernels/fusion/gpu/quant_utils.h"

// namespace phi {

template <typename T, int VecSize>
struct __align__(sizeof(T) * VecSize) VecType {
  T val[VecSize];
  __host__ __device__ inline T& operator[](size_t i) { return val[i]; }
  __host__ __device__ inline const T& operator[](size_t i) const {
    return val[i];
  }
};

template <typename InT, int VecSize>
__device__ void BlockLoad(const InT* input,
                          const float* input_scales,
                          __nv_bfloat16 x[8][4],
                          size_t K,
                          size_t k_scaled) {
  constexpr bool need_dequant = std::is_same_v<InT, phi::float8_e4m3fn>;

#pragma unroll
  for (uint32_t i = 0; i < 8; i++) {
    const uint32_t local_off_M = threadIdx.y + i * 16;
    const uint32_t off_m = blockIdx.x * 128 + local_off_M;
    const uint32_t off_k = blockIdx.y * 128 + threadIdx.x * VecSize;
    const size_t offset =
        static_cast<size_t>(off_m) * static_cast<size_t>(K) + off_k;

    float scale;
    if constexpr (need_dequant) {
      const uint32_t m_base = blockIdx.x * 128;
      const uint32_t m_stride = k_scaled;
      scale = input_scales[off_m * m_stride + blockIdx.y];
    }

#pragma unroll
    for (uint32_t j = 0; j < 4; j += VecSize) {
      if (off_k + j * 32 < K) {
        const size_t idx = offset + j * 32;
        using LoadT = VecType<InT, VecSize>;
        LoadT data = *reinterpret_cast<const LoadT*>(input + idx);
#pragma unroll
        for (uint32_t k = 0; k < VecSize; k++) {
          if constexpr (need_dequant) {
            x[i][j + k] = __float2bfloat16(static_cast<float>(data[k]) * scale);
          } else {
            x[i][j + k] = (*reinterpret_cast<__nv_bfloat16*>(&data[k]));
          }
        }
      }
    }
  }
}
template <bool Pow2Scales>
__device__ void BlockColumnScale(const __nv_bfloat16 x[8][4],
                                 float scales[128],
                                 __nv_bfloat16* shm) {
  // reduce [(8), 16, 32, 4] => [16, 32, 4]
  __nv_bfloat16 warp_max[4];
#pragma unroll
  for (uint32_t i = 0; i < 8; i++) {
#pragma unroll
    for (uint32_t j = 0; j < 4; j++) {
      const __nv_bfloat16 t = BF16_ABS(x[i][j]);
      warp_max[j] = i == 0 ? t : BF16_MAX(warp_max[j], t);
    }
  }

  // reduce [(16), 32, 4] => [8, 32, 4]
  if (threadIdx.y >= 8) {
#pragma unroll
    for (uint32_t j = 0; j < 4; j++) {
      shm[(threadIdx.y - 8) * 128 + threadIdx.x + j * 32] = warp_max[j];
    }
  }
  __syncthreads();

  // reduce [(8), 32, 4] => [32, 4]
  for (uint32_t offset = 8; offset > 0; offset /= 2) {
    if (threadIdx.y < offset) {
#pragma unroll
      for (uint32_t j = 0; j < 4; j++) {
        const __nv_bfloat16 other =
            offset == 8
                ? warp_max[j]
                : shm[(threadIdx.y + offset) * 128 + threadIdx.x + j * 32];
        __nv_bfloat16 next_val =
            BF16_MAX(shm[threadIdx.y * 128 + threadIdx.x + j * 32], other);
        if (offset > 1) {
          shm[threadIdx.y * 128 + threadIdx.x + j * 32] = next_val;
        } else {
          scales[threadIdx.x + j * 32] =
              ComputeScale<__nv_bfloat16, __nv_fp8_e4m3, Pow2Scales>(
                  static_cast<float>(next_val), 0.0f);
        }
      }
    }
    __syncthreads();
  }
}

template <typename OutT, int VecSize, bool Use_UE8M0>
__device__ void BlockStoreScale(void* scale,
                                size_t off_m,
                                float scales[128],
                                size_t K) {
  if (threadIdx.y < 4) {
    uint32_t off = threadIdx.y * 32 + threadIdx.x;
    if constexpr (VecSize == 4) {
      off = (off % 4) * 32 + off / 4;
    } else if constexpr (VecSize == 2) {
      off = (off / 64) * 64 + (off % 2) * 32 + (off % 64) / 2;
    }
    float scale_out = 1.0f / scales[off];
    const size_t idx_y = blockIdx.x - off_m / 128;
    const size_t idx_x = blockIdx.y * 128 + threadIdx.y * 32 + threadIdx.x;
    const size_t idx = idx_y * K + idx_x;
    if (idx_x < K) {
      if constexpr (Use_UE8M0) {
        // M -> 12 * 128, idy = M / 128 = 12
        // K -> 128
        // pack_idx = idy >> 2
        // byte_idx = idy % 4

        const size_t byte_idx = idx_y >> 2;
        const size_t pack_idx = idx_y % 4;
        // const size_t idx_x = blockIdx.y * 128 + threadIdx.y * 32 +
        // threadIdx.x;
        const size_t uint32_idx = byte_idx * K + idx_x;

        uint32_t* scale_u32 = reinterpret_cast<uint32_t*>(scale);
        const int exp = (reinterpret_cast<const int&>(scale_out) >> 23) & 0xFF;
        reinterpret_cast<uint8_t*>(&scale_u32[uint32_idx])[pack_idx] =
            static_cast<const uint8_t>(exp);
      } else {
        float* scale_f32 = reinterpret_cast<float*>(scale);
        scale_f32[idx] = scale_out;
      }
    }
  }
}

template <typename OutT, int VecSize>
__device__ void BlockStoreOut(OutT* out,
                              size_t off_m,
                              size_t cur_tokens,
                              const OutT shm[128][129],
                              size_t K) {
#pragma unroll
  for (uint32_t i = 0; i < 8; i++) {
    const size_t idx_m = blockIdx.x * size_t(128) + threadIdx.x * 4;
    const size_t idx_k = blockIdx.y * 128 + threadIdx.y + i * 16;
    const size_t idx = idx_k * cur_tokens + (idx_m - off_m);

    if (idx_k < K) {
      using StoreT = VecType<OutT, VecSize>;
      StoreT data;
#pragma unroll
      for (uint32_t j = 0; j < VecSize; j++) {
        data[j] = shm[i * 16 + threadIdx.y][threadIdx.x * 4 + j];
      }
      *reinterpret_cast<StoreT*>(out + idx) = data;
    }
  }
}

template <typename InT,
          typename OutT,
          bool Pow2Scales,
          int VecSize,
          bool Use_UE8M0>
__global__ void __launch_bounds__(512)
    FusedTransposeSplitQuantKernel(const InT* __restrict__ input,
                                   const float* __restrict__ input_scales,
                                   int64_t* __restrict__ meta,
                                   size_t num_experts,
                                   size_t K,
                                   size_t k_scaled) {
  __shared__ OutT shm[128][129];
  __shared__ size_t expert_info[2];
  __shared__ float scales[128];  // May be reused? Is it worthy?

  int64_t* tokens_per_expert = meta;
  OutT** out_ptrs = reinterpret_cast<OutT**>(meta + num_experts);
  void** scale_ptrs = reinterpret_cast<void**>(meta + num_experts * 2);

  // 1. Load 128x128 elements from input
  __nv_bfloat16 x[8][4];
  BlockLoad<InT, VecSize>(input, input_scales, x, K, k_scaled);

  // 2. Get expert index and offset of the current block
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    size_t idx_m = static_cast<size_t>(blockIdx.x) * size_t(128);
    size_t off_m = 0, next_off_m = 0;
    size_t expert_idx;
    for (expert_idx = 0; expert_idx < num_experts; expert_idx++) {
      next_off_m += tokens_per_expert[expert_idx];
      if (idx_m >= off_m && idx_m < next_off_m) {
        break;
      }
      off_m = next_off_m;
    }
    expert_info[0] = expert_idx;
    expert_info[1] = off_m;
  }

  // 3. Calculate scale along the column
  BlockColumnScale<Pow2Scales>(
      x, scales, reinterpret_cast<__nv_bfloat16*>(shm));

  // 4. Store scale
  const size_t expert_idx = expert_info[0];
  const size_t off_m = expert_info[1];
  BlockStoreScale<OutT, VecSize, Use_UE8M0>(
      scale_ptrs[expert_idx], off_m, scales, K);

// 5. Scale x and save into shared memory with transposed layout
#pragma unroll
  for (uint32_t i = 0; i < 8; i++) {
#pragma unroll
    for (uint32_t j = 0; j < 4; j += VecSize) {
#pragma unroll
      for (uint32_t k = 0; k < VecSize; k++) {
        float x_fp32 = static_cast<float>(x[i][j + k]);
        float x_scaled = x_fp32 * scales[threadIdx.x + (j + k) * 32];
        shm[threadIdx.x * VecSize + j * 32 + k][i * 16 + threadIdx.y] =
            static_cast<OutT>(x_scaled);
      }
    }
  }
  __syncthreads();

  // 6. Store 128x128 elements back
  // Note: out is always 4x vectorizable.
  BlockStoreOut<OutT, 4>(
      out_ptrs[expert_idx], off_m, tokens_per_expert[expert_idx], shm, K);
}

void fuse_transpose_split_fp8_quant(
    const paddle::Tensor& x,
    const paddle::optional<paddle::Tensor>& input_scales,
    std::vector<paddle::Tensor>& outs,
    std::vector<paddle::Tensor>& scales,
    const std::vector<int64_t>& tokens_per_expert,
    bool pow_2_scales,
    bool use_ue8m0) {
  auto place = x.place();
  auto stream = x.stream();

  auto x_dims = x.dims();
  const int64_t M = x_dims[0];
  const int64_t K = x_dims[1];
  const size_t num_experts = tokens_per_expert.size();

  if (M == 0 || K == 0 || num_experts == 0) {
    return;
  }

  PD_CHECK(x_dims.size() == 2);
  PD_CHECK(scales.size() == num_experts);
  for (size_t i = 0; i < num_experts; i++) {
    PD_CHECK(outs[i].dtype() == paddle::DataType::FLOAT8_E4M3FN);
    std::vector<int64_t> out_shape = outs[i].shape();
    PD_CHECK(out_shape.size() == 2);
    PD_CHECK(out_shape[0] == K);
    PD_CHECK(out_shape[1] % 128 == 0);

    std::vector<int64_t> scale_shape = scales[i].shape();
    PD_CHECK(scale_shape.size() == 2);
    PD_CHECK(scale_shape[1] == K);

    if (use_ue8m0) {
      PD_CHECK(scales[i].dtype() == paddle::DataType::INT32);
    } else {
      PD_CHECK(scales[i].dtype() == paddle::DataType::FLOAT32);
      PD_CHECK(scale_shape[0] == out_shape[1] / 128);
    }
  }

  PD_CHECK(K <= 65535 * 128, "only supports K <= 65535 * 128");

  // Copy meta (tokens_per_expert, out_ptrs, scale_ptrs) to device
  paddle::Tensor meta_cpu = paddle::empty(
      {static_cast<int64_t>(num_experts * 3)}, paddle::DataType::INT64);
  int64_t* meta_ptr = meta_cpu.data<int64_t>();
  for (size_t i = 0; i < num_experts; i++) {
    meta_ptr[i] = static_cast<int64_t>(tokens_per_expert[i]);
  }
  for (size_t i = 0; i < num_experts; i++) {
    meta_ptr[num_experts + i] =
        reinterpret_cast<int64_t>(outs[i].data<phi::float8_e4m3fn>());
  }
  for (size_t i = 0; i < num_experts; i++) {
    if (use_ue8m0) {
      meta_ptr[num_experts * 2 + i] =
          reinterpret_cast<int64_t>(scales[i].data<int>());
    } else {
      meta_ptr[num_experts * 2 + i] =
          reinterpret_cast<int64_t>(scales[i].data<float>());
    }
  }
  paddle::Tensor meta_gpu = meta_cpu.copy_to(x.place(), /*blocking=*/false);

  // pre-compute on CPU to reduce size_t division cost in kernel
  const size_t k_scaled = (K + 127) / 128;
  dim3 grid(M / 128, k_scaled);
  dim3 block(32, 16);

#define DTYPE_CASE(dtype, type) dtype == phi::DataType::type
#define LAUNCH_KERNEL(T, POW_2_SCALES, VEC_SIZE, USE_UE8M0)              \
  FusedTransposeSplitQuantKernel<T,                                      \
                                 phi::float8_e4m3fn,                     \
                                 POW_2_SCALES,                           \
                                 VEC_SIZE,                               \
                                 USE_UE8M0><<<grid, block, 0, stream>>>( \
      x.data<T>(),                                                       \
      input_scales ? input_scales.get().data<float>() : nullptr,         \
      meta_gpu.data<int64_t>(),                                          \
      num_experts,                                                       \
      K,                                                                 \
      k_scaled);

#define DISPATCH_UE8M0(T, POW_2_SCALES, VEC_SIZE)    \
  if (use_ue8m0) {                                   \
    LAUNCH_KERNEL(T, POW_2_SCALES, VEC_SIZE, true);  \
  } else {                                           \
    LAUNCH_KERNEL(T, POW_2_SCALES, VEC_SIZE, false); \
  }

#define DISPATCH_DATATYPE(POW_2_SCALES, VEC_SIZE)               \
  if (DTYPE_CASE(x.dtype(), BFLOAT16)) {                        \
    DISPATCH_UE8M0(phi::bfloat16, POW_2_SCALES, VEC_SIZE);      \
  } else if (DTYPE_CASE(x.dtype(), FLOAT8_E4M3FN)) {            \
    DISPATCH_UE8M0(phi::float8_e4m3fn, POW_2_SCALES, VEC_SIZE); \
  }

#define LAUNCH_KERNEL_PARTIAL(VEC_SIZE) \
  if (pow_2_scales) {                   \
    DISPATCH_DATATYPE(true, VEC_SIZE);  \
  } else {                              \
    DISPATCH_DATATYPE(false, VEC_SIZE); \
  }

  if (K % 4 == 0) {
    LAUNCH_KERNEL_PARTIAL(4);
  } else if (K % 2 == 0) {
    LAUNCH_KERNEL_PARTIAL(2);
  } else {
    LAUNCH_KERNEL_PARTIAL(1);
  }

#undef LAUNCH_KERNEL_PARTIAL
#undef LAUNCH_KERNEL
}

// }  // namespace phi

PD_BUILD_OP(fuse_transpose_split_fp8_quant)
    .Inputs({"x",
             paddle::Optional("input_scales"),
             paddle::Vec("outs"),
             paddle::Vec("scales")})
    .Attrs({"tokens_per_expert: std::vector<int64_t>",
            "pow_2_scales: bool",
            "use_ue8m0: bool"})
    .SetKernelFn(PD_KERNEL(fuse_transpose_split_fp8_quant));
