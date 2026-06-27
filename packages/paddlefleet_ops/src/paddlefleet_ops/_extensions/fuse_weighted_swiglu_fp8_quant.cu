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

#include <algorithm>
#include <cstdint>
#include <limits>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/fusion/gpu/quant_utils.h"

__host__ __device__ __forceinline__ int ceil_div(int x, int y) {
  return (x + y - 1) / y;
}

__host__ __device__ __forceinline__ int64_t ceil_div_i64(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

inline paddle::Tensor GetEmptyTensor(const common::DDim& dims,
                                     const paddle::DataType& dtype,
                                     const paddle::Place& place) {
  auto* allocator = paddle::GetAllocator(place);
  phi::DenseTensor dense_tensor;
  dense_tensor.Resize(dims);
  dense_tensor.AllocateFrom(
      allocator, dtype, dense_tensor.numel() * phi::SizeOf(dtype));
  return paddle::Tensor(std::make_shared<phi::DenseTensor>(dense_tensor));
}

inline paddle::Tensor GetEmptyTensor(const common::DDim& dims,
                                     const common::DDim& strides,
                                     const paddle::DataType& dtype,
                                     const paddle::Place& place) {
  auto* allocator = paddle::GetAllocator(place);
  phi::DenseTensor dense_tensor;
  dense_tensor.Resize(dims);
  dense_tensor.AllocateFrom(
      allocator, dtype, dense_tensor.numel() * phi::SizeOf(dtype));
  dense_tensor.set_strides(strides);
  return paddle::Tensor(std::make_shared<phi::DenseTensor>(dense_tensor));
}

// Unified launch macros.  When __k_has_clamp = false, `clamp_value` is passed
// as 0.0 and is statically eliminated inside the kernel via `if constexpr`,
// so the non-clamp instantiation produces math identical to the original.
#define LAUNCH_FUSED_SPAQ(__using_pow2_scaling, __with_prob, __k_has_clamp) \
  do {                                                                      \
    auto kernel = FusedSPAQKernel<__using_pow2_scaling,                     \
                                  __with_prob,                              \
                                  ScaleT,                                   \
                                  use_ue8m0,                                \
                                  __k_has_clamp>;                           \
    kernel<<<grid, block, 0, stream>>>(                                     \
        x_data, prob_data, out_data, scale_data, rows, cols, clamp_value);  \
  } while (0)

#define LAUNCH_FUSED_SPAQ_VEC4(                             \
    __using_pow2_scaling, __with_prob, __k_has_clamp)       \
  do {                                                      \
    auto kernel = FusedSPAQKernelVec4<__using_pow2_scaling, \
                                      __with_prob,          \
                                      thread_per_block,     \
                                      ScaleT,               \
                                      use_ue8m0,            \
                                      __k_has_clamp>;       \
    kernel<<<grid, block, 0, stream>>>(x_data,              \
                                       prob_data,           \
                                       out_data,            \
                                       scale_data,          \
                                       rows,                \
                                       cols,                \
                                       scale_cols,          \
                                       clamp_value);        \
  } while (0)

#define LAUNCH_FUSED_SPAQ_VEC8(                             \
    __using_pow2_scaling, __with_prob, __k_has_clamp)       \
  do {                                                      \
    auto kernel = FusedSPAQKernelVec8<__using_pow2_scaling, \
                                      __with_prob,          \
                                      thread_per_block,     \
                                      ScaleT,               \
                                      use_ue8m0,            \
                                      __k_has_clamp>;       \
    kernel<<<grid, block, 0, stream>>>(x_data,              \
                                       prob_data,           \
                                       out_data,            \
                                       scale_data,          \
                                       rows,                \
                                       cols,                \
                                       scale_cols,          \
                                       clamp_value);        \
  } while (0)

// Scalar fallback: cols % 8 != 0. Plain elementwise kernel without an
// in-kernel row loop, so we chunk rows on the host to honor gridDim.y <=
// 65535 while the dispatch arithmetic stays in int64_t. Pointers are
// advanced per chunk; scale_stride is the row stride into the [rows,
// scale_stride] scale tensor.
#define LAUNCH_FUSED_SPAQ_SCALAR(                                             \
    __using_pow2_scaling, __with_prob, __k_has_clamp)                         \
  do {                                                                        \
    auto kernel = FusedSPAQKernel<__using_pow2_scaling,                       \
                                  __with_prob,                                \
                                  ScaleT,                                     \
                                  use_ue8m0,                                  \
                                  __k_has_clamp>;                             \
    constexpr int64_t kMaxRowsPerLaunch = 65535;                              \
    const int64_t scale_stride = (cols / 2 + 127) / 128;                      \
    grid.x = static_cast<unsigned int>(ceil_div_i64(cols / 2, block.x));      \
    for (int64_t row_offset = 0; row_offset < rows;                           \
         row_offset += kMaxRowsPerLaunch) {                                   \
      const int64_t chunk_rows =                                              \
          std::min<int64_t>(kMaxRowsPerLaunch, rows - row_offset);            \
      grid.y = static_cast<unsigned int>(chunk_rows);                         \
      const phi::bfloat16* x_chunk = x_data + row_offset * cols;              \
      const float* prob_chunk = prob_data ? prob_data + row_offset : nullptr; \
      phi::float8_e4m3fn* out_chunk = out_data + row_offset * (cols / 2);     \
      ScaleT* scale_chunk = scale_data + row_offset * scale_stride;           \
      kernel<<<grid, block, 0, stream>>>(x_chunk,                             \
                                         prob_chunk,                          \
                                         out_chunk,                           \
                                         scale_chunk,                         \
                                         chunk_rows,                          \
                                         cols,                                \
                                         clamp_value);                        \
    }                                                                         \
  } while (0)

#define DISPATCH_BOOL(cond, k_name, ...) \
  if (cond) {                            \
    constexpr bool k_name = true;        \
    __VA_ARGS__;                         \
  } else {                               \
    constexpr bool k_name = false;       \
    __VA_ARGS__;                         \
  }

typedef struct __align__(8) {
  __nv_bfloat16 x;
  __nv_bfloat16 y;
  __nv_bfloat16 z;
  __nv_bfloat16 w;
} bfloat16x4_t;

typedef struct __align__(4) {
  __nv_fp8_e4m3 x;
  __nv_fp8_e4m3 y;
  __nv_fp8_e4m3 z;
  __nv_fp8_e4m3 w;
} fp8_e4m3x4_t;

// ---------------------------------------------------------------------------
// SwiGLU device helpers
//   kHasClamp = false  -> identical to the original `fast_swiglu` /
//                         `fast_swiglu_vec4`.  The `if constexpr` branch is
//                         eliminated at compile time.
//   kHasClamp = true   -> applies elementwise clamp on g and symmetric clamp
//                         on v, then runs the same SwiGLU.
// ---------------------------------------------------------------------------
template <bool kHasClamp = false>
__device__ __forceinline__ float fast_swiglu(const __nv_bfloat16 x,
                                             const __nv_bfloat16 y,
                                             const double clamp_value = 0.0) {
  float cv = static_cast<float>(clamp_value);
  float x_f = __bfloat162float(x);
  float y_f = __bfloat162float(y);
  if constexpr (kHasClamp) {
    x_f = fminf(x_f, cv);
    y_f = fmaxf(fminf(y_f, cv), -cv);
  }
  // Use IEEE division for clamp path to match reference bit-exact,
  // fast reciprocal for non-clamp (pre-existing behaviour).
  const float silu = [&] {
    if constexpr (kHasClamp) {
      return x_f * (1.0f / (1.0f + __expf(-x_f)));
    } else {
      return x_f * __frcp_rn(1.0f + __expf(-x_f));
    }
  }();
  return silu * y_f;
}

template <bool kHasClamp = false>
__device__ __forceinline__ float4
fast_swiglu_vec4(const bfloat16x4_t& lhs,
                 const bfloat16x4_t& rhs,
                 const double clamp_value = 0.0) {
  float cv = static_cast<float>(clamp_value);
  float x_f_x = __bfloat162float(lhs.x);
  float x_f_y = __bfloat162float(lhs.y);
  float x_f_z = __bfloat162float(lhs.z);
  float x_f_w = __bfloat162float(lhs.w);

  float y_f_x = __bfloat162float(rhs.x);
  float y_f_y = __bfloat162float(rhs.y);
  float y_f_z = __bfloat162float(rhs.z);
  float y_f_w = __bfloat162float(rhs.w);

  if constexpr (kHasClamp) {
    x_f_x = fminf(x_f_x, cv);
    x_f_y = fminf(x_f_y, cv);
    x_f_z = fminf(x_f_z, cv);
    x_f_w = fminf(x_f_w, cv);

    y_f_x = fmaxf(fminf(y_f_x, cv), -cv);
    y_f_y = fmaxf(fminf(y_f_y, cv), -cv);
    y_f_z = fmaxf(fminf(y_f_z, cv), -cv);
    y_f_w = fmaxf(fminf(y_f_w, cv), -cv);
  }

  // Use IEEE division for clamp path to match reference bit-exact,
  // fast reciprocal for non-clamp (pre-existing behaviour).
  const float sig_x = [&] {
    if constexpr (kHasClamp) {
      return 1.0f / (1.0f + __expf(-x_f_x));
    } else {
      return __frcp_rn(1.0f + __expf(-x_f_x));
    }
  }();
  const float sig_y = [&] {
    if constexpr (kHasClamp) {
      return 1.0f / (1.0f + __expf(-x_f_y));
    } else {
      return __frcp_rn(1.0f + __expf(-x_f_y));
    }
  }();
  const float sig_z = [&] {
    if constexpr (kHasClamp) {
      return 1.0f / (1.0f + __expf(-x_f_z));
    } else {
      return __frcp_rn(1.0f + __expf(-x_f_z));
    }
  }();
  const float sig_w = [&] {
    if constexpr (kHasClamp) {
      return 1.0f / (1.0f + __expf(-x_f_w));
    } else {
      return __frcp_rn(1.0f + __expf(-x_f_w));
    }
  }();

  const float silu_x = x_f_x * sig_x;
  const float silu_y = x_f_y * sig_y;
  const float silu_z = x_f_z * sig_z;
  const float silu_w = x_f_w * sig_w;

  return {silu_x * y_f_x, silu_y * y_f_y, silu_z * y_f_z, silu_w * y_f_w};
}

__device__ __forceinline__ float amax_float4(const float4& vec) {
  return fmaxf(fmaxf(fabsf(vec.x), fabsf(vec.y)),
               fmaxf(fabsf(vec.z), fabsf(vec.w)));
}

__device__ __forceinline__ fp8_e4m3x4_t
scale_fp32x4_to_fp8x4(const float4& vec, const float scale) {
  return {static_cast<__nv_fp8_e4m3>(vec.x * scale),
          static_cast<__nv_fp8_e4m3>(vec.y * scale),
          static_cast<__nv_fp8_e4m3>(vec.z * scale),
          static_cast<__nv_fp8_e4m3>(vec.w * scale)};
}

// ---------------------------------------------------------------------------
// FusedSPAQKernelVec4 — unified, kHasClamp templated
// ---------------------------------------------------------------------------
template <bool using_pow2_scaling,
          bool with_prob,
          int thread_per_block,
          typename ScaleT,
          bool ue8m0,
          bool kHasClamp = false>
__global__ void FusedSPAQKernelVec4(const phi::bfloat16* __restrict__ Xin,
                                    const float* __restrict__ prob,
                                    phi::float8_e4m3fn* __restrict__ out,
                                    ScaleT* __restrict__ scales,
                                    const int64_t rows,
                                    const int64_t cols,
                                    const int64_t scale_cols,
                                    const double clamp_value = 0.0) {
  constexpr int elements_per_thread = 4;
  constexpr int warp_size = 32;
  constexpr int warp_num = thread_per_block / warp_size;
  const int64_t scale_stride = scale_cols;
  const int lane = threadIdx.x % warp_size;
  const int64_t x_offset =
      static_cast<int64_t>(threadIdx.x) * elements_per_thread;
  const unsigned int mask = 0xffffffff;  // whole warp mask

  for (int64_t base_y = static_cast<int64_t>(blockIdx.y); base_y < rows;
       base_y += static_cast<int64_t>(gridDim.y)) {
    const int64_t in_y_idx = base_y;
    const int64_t in_x_idx = static_cast<int64_t>(blockIdx.x) *
                                 static_cast<int64_t>(blockDim.x) *
                                 elements_per_thread +
                             x_offset;
    const int64_t src_idx = in_y_idx * cols + in_x_idx;

    float p_t0;

    if (in_x_idx >= cols / 2) [[unlikely]]
      continue;

    if constexpr (with_prob) {
      // Prefetch prob
      if (lane == 0) p_t0 = prob[in_y_idx];
    }

    const __nv_bfloat16* X = reinterpret_cast<const __nv_bfloat16*>(Xin);

    // Initialize activation storage
    float4 act_f32x4;
    bfloat16x4_t lhs_bf16x4, rhs_bf16x4;

    // Reinterpret input pointer as bfloat16x4_t* for vectorized loading
    const bfloat16x4_t* X_lhs_vec =
        reinterpret_cast<const bfloat16x4_t*>(X + src_idx);
    const bfloat16x4_t* X_rhs_vec =
        reinterpret_cast<const bfloat16x4_t*>(X + src_idx + cols / 2);

    lhs_bf16x4 = *X_lhs_vec;
    rhs_bf16x4 = *X_rhs_vec;

    act_f32x4 =
        fast_swiglu_vec4<kHasClamp>(lhs_bf16x4, rhs_bf16x4, clamp_value);

    if constexpr (with_prob) {
      // Warp level sync to avoid syncthreads
      const float p = __shfl_sync(mask, p_t0, 0);
      act_f32x4.x *= p;
      act_f32x4.y *= p;
      act_f32x4.z *= p;
      act_f32x4.w *= p;
    }

    // Phase 2: Block Reduction to find per-quant block absolute maxima
    // Compute absolute values
    float thread_amax = amax_float4(act_f32x4);

// All-Reduce within the warp
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      const float val = __shfl_down_sync(mask, thread_amax, offset);
      thread_amax = fmaxf(thread_amax, val);
    }
    const float final_amax = __shfl_sync(mask, thread_amax, 0);

    // Phase 3: Compute scales and quantize the outputs
    const float scale = ComputeScale<float, __nv_fp8_e4m3, using_pow2_scaling>(
        final_amax, 0.0f);
    const float inv_scale = __frcp_rn(scale);

    const fp8_e4m3x4_t act_fp8x4 = scale_fp32x4_to_fp8x4(act_f32x4, scale);
    fp8_e4m3x4_t* const out_vec_addr =
        reinterpret_cast<fp8_e4m3x4_t*>(out + in_y_idx * cols / 2 + in_x_idx);
    *out_vec_addr = act_fp8x4;

    if (lane == 0) {
      if constexpr (ue8m0) {
        const size_t row_idx = in_y_idx;
        const size_t col_idx = in_x_idx >> 7;
        const size_t idx = (col_idx >> 2) * (static_cast<size_t>(rows) * 4) +
                           row_idx * 4 + (col_idx & 0x3);
        const uint8_t exp =
            (reinterpret_cast<const int&>(inv_scale) >> 23) & 0xFF;
        uint8_t* const dst = reinterpret_cast<uint8_t*>(scales) + idx;
        *dst = exp;
      } else {
        const int64_t scale_idx = in_y_idx * scale_stride + in_x_idx / 128;
        scales[scale_idx] = inv_scale;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// FusedSPAQKernelVec8 — unified, kHasClamp templated
// ---------------------------------------------------------------------------
template <bool using_pow2_scaling,
          bool with_prob,
          int thread_per_block,
          typename ScaleT,
          bool ue8m0,
          bool kHasClamp = false>
__global__ void FusedSPAQKernelVec8(const phi::bfloat16* __restrict__ Xin,
                                    const float* __restrict__ prob,
                                    phi::float8_e4m3fn* __restrict__ out,
                                    ScaleT* __restrict__ scales,
                                    const int64_t rows,
                                    const int64_t cols,
                                    const int64_t scale_cols,
                                    const double clamp_value = 0.0) {
  constexpr int elements_per_thread = 8;
  constexpr int warp_num = thread_per_block / 32;
  const int lane = threadIdx.x & 31;
  const int64_t x_offset =
      static_cast<int64_t>(threadIdx.x) * elements_per_thread;
  const unsigned int mask = 0xffffffff;

  for (int64_t base_y = static_cast<int64_t>(blockIdx.y); base_y < rows;
       base_y += static_cast<int64_t>(gridDim.y)) {
    const int64_t in_x_idx =
        ((static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x)) *
         elements_per_thread) +
        x_offset;
    const int64_t src_idx = base_y * cols + in_x_idx;

    if (in_x_idx >= cols / 2) [[unlikely]]
      continue;

    float p_t0;
    if constexpr (with_prob) {
      // Prefetch prob
      if (lane == 0) p_t0 = prob[base_y];
    }

    const __nv_bfloat16* X = reinterpret_cast<const __nv_bfloat16*>(Xin);

    // Initialize activation storage (2x vec4 = vec8)
    float4 act_f32x4_0, act_f32x4_1;
    bfloat16x4_t lhs_bf16x4_0, lhs_bf16x4_1;
    bfloat16x4_t rhs_bf16x4_0, rhs_bf16x4_1;

    // Reinterpret input pointer as int4* for 128-bit vectorized loading
    const int4 lhs_packed = *reinterpret_cast<const int4*>(X + src_idx);
    const int4 rhs_packed =
        *reinterpret_cast<const int4*>(X + src_idx + cols / 2);

    lhs_bf16x4_0 = *reinterpret_cast<const bfloat16x4_t*>(&lhs_packed.x);
    lhs_bf16x4_1 = *reinterpret_cast<const bfloat16x4_t*>(&lhs_packed.z);
    rhs_bf16x4_0 = *reinterpret_cast<const bfloat16x4_t*>(&rhs_packed.x);
    rhs_bf16x4_1 = *reinterpret_cast<const bfloat16x4_t*>(&rhs_packed.z);

    act_f32x4_0 =
        fast_swiglu_vec4<kHasClamp>(lhs_bf16x4_0, rhs_bf16x4_0, clamp_value);
    act_f32x4_1 =
        fast_swiglu_vec4<kHasClamp>(lhs_bf16x4_1, rhs_bf16x4_1, clamp_value);

    if constexpr (with_prob) {
      // Warp level sync to avoid syncthreads
      const float p = __shfl_sync(mask, p_t0, 0);
      act_f32x4_0.x *= p;
      act_f32x4_0.y *= p;
      act_f32x4_0.z *= p;
      act_f32x4_0.w *= p;
      act_f32x4_1.x *= p;
      act_f32x4_1.y *= p;
      act_f32x4_1.z *= p;
      act_f32x4_1.w *= p;
    }

    // Phase 2: Block Reduction to find per-quant block absolute maxima
    // Compute absolute values for 8 elements
    float thread_amax =
        fmaxf(amax_float4(act_f32x4_0), amax_float4(act_f32x4_1));

    // All-Reduce within 16-thread sub-groups (128 elements per group)
    // Use XOR sync to reduce independent groups: [0-15] and [16-31]
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
      const float val = __shfl_xor_sync(mask, thread_amax, offset);
      thread_amax = fmaxf(thread_amax, val);
    }

    // Phase 3: Compute scales and quantize the outputs
    const float scale = ComputeScale<float, __nv_fp8_e4m3, using_pow2_scaling>(
        thread_amax, 0.0f);

    union {
      struct {
        fp8_e4m3x4_t v0;
        fp8_e4m3x4_t v1;
      } vec;
      uint64_t packed;
    } out_packer;

    out_packer.vec.v0 = scale_fp32x4_to_fp8x4(act_f32x4_0, scale);
    out_packer.vec.v1 = scale_fp32x4_to_fp8x4(act_f32x4_1, scale);

    // Store 64 bits at once
    *reinterpret_cast<uint64_t*>(out + base_y * (cols >> 1) + in_x_idx) =
        out_packer.packed;

    // Write scale (one scale per 16 threads / 128 elements)
    if (lane % 16 == 0) {
      const float inv_scale = __frcp_rn(scale);
      if constexpr (ue8m0) {
        const size_t col_idx = in_x_idx >> 7;
        const size_t idx = (col_idx >> 2) * (static_cast<size_t>(rows) * 4) +
                           static_cast<size_t>(base_y) * 4 + (col_idx & 0x3);
        const uint8_t exp =
            (reinterpret_cast<const int&>(inv_scale) >> 23) & 0xFF;
        uint8_t* const dst = reinterpret_cast<uint8_t*>(scales) + idx;
        *dst = exp;
      } else {
        const int64_t scale_idx = base_y * scale_cols + (in_x_idx >> 7);
        scales[scale_idx] = inv_scale;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// FusedSPAQKernel (scalar) — unified, kHasClamp templated
// ---------------------------------------------------------------------------
template <bool using_pow2_scaling,
          bool with_prob,
          typename ScaleT,
          bool ue8m0,
          bool kHasClamp = false>
__global__ void FusedSPAQKernel(const phi::bfloat16* __restrict__ Xin,
                                const float* __restrict__ prob,
                                phi::float8_e4m3fn* __restrict__ out,
                                ScaleT* __restrict__ scales,
                                const int64_t rows,
                                const int64_t cols,
                                const double clamp_value = 0.0) {
  // Configure shared memory
  __shared__ float smem_tile[256];  // Shared memory for activation values
  __shared__ float warp_max[2][4];  // Shared memory for warp maxima (2 quant
                                    // blocks x 4 warps)
  __shared__ __nv_bfloat16
      quant_block_amax[2];  // Shared memory for quant block maxima

  const __nv_bfloat16* X = reinterpret_cast<const __nv_bfloat16*>(Xin);
  const int x_offset = threadIdx.x;
  const int quant_block_idx =
      threadIdx.x / 128;  // 0 or 1, two quant blocks per block
  const int64_t in_y_idx = static_cast<int64_t>(blockIdx.y);
  const int64_t in_x_idx =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
      static_cast<int64_t>(x_offset);
  const int64_t src_idx = in_y_idx * cols + in_x_idx;

  // Load data and compute swiGLU activation
  if (in_x_idx < cols / 2) [[likely]] {        // NOLINT
    __nv_bfloat16 x1 = X[src_idx];             // First half of the input
    __nv_bfloat16 x2 = X[src_idx + cols / 2];  // Second half of the input

    if constexpr (with_prob) {
      float row_prob = prob[in_y_idx];
      smem_tile[x_offset] =
          fast_swiglu<kHasClamp>(x1, x2, clamp_value) * row_prob;
    } else {
      smem_tile[x_offset] = fast_swiglu<kHasClamp>(x1, x2, clamp_value);
    }
  }

  __syncthreads();  // Ensure all threads have loaded their data

  // Phase 2: Block Reduction to find per-quant block absolute maximums
  float local_max = (in_x_idx < (cols / 2)) ? fabsf(smem_tile[x_offset]) : 0.0f;

  // Warp-level reduction
  unsigned int mask = 0xffffffff;
  int lane = threadIdx.x % 32;
  int warp_id =
      (threadIdx.x % 128) / 32;  // Warp ID within the quant block (0-3)

  // Reduce within the warp
  for (int offset = 16; offset > 0; offset /= 2) {
    float val = __shfl_down_sync(mask, local_max, offset);
    local_max = fmaxf(local_max, val);
  }

  // Store warp maxima
  if (lane == 0) {
    warp_max[quant_block_idx][warp_id] = local_max;
  }

  __syncthreads();

  // Reduce warp maxima to get quant block maxima
  if (warp_id == 0 && lane < 4) {
    if (threadIdx.x < 256) {  // Ensure only valid threads participate
      float block_max = warp_max[quant_block_idx][lane];
      // Reduce over the 4 warp maxima
      if (lane == 0) {
        block_max = fmaxf(block_max, warp_max[quant_block_idx][1]);
        block_max = fmaxf(block_max, warp_max[quant_block_idx][2]);
        block_max = fmaxf(block_max, warp_max[quant_block_idx][3]);
        quant_block_amax[quant_block_idx] = __float2bfloat16(block_max);
      }
    }
  }

  __syncthreads();

  // Phase 3: Compute scales and quantize the outputs
  const float block_max_float =
      static_cast<float>(quant_block_amax[quant_block_idx]);
  const int64_t scale_stride = (cols / 2 + 127) / 128;

  float scale = ComputeScale<float, __nv_fp8_e4m3, using_pow2_scaling>(
      block_max_float, 0.0f);
  float inv_scale = __frcp_rn(scale);

  // Quantize
  float output_scaled_fp32 = smem_tile[x_offset] * scale;

  const int64_t g_output_y_offset = in_y_idx;
  const int64_t g_output_x_offset = in_x_idx;

  // Write output and scales
  if (g_output_y_offset < rows && g_output_x_offset < cols / 2) {
    out[g_output_y_offset * (cols / 2) + g_output_x_offset] =
        static_cast<phi::float8_e4m3fn>(output_scaled_fp32);
    if (x_offset % 128 == 0) {
      if constexpr (ue8m0) {
        const size_t row_idx = g_output_y_offset;
        const size_t col_idx = in_x_idx >> 7;
        const size_t idx = (col_idx >> 2) * (static_cast<size_t>(rows) * 4) +
                           row_idx * 4 + (col_idx & 0x3);
        const uint8_t exp =
            (reinterpret_cast<const int&>(inv_scale) >> 23) & 0xFF;
        uint8_t* const dst = reinterpret_cast<uint8_t*>(scales) + idx;
        *dst = exp;
      } else {
        // Only one thread per quant block writes the scale
        scales[g_output_y_offset * scale_stride + in_x_idx / 128] = inv_scale;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// dispatch_fused_spaq — unified, kHasClamp templated.  Non-clamp callers omit
// `clamp_value` (defaults to 0.0) and the clamp branch is statically removed.
// ---------------------------------------------------------------------------
template <typename ScaleT, bool use_ue8m0, bool kHasClamp = false>
void dispatch_fused_spaq(const phi::bfloat16* x_data,
                         const float* prob_data,
                         phi::float8_e4m3fn* out_data,
                         void* void_scale_data,
                         cudaStream_t stream,
                         const int64_t rows,
                         const int64_t cols,
                         const bool& using_pow2_scaling,
                         const bool& with_prob,
                         const double clamp_value = 0.0) {
  constexpr int thread_per_block = 256;
  dim3 grid;
  dim3 block;

  ScaleT* scale_data = static_cast<ScaleT*>(void_scale_data);

  if (cols % 16 == 0) {
    block.x = thread_per_block;
    constexpr int vec_numel = 8;
    const int64_t scale_cols = (cols / 2 + 127) / 128;
    DISPATCH_BOOL(
        using_pow2_scaling,
        k_using_pow2_scaling,
        DISPATCH_BOOL(
            with_prob,
            k_with_prob,
            grid.y = rows > 65535 ? 65535 : static_cast<unsigned int>(rows);
            grid.x = static_cast<unsigned int>(
                ceil_div_i64(cols / 2, block.x * vec_numel));
            LAUNCH_FUSED_SPAQ_VEC8(
                k_using_pow2_scaling, k_with_prob, kHasClamp);))
  } else if (cols % 8 == 0) {
    // Use mixed vectorizing strategy, while cols size be 8x (4x2)
    // Each thread process 4 bfloat16 element in same row, each warp handles
    // 1x128 vector Each block handles several sub-row (numel = 4 x blockDim.x)
    // of input vector
    block.x = thread_per_block;
    constexpr int vec_numel = 4;
    const int64_t scale_cols = (cols / 2 + 127) / 128;
    DISPATCH_BOOL(
        using_pow2_scaling,
        k_using_pow2_scaling,
        DISPATCH_BOOL(
            with_prob,
            k_with_prob,
            grid.y = rows > 65535 ? 65535 : static_cast<unsigned int>(rows);
            grid.x = static_cast<unsigned int>(
                ceil_div_i64(cols / 2, block.x * vec_numel));
            LAUNCH_FUSED_SPAQ_VEC4(
                k_using_pow2_scaling, k_with_prob, kHasClamp);))
  } else {
    // Plain elementwise strategy:
    // Each block processes a sub-row (numel = blockDim.x) of the input
    // tensor. The kernel has no in-kernel row loop, so LAUNCH_FUSED_SPAQ_SCALAR
    // chunks rows on the host (kMaxRowsPerLaunch = 65535) and advances all
    // pointers per chunk. ue8m0 path requires cols % 8 == 0 so it never lands
    // here, which keeps the scale layout row-major [rows, scale_stride] and
    // makes scale_chunk pointer arithmetic straightforward.
    block.x = thread_per_block;
    DISPATCH_BOOL(
        using_pow2_scaling,
        k_using_pow2_scaling,
        DISPATCH_BOOL(with_prob,
                      k_with_prob,
                      LAUNCH_FUSED_SPAQ_SCALAR(
                          k_using_pow2_scaling, k_with_prob, kHasClamp);))
  }
}

// ---------------------------------------------------------------------------
// Shared host wrapper body (templated on kHasClamp).  Both Python-facing ops
// reuse this; clamp_value is forwarded to the kernel only when kHasClamp.
// ---------------------------------------------------------------------------
template <bool kHasClamp>
static std::vector<paddle::Tensor> FusedWeightedSwigluActQuantImpl(
    const paddle::Tensor& x,
    const paddle::optional<paddle::Tensor>& prob,
    const bool using_pow2_scaling,
    const bool use_ue8m0,
    const double clamp_value) {
  auto place = x.place();
  auto stream = x.stream();

  // Arguments check
  PADDLE_ENFORCE_EQ(
      x.dtype(),
      phi::DataType::BFLOAT16,
      common::errors::InvalidArgument("Input X must be bfloat16, but got %s",
                                      phi::DataTypeToString(x.dtype())));

  if (prob) {
    PADDLE_ENFORCE_EQ(prob.get().dtype(),
                      phi::DataType::FLOAT32,
                      common::errors::InvalidArgument(
                          "Input prob must be float32, but got %s",
                          phi::DataTypeToString(prob.get().dtype())));
  }

  auto x_dims = x.dims();
  int64_t rows = phi::product(phi::slice_ddim(x_dims, 0, x_dims.size() - 1));
  int64_t cols = x_dims[x_dims.size() - 1];

  PADDLE_ENFORCE_EQ(cols % 2,
                    0,
                    common::errors::InvalidArgument(
                        "The last dim of Input(X) should be exactly divided "
                        "by 2, but got %d",
                        cols));

  if (prob) {
    PADDLE_ENFORCE_EQ(prob.get().dims()[0],
                      rows,
                      common::errors::InvalidArgument(
                          "The first dim of Input(X) should be equal to the "
                          "first dim of Input(prob) but got X.shape[0]: %d, "
                          "prob.shape[0]: %d",
                          rows,
                          prob.get().dims()[0]));
  }

  paddle::Tensor out;
  paddle::Tensor scale;

  if (use_ue8m0) {
    auto input_dim = x.dims();
    const int64_t token_num = input_dim[0];
    const int64_t hidden_size = input_dim[1];

    constexpr int64_t kMaxRowsForUe8m0ScaleIndex =
        static_cast<int64_t>(std::numeric_limits<size_t>::max() / 4);
    PADDLE_ENFORCE_LE(
        rows,
        kMaxRowsForUe8m0ScaleIndex,
        common::errors::InvalidArgument(
            "rows is too large for ue8m0 scale index calculation, got %ld.",
            rows));

    PADDLE_ENFORCE(hidden_size % 1024 == 0,
                   "hidden_size must be divisible by 1024");
    const int64_t hidden_size_scale = hidden_size / 2 / 128;
    out = GetEmptyTensor(
        {token_num, hidden_size / 2}, phi::DataType::FLOAT8_E4M3FN, x.place());
    const int64_t tma_alignment_bytes = 16;
    const int64_t tma_alignment_elements = tma_alignment_bytes / sizeof(float);

    int64_t padded_token_num =
        ((token_num + tma_alignment_elements - 1) / tma_alignment_elements) *
        tma_alignment_elements;
    scale =
        GetEmptyTensor({padded_token_num, ceil_div_i64(hidden_size_scale, 4)},
                       {1, padded_token_num},
                       paddle::DataType::INT32,
                       x.place());
    if (token_num == 0 || hidden_size == 0) {
      return {out, scale};
    }
  } else {
    scale = GetEmptyTensor(
        {rows, (cols / 2 + 127) / 128}, phi::DataType::FLOAT32, place);
    out = GetEmptyTensor({rows, cols / 2}, phi::DataType::FLOAT8_E4M3FN, place);
    if (rows == 0 || cols == 0) {
      return {out, scale};
    }
  }

  // Get data pointers
  const auto* x_data = x.data<phi::bfloat16>();
  const float* prob_data = prob ? prob.get().data<float>() : nullptr;
  auto* out_data = out.data<phi::float8_e4m3fn>();

  void* scale_data;

  if (use_ue8m0) {
    scale_data = static_cast<void*>(scale.data<int32_t>());
  } else {
    scale_data = static_cast<void*>(scale.data<float>());
  }
  // Launch kernel

  if (use_ue8m0 && cols % 8 == 0) {
    dispatch_fused_spaq<phi::float8_e4m3fn, true, kHasClamp>(x_data,
                                                             prob_data,
                                                             out_data,
                                                             scale_data,
                                                             stream,
                                                             rows,
                                                             cols,
                                                             using_pow2_scaling,
                                                             !!prob,
                                                             clamp_value);
  } else {
    dispatch_fused_spaq<float, false, kHasClamp>(x_data,
                                                 prob_data,
                                                 out_data,
                                                 scale_data,
                                                 stream,
                                                 rows,
                                                 cols,
                                                 using_pow2_scaling,
                                                 !!prob,
                                                 clamp_value);
  }

  return {out, scale};
}

// ---- Python-facing op kernels ----

std::vector<paddle::Tensor> FusedWeightedSwigluActQuantKernel(
    const paddle::Tensor& x,
    const paddle::optional<paddle::Tensor>& prob,
    const bool using_pow2_scaling,
    const bool use_ue8m0) {
  return FusedWeightedSwigluActQuantImpl</*kHasClamp=*/false>(
      x, prob, using_pow2_scaling, use_ue8m0, /*clamp_value=*/0.0);
}

std::vector<paddle::Tensor> FusedWeightedSwigluActQuantClampKernel(
    const paddle::Tensor& x,
    const paddle::optional<paddle::Tensor>& prob,
    const bool using_pow2_scaling,
    const bool use_ue8m0,
    const double clamp_value) {
  return FusedWeightedSwigluActQuantImpl</*kHasClamp=*/true>(
      x, prob, using_pow2_scaling, use_ue8m0, clamp_value);
}

PD_BUILD_OP(fuse_weighted_swiglu_fp8_quant)
    .Inputs({"expert_out_list", paddle::Optional("prob")})
    .Attrs({"using_pow2_scaling: bool", "use_ue8m0: bool"})
    .Outputs({"out", "scale"})
    .SetKernelFn(PD_KERNEL(FusedWeightedSwigluActQuantKernel));

PD_BUILD_OP(fuse_weighted_swiglu_fp8_quant_clamp)
    .Inputs({"expert_out_list", paddle::Optional("prob")})
    .Attrs({"using_pow2_scaling: bool",
            "use_ue8m0: bool",
            "clamp_value: double"})
    .Outputs({"out", "scale"})
    .SetKernelFn(PD_KERNEL(FusedWeightedSwigluActQuantClampKernel));
