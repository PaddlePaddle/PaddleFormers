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

#include <cuda_bf16.h>
#include <cstdint>
#include <limits>
#include <vector>
#include "paddle/extension.h"
#include "utils.h"  // NOLINT

// ==========================================================================
// Utils: Packed Memory Access (128-bit Vectorization)
// ==========================================================================

struct __align__(16) Packed128 {
  int4 data;
};

constexpr int kSwiGLUBlockSize = 256;

// ------------------------------------------------------------------
// Sigmoid implementation
// ------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ float precise_sigmoid(T x) {
  return 1.0f / (1.0f + expf(-static_cast<float>(x)));
}

// ==========================================================================
// Optimized Forward Kernel
//   kHasClamp = false  -> identical math to the original non-clamp kernel.
//                         The `if constexpr` branch is statically eliminated,
//                         so PTX/SASS matches the pre-refactor binary.
//   kHasClamp = true   -> applies elementwise clamp on g and symmetric clamp
//                         on v, then runs the same SwiGLU * scale.
// ==========================================================================
template <typename T, typename ScaleT, int VEC_SIZE, bool kHasClamp = false>
__global__ void VectorizedFusedSwiGLUFwd(const T* __restrict__ x,
                                         const ScaleT* __restrict__ scale,
                                         T* __restrict__ out,
                                         int64_t rows,
                                         int64_t hidden_size,
                                         int64_t row_stride,
                                         double clamp_value = 0.0) {
  int tid = threadIdx.x;
  int64_t lane_idx = static_cast<int64_t>(tid) * VEC_SIZE;
  float cv = static_cast<float>(clamp_value);

  // Grid-stride loop over rows so a bounded gridDim.x can cover any int64
  // rows count (no INT_MAX restriction at host).
  for (int64_t row = static_cast<int64_t>(blockIdx.x); row < rows;
       row += static_cast<int64_t>(gridDim.x)) {
    float s = static_cast<float>(scale[row]);

    for (int64_t col = lane_idx; col < hidden_size;
         col += static_cast<int64_t>(blockDim.x) * VEC_SIZE) {
      int64_t gate_offset = row * row_stride + col;
      int64_t val_offset = gate_offset + hidden_size;
      int64_t out_offset = row * hidden_size + col;

      Packed128 gate_pack =
          *reinterpret_cast<const Packed128*>(&x[gate_offset]);
      Packed128 val_pack = *reinterpret_cast<const Packed128*>(&x[val_offset]);

      T* gate_ptr = reinterpret_cast<T*>(&gate_pack);
      T* val_ptr = reinterpret_cast<T*>(&val_pack);

      T res_buffer[VEC_SIZE];

#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        float g = static_cast<float>(gate_ptr[i]);
        float v = static_cast<float>(val_ptr[i]);

        if constexpr (kHasClamp) {
          g = fminf(g, cv);
          v = fmaxf(fminf(v, cv), -cv);
        }

        float swiglu = (g * precise_sigmoid(g)) * v;
        res_buffer[i] = static_cast<T>(swiglu * s);
      }

      *reinterpret_cast<Packed128*>(&out[out_offset]) =
          *reinterpret_cast<Packed128*>(res_buffer);
    }
  }
}

// ==========================================================================
// Optimized Backward Kernel
//   kHasClamp = false -> identical math to the original non-clamp kernel.
//   kHasClamp = true  -> uses clamped g_eff/v_eff with masks so gradients
//                        stop flowing through saturated entries.
// ==========================================================================
template <typename T, typename ScaleT, int VEC_SIZE, bool kHasClamp = false>
__global__ void VectorizedFusedSwiGLUBwd(const T* __restrict__ x,
                                         const ScaleT* __restrict__ scale,
                                         const T* __restrict__ d_out,
                                         T* __restrict__ d_x,
                                         ScaleT* __restrict__ d_scale,
                                         int64_t rows,
                                         int64_t hidden_size,
                                         int64_t row_stride,
                                         double clamp_value = 0.0) {
  int tid = threadIdx.x;
  int64_t lane_idx = static_cast<int64_t>(tid) * VEC_SIZE;
  float cv = static_cast<float>(clamp_value);

  __shared__ float shared_sum[kSwiGLUBlockSize];

  // Grid-stride loop over rows so a bounded gridDim.x can cover any int64
  // rows count (no INT_MAX restriction at host).
  for (int64_t row = static_cast<int64_t>(blockIdx.x); row < rows;
       row += static_cast<int64_t>(gridDim.x)) {
    float local_d_scale_sum = 0.0f;
    float s = static_cast<float>(scale[row]);

    for (int64_t col = lane_idx; col < hidden_size;
         col += static_cast<int64_t>(blockDim.x) * VEC_SIZE) {
      int64_t gate_offset = row * row_stride + col;
      int64_t val_offset = gate_offset + hidden_size;
      int64_t out_offset = row * hidden_size + col;

      Packed128 gate_pack =
          *reinterpret_cast<const Packed128*>(&x[gate_offset]);
      Packed128 val_pack = *reinterpret_cast<const Packed128*>(&x[val_offset]);
      Packed128 dout_pack =
          *reinterpret_cast<const Packed128*>(&d_out[out_offset]);

      T* gate_ptr = reinterpret_cast<T*>(&gate_pack);
      T* val_ptr = reinterpret_cast<T*>(&val_pack);
      T* dout_ptr = reinterpret_cast<T*>(&dout_pack);

      T dg_buffer[VEC_SIZE];
      T dv_buffer[VEC_SIZE];

#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        float g = static_cast<float>(gate_ptr[i]);
        float v = static_cast<float>(val_ptr[i]);
        float dout = static_cast<float>(dout_ptr[i]);

        float g_eff, v_eff, g_mask, v_mask;
        if constexpr (kHasClamp) {
          g_eff = fminf(g, cv);
          v_eff = fmaxf(fminf(v, cv), -cv);
          g_mask = (g <= cv) ? 1.0f : 0.0f;
          v_mask = (v <= cv && v >= -cv) ? 1.0f : 0.0f;
        } else {
          g_eff = g;
          v_eff = v;
          // unused under kHasClamp=false; kept defined to avoid maybe-uninit
          g_mask = 1.0f;
          v_mask = 1.0f;
        }

        float sig_g = precise_sigmoid(g_eff);
        float silu_g = g_eff * sig_g;
        float swiglu_val = silu_g * v_eff;

        //   sum(swiglu_val.cast(dtype) * d_out.cast(scale_dtype))
        // Same-type multiply preserves native precision (bf16*bf16→bf16)
        // matching reference bit-exact; mixed types promote to float.
        // Only applied to the clamp path to avoid affecting non-clamp
        // numerical behaviour.
        if constexpr (kHasClamp) {
          if constexpr (std::is_same_v<T, ScaleT>) {
            local_d_scale_sum += static_cast<float>(
                static_cast<T>(swiglu_val) * static_cast<ScaleT>(dout_ptr[i]));
          } else {
            local_d_scale_sum +=
                static_cast<float>(static_cast<T>(swiglu_val)) *
                static_cast<float>(static_cast<ScaleT>(dout_ptr[i]));
          }
        } else {
          local_d_scale_sum += dout * swiglu_val;
        }

        float d_u = dout * s;

        if constexpr (kHasClamp) {
          dv_buffer[i] = static_cast<T>(d_u * silu_g * v_mask);
          float d_g_val =
              d_u * sig_g * (1.0f + g_eff * (1.0f - sig_g)) * v_eff * g_mask;
          dg_buffer[i] = static_cast<T>(d_g_val);
        } else {
          dv_buffer[i] = static_cast<T>(d_u * silu_g);
          float d_g_val = d_u * v_eff * sig_g * (1.0f + g_eff * (1.0f - sig_g));
          dg_buffer[i] = static_cast<T>(d_g_val);
        }
      }

      *reinterpret_cast<Packed128*>(&d_x[gate_offset]) =
          *reinterpret_cast<Packed128*>(dg_buffer);
      *reinterpret_cast<Packed128*>(&d_x[val_offset]) =
          *reinterpret_cast<Packed128*>(dv_buffer);
    }

    shared_sum[tid] = local_d_scale_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        shared_sum[tid] += shared_sum[tid + stride];
      }
      __syncthreads();
    }

    if (tid == 0) {
      d_scale[row] = static_cast<ScaleT>(shared_sum[0]);
    }
    // Ensure all threads complete the write/read of shared_sum before the
    // next row iteration overwrites it.
    __syncthreads();
  }
}

// ==========================================================================
// Combined Forward + Weighted Backward Kernel
//   Computes dx, dprobs, and forward output in a single fused launch.
//   Supports both kHasClamp=true and kHasClamp=false via constexpr.
//   Replaces the two-call pattern in the clamp path and can also replace
//   external FusedQuantOps._fused_swiglu_probs_bwd / paddle's
//   fused_swiglu_weighted_bwd.
// ==========================================================================
template <typename T, typename ScaleT, int VEC_SIZE, bool kHasClamp = false>
__global__ void VectorizedFusedSwiGLUWeightedBwd(
    const T* __restrict__ x,
    const ScaleT* __restrict__ probs,
    const T* __restrict__ d_out,
    T* __restrict__ d_x,
    ScaleT* __restrict__ d_probs,
    T* __restrict__ out,  // forward result: silu(clamp(g)) * clamp(v) * probs
    int64_t rows,
    int64_t hidden_size,
    int64_t row_stride,
    double clamp_value = 0.0) {
  int tid = threadIdx.x;
  int64_t lane_idx = static_cast<int64_t>(tid) * VEC_SIZE;
  float cv = static_cast<float>(clamp_value);

  __shared__ float shared_sum[kSwiGLUBlockSize];

  // Grid-stride loop over rows so a bounded gridDim.x can cover any int64
  // rows count (no INT_MAX restriction at host).
  for (int64_t row = static_cast<int64_t>(blockIdx.x); row < rows;
       row += static_cast<int64_t>(gridDim.x)) {
    float local_d_probs_sum = 0.0f;
    float p = static_cast<float>(probs[row]);

    for (int64_t col = lane_idx; col < hidden_size;
         col += static_cast<int64_t>(blockDim.x) * VEC_SIZE) {
      int64_t gate_offset = row * row_stride + col;
      int64_t val_offset = gate_offset + hidden_size;
      int64_t out_offset = row * hidden_size + col;

      Packed128 gate_pack =
          *reinterpret_cast<const Packed128*>(&x[gate_offset]);
      Packed128 val_pack = *reinterpret_cast<const Packed128*>(&x[val_offset]);
      Packed128 dout_pack =
          *reinterpret_cast<const Packed128*>(&d_out[out_offset]);

      T* gate_ptr = reinterpret_cast<T*>(&gate_pack);
      T* val_ptr = reinterpret_cast<T*>(&val_pack);
      T* dout_ptr = reinterpret_cast<T*>(&dout_pack);

      T dg_buffer[VEC_SIZE];
      T dv_buffer[VEC_SIZE];
      T out_buffer[VEC_SIZE];

#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        float g = static_cast<float>(gate_ptr[i]);
        float v = static_cast<float>(val_ptr[i]);
        float dout = static_cast<float>(dout_ptr[i]);

        float g_eff, v_eff, g_mask, v_mask;
        if constexpr (kHasClamp) {
          g_eff = fminf(g, cv);
          v_eff = fmaxf(fminf(v, cv), -cv);
          g_mask = (g <= cv) ? 1.0f : 0.0f;
          v_mask = (v <= cv && v >= -cv) ? 1.0f : 0.0f;
        } else {
          g_eff = g;
          v_eff = v;
          g_mask = 1.0f;
          v_mask = 1.0f;
        }

        float sig_g = precise_sigmoid(g_eff);
        float silu_g = g_eff * sig_g;
        float swiglu_val = silu_g * v_eff;

        // forward result
        out_buffer[i] = static_cast<T>(swiglu_val * p);

        //   sum(swiglu_val.cast(dtype) * d_out.cast(probs_dtype))
        // Same-type multiply preserves native precision (bf16*bf16→bf16)
        // matching reference bit-exact; only applied to clamp path.
        if constexpr (kHasClamp) {
          if constexpr (std::is_same_v<T, ScaleT>) {
            local_d_probs_sum += static_cast<float>(
                static_cast<T>(swiglu_val) * static_cast<ScaleT>(dout_ptr[i]));
          } else {
            local_d_probs_sum +=
                static_cast<float>(static_cast<T>(swiglu_val)) *
                static_cast<float>(static_cast<ScaleT>(dout_ptr[i]));
          }
        } else {
          local_d_probs_sum += dout * swiglu_val;
        }

        // d_u = dout * p
        float d_u = dout * p;

        if constexpr (kHasClamp) {
          dv_buffer[i] = static_cast<T>(d_u * silu_g * v_mask);
          float d_g_val =
              d_u * sig_g * (1.0f + g_eff * (1.0f - sig_g)) * v_eff * g_mask;
          dg_buffer[i] = static_cast<T>(d_g_val);
        } else {
          dv_buffer[i] = static_cast<T>(d_u * silu_g);
          float d_g_val = d_u * v_eff * sig_g * (1.0f + g_eff * (1.0f - sig_g));
          dg_buffer[i] = static_cast<T>(d_g_val);
        }
      }

      *reinterpret_cast<Packed128*>(&d_x[gate_offset]) =
          *reinterpret_cast<Packed128*>(dg_buffer);
      *reinterpret_cast<Packed128*>(&d_x[val_offset]) =
          *reinterpret_cast<Packed128*>(dv_buffer);
      *reinterpret_cast<Packed128*>(&out[out_offset]) =
          *reinterpret_cast<Packed128*>(out_buffer);
    }

    shared_sum[tid] = local_d_probs_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        shared_sum[tid] += shared_sum[tid + stride];
      }
      __syncthreads();
    }

    if (tid == 0) {
      d_probs[row] = static_cast<ScaleT>(shared_sum[0]);
    }
    // Ensure all threads complete the write/read of shared_sum before the
    // next row iteration overwrites it.
    __syncthreads();
  }
}

// ==========================================================================
// Host Wrappers (templated on kHasClamp; non-clamp wrappers forward 0.0)
// ==========================================================================

template <bool kHasClamp>
static std::vector<paddle::Tensor> FusedSwiGLUScaleForwardImpl(
    const paddle::Tensor& x, const paddle::Tensor& scale, double clamp_value) {
  auto rows = x.shape()[0];
  auto hidden2 = x.shape()[1];
  auto hidden_size = hidden2 / 2;
  auto out = paddle::empty({rows, hidden_size}, x.dtype(), x.place());

  if (rows == 0 || hidden_size == 0) {
    return {out};
  }

  // Paddle extension gridDim is int. The kernel uses a grid-stride loop
  // over rows, so we cap grid_size at kMaxSwiGLUGridSize and let the kernel
  // chunk arbitrary int64 rows on device.
  int grid_size = GetSwiGLURowGridSize(rows);
  int block_size = kSwiGLUBlockSize;
  auto stream = x.stream();

  if (x.dtype() == paddle::DataType::BFLOAT16) {
    using paddle_bf16 = paddle::bfloat16;
    using cuda_bf16 = __nv_bfloat16;
    if (scale.dtype() == paddle::DataType::FLOAT32) {
      VectorizedFusedSwiGLUFwd<cuda_bf16, float, 8, kHasClamp>
          <<<grid_size, block_size, 0, stream>>>(
              reinterpret_cast<const cuda_bf16*>(x.data<paddle_bf16>()),
              scale.data<float>(),
              reinterpret_cast<cuda_bf16*>(out.data<paddle_bf16>()),
              rows,
              hidden_size,
              hidden2,
              clamp_value);
    } else {
      VectorizedFusedSwiGLUFwd<cuda_bf16, cuda_bf16, 8, kHasClamp>
          <<<grid_size, block_size, 0, stream>>>(
              reinterpret_cast<const cuda_bf16*>(x.data<paddle_bf16>()),
              reinterpret_cast<const cuda_bf16*>(scale.data<paddle_bf16>()),
              reinterpret_cast<cuda_bf16*>(out.data<paddle_bf16>()),
              rows,
              hidden_size,
              hidden2,
              clamp_value);
    }
  } else if (x.dtype() == paddle::DataType::FLOAT32) {
    VectorizedFusedSwiGLUFwd<float, float, 4, kHasClamp>
        <<<grid_size, block_size, 0, stream>>>(x.data<float>(),
                                               scale.data<float>(),
                                               out.data<float>(),
                                               rows,
                                               hidden_size,
                                               hidden2,
                                               clamp_value);
  }
  return {out};
}

template <bool kHasClamp>
static std::vector<paddle::Tensor> FusedSwiGLUScaleBackwardImpl(
    const paddle::Tensor& x,
    const paddle::Tensor& scale,
    const paddle::Tensor& d_out,
    double clamp_value) {
  auto rows = x.shape()[0];
  auto hidden2 = x.shape()[1];
  auto hidden_size = hidden2 / 2;
  auto d_x = paddle::empty_like(x);
  // Align d_scale shape: keepdim semantics -> [rows, 1] for clamp path,
  // empty_like for non-clamp (pre-existing behaviour).
  auto d_scale = paddle::empty_like(scale);
  if constexpr (kHasClamp) {
    d_scale = paddle::empty({rows, 1}, scale.dtype(), x.place());
  }

  if (rows == 0 || hidden_size == 0) {
    return {d_x, d_scale};
  }

  // Paddle extension gridDim is int. The kernel uses a grid-stride loop
  // over rows, so we cap grid_size at kMaxSwiGLUGridSize and let the kernel
  // chunk arbitrary int64 rows on device.
  int grid_size = GetSwiGLURowGridSize(rows);
  int block_size = kSwiGLUBlockSize;
  auto stream = x.stream();

  if (x.dtype() == paddle::DataType::BFLOAT16) {
    using paddle_bf16 = paddle::bfloat16;
    using cuda_bf16 = __nv_bfloat16;
    if (scale.dtype() == paddle::DataType::FLOAT32) {
      VectorizedFusedSwiGLUBwd<cuda_bf16, float, 8, kHasClamp>
          <<<grid_size, block_size, 0, stream>>>(
              reinterpret_cast<const cuda_bf16*>(x.data<paddle_bf16>()),
              scale.data<float>(),
              reinterpret_cast<const cuda_bf16*>(d_out.data<paddle_bf16>()),
              reinterpret_cast<cuda_bf16*>(d_x.data<paddle_bf16>()),
              d_scale.data<float>(),
              rows,
              hidden_size,
              hidden2,
              clamp_value);
    } else {
      VectorizedFusedSwiGLUBwd<cuda_bf16, cuda_bf16, 8, kHasClamp>
          <<<grid_size, block_size, 0, stream>>>(
              reinterpret_cast<const cuda_bf16*>(x.data<paddle_bf16>()),
              reinterpret_cast<const cuda_bf16*>(scale.data<paddle_bf16>()),
              reinterpret_cast<const cuda_bf16*>(d_out.data<paddle_bf16>()),
              reinterpret_cast<cuda_bf16*>(d_x.data<paddle_bf16>()),
              reinterpret_cast<cuda_bf16*>(d_scale.data<paddle_bf16>()),
              rows,
              hidden_size,
              hidden2,
              clamp_value);
    }
  } else if (x.dtype() == paddle::DataType::FLOAT32) {
    VectorizedFusedSwiGLUBwd<float, float, 4, kHasClamp>
        <<<grid_size, block_size, 0, stream>>>(x.data<float>(),
                                               scale.data<float>(),
                                               d_out.data<float>(),
                                               d_x.data<float>(),
                                               d_scale.data<float>(),
                                               rows,
                                               hidden_size,
                                               hidden2,
                                               clamp_value);
  }
  return {d_x, d_scale};
}

// ==========================================================================
// Weighted Backward Host Wrapper (combines forward+backward in one launch)
// ==========================================================================
template <bool kHasClamp>
static std::vector<paddle::Tensor> FusedSwiGLUWeightedBackwardImpl(
    const paddle::Tensor& x,
    const paddle::Tensor& probs,
    const paddle::Tensor& d_out,
    double clamp_value) {
  int64_t rows = x.shape()[0];
  int64_t hidden2 = x.shape()[1];
  int64_t hidden_size = hidden2 / 2;
  auto d_x = paddle::empty_like(x);
  auto d_probs = paddle::empty_like(probs);
  if constexpr (kHasClamp) {
    d_probs = paddle::empty({rows, 1}, probs.dtype(), x.place());
  }
  auto out = paddle::empty({rows, hidden_size}, x.dtype(), x.place());

  if (rows == 0 || hidden_size == 0) {
    return {d_x, d_probs, out};
  }

  // Paddle extension gridDim is int. The kernel uses a grid-stride loop
  // over rows, so we cap grid_size at kMaxSwiGLUGridSize and let the kernel
  // chunk arbitrary int64 rows on device.
  int grid_size = GetSwiGLURowGridSize(rows);
  int block_size = kSwiGLUBlockSize;
  auto stream = x.stream();

  if (x.dtype() == paddle::DataType::BFLOAT16) {
    using paddle_bf16 = paddle::bfloat16;
    using cuda_bf16 = __nv_bfloat16;
    if (probs.dtype() == paddle::DataType::FLOAT32) {
      VectorizedFusedSwiGLUWeightedBwd<cuda_bf16, float, 8, kHasClamp>
          <<<grid_size, block_size, 0, stream>>>(
              reinterpret_cast<const cuda_bf16*>(x.data<paddle_bf16>()),
              probs.data<float>(),
              reinterpret_cast<const cuda_bf16*>(d_out.data<paddle_bf16>()),
              reinterpret_cast<cuda_bf16*>(d_x.data<paddle_bf16>()),
              d_probs.data<float>(),
              reinterpret_cast<cuda_bf16*>(out.data<paddle_bf16>()),
              rows,
              hidden_size,
              hidden2,
              clamp_value);
    } else {
      VectorizedFusedSwiGLUWeightedBwd<cuda_bf16, cuda_bf16, 8, kHasClamp>
          <<<grid_size, block_size, 0, stream>>>(
              reinterpret_cast<const cuda_bf16*>(x.data<paddle_bf16>()),
              reinterpret_cast<const cuda_bf16*>(probs.data<paddle_bf16>()),
              reinterpret_cast<const cuda_bf16*>(d_out.data<paddle_bf16>()),
              reinterpret_cast<cuda_bf16*>(d_x.data<paddle_bf16>()),
              reinterpret_cast<cuda_bf16*>(d_probs.data<paddle_bf16>()),
              reinterpret_cast<cuda_bf16*>(out.data<paddle_bf16>()),
              rows,
              hidden_size,
              hidden2,
              clamp_value);
    }
  } else if (x.dtype() == paddle::DataType::FLOAT32) {
    VectorizedFusedSwiGLUWeightedBwd<float, float, 4, kHasClamp>
        <<<grid_size, block_size, 0, stream>>>(x.data<float>(),
                                               probs.data<float>(),
                                               d_out.data<float>(),
                                               d_x.data<float>(),
                                               d_probs.data<float>(),
                                               out.data<float>(),
                                               rows,
                                               hidden_size,
                                               hidden2,
                                               clamp_value);
  }
  return {d_x, d_probs, out};
}

// ----- Op-facing wrappers -----

std::vector<paddle::Tensor> FusedSwiGLUScaleForward(
    const paddle::Tensor& x, const paddle::Tensor& scale) {
  return FusedSwiGLUScaleForwardImpl</*kHasClamp=*/false>(x, scale, 0.0);
}

std::vector<paddle::Tensor> FusedSwiGLUScaleBackward(
    const paddle::Tensor& x,
    const paddle::Tensor& scale,
    const paddle::Tensor& d_out) {
  return FusedSwiGLUScaleBackwardImpl</*kHasClamp=*/false>(
      x, scale, d_out, 0.0);
}

std::vector<paddle::Tensor> FusedSwiGLUScaleClampForward(
    const paddle::Tensor& x, const paddle::Tensor& scale, double clamp_value) {
  return FusedSwiGLUScaleForwardImpl</*kHasClamp=*/true>(x, scale, clamp_value);
}

std::vector<paddle::Tensor> FusedSwiGLUScaleClampBackward(
    const paddle::Tensor& x,
    const paddle::Tensor& scale,
    const paddle::Tensor& d_out,
    double clamp_value) {
  return FusedSwiGLUScaleBackwardImpl</*kHasClamp=*/true>(
      x, scale, d_out, clamp_value);
}

std::vector<paddle::Tensor> FusedSwiGLUWeightedClampBackward(
    const paddle::Tensor& x,
    const paddle::Tensor& probs,
    const paddle::Tensor& d_out,
    double clamp_value) {
  return FusedSwiGLUWeightedBackwardImpl</*kHasClamp=*/true>(
      x, probs, d_out, clamp_value);
}

// ==========================================================================
// Op Registration
// ==========================================================================

std::vector<std::vector<int64_t>> FusedGradInferShape(
    std::vector<int64_t> x_shape,
    std::vector<int64_t> scale_shape,
    std::vector<int64_t> dout_shape) {
  return {x_shape, scale_shape};
}

std::vector<paddle::DataType> FusedGradInferDtype(paddle::DataType x_dtype,
                                                  paddle::DataType scale_dtype,
                                                  paddle::DataType dout_dtype) {
  return {x_dtype, scale_dtype};
}

// Forward: output is SwiGLU(x) * scale with shape {rows, hidden_size/2}
std::vector<std::vector<int64_t>> FusedFwdInferShape(
    std::vector<int64_t> x_shape, std::vector<int64_t> scale_shape) {
  return {{x_shape[0], x_shape[1] / 2}};
}

std::vector<paddle::DataType> FusedFwdInferDtype(paddle::DataType x_dtype,
                                                 paddle::DataType scale_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(fused_swiglu_scale_bwd)
    .Inputs({"X", "Scale", "DOut"})
    .Outputs({"DX", "DScale"})
    .SetKernelFn(PD_KERNEL(FusedSwiGLUScaleBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedGradInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedGradInferDtype));

PD_BUILD_OP(fused_swiglu_scale)
    .Inputs({"X", "Scale"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(FusedSwiGLUScaleForward))
    .SetInferShapeFn(
        PD_INFER_SHAPE(FusedGradInferShape))  // Reuse infer shape logic
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedGradInferDtype));

PD_BUILD_GRAD_OP(fused_swiglu_scale)
    .Inputs({"X", "Scale", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X"), paddle::Grad("Scale")})
    .SetKernelFn(PD_KERNEL(FusedSwiGLUScaleBackward));

// Clamp InferShape: d_scale always [rows, 1] matching Megatron keepdim
// semantics.
std::vector<std::vector<int64_t>> FusedGradClampInferShape(
    std::vector<int64_t> x_shape,
    std::vector<int64_t> scale_shape,
    std::vector<int64_t> dout_shape) {
  return {x_shape, {x_shape[0], 1}};
}

PD_BUILD_OP(fused_swiglu_scale_clamp_bwd)
    .Inputs({"X", "Scale", "DOut"})
    .Outputs({"DX", "DScale"})
    .Attrs({"clamp_value: double"})
    .SetKernelFn(PD_KERNEL(FusedSwiGLUScaleClampBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedGradClampInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedGradInferDtype));

PD_BUILD_OP(fused_swiglu_scale_clamp)
    .Inputs({"X", "Scale"})
    .Outputs({"Out"})
    .Attrs({"clamp_value: double"})
    .SetKernelFn(PD_KERNEL(FusedSwiGLUScaleClampForward))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedFwdInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedFwdInferDtype));

PD_BUILD_GRAD_OP(fused_swiglu_scale_clamp)
    .Inputs({"X", "Scale", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X"), paddle::Grad("Scale")})
    .Attrs({"clamp_value: double"})
    .SetKernelFn(PD_KERNEL(FusedSwiGLUScaleClampBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedGradClampInferShape));

// ---- Weighted backward (fused forward + backward in one launch) ----

std::vector<std::vector<int64_t>> WeightedBwdInferShape(
    std::vector<int64_t> x_shape,
    std::vector<int64_t> probs_shape,
    std::vector<int64_t> dout_shape) {
  return {
      x_shape, probs_shape, {x_shape[0], x_shape[1] / 2}};  // dx, dprobs, out
}

std::vector<paddle::DataType> WeightedBwdInferDtype(
    paddle::DataType x_dtype,
    paddle::DataType probs_dtype,
    paddle::DataType dout_dtype) {
  return {x_dtype, probs_dtype, x_dtype};
}

// Clamp InferShape: d_probs always [rows, 1] matching Megatron keepdim
// semantics.
std::vector<std::vector<int64_t>> WeightedBwdClampInferShape(
    std::vector<int64_t> x_shape,
    std::vector<int64_t> probs_shape,
    std::vector<int64_t> dout_shape) {
  return {x_shape, {x_shape[0], 1}, {x_shape[0], x_shape[1] / 2}};
}

PD_BUILD_OP(fused_swiglu_weighted_clamp_bwd)
    .Inputs({"X", "Probs", "DOut"})
    .Outputs({"DX", "DProbs", "Out"})
    .Attrs({"clamp_value: double"})
    .SetKernelFn(PD_KERNEL(FusedSwiGLUWeightedClampBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(WeightedBwdClampInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(WeightedBwdInferDtype));
