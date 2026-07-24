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

#include <limits> // NOLINT

#include "paddle/phi/kernels/funcs/fast_divmod.h"              // NOLINT
#include "paddle/phi/kernels/funcs/segmented_array.h"          // NOLINT
#include "paddle/phi/kernels/fusion/gpu/quant_utils.h"         // NOLINT
#include "paddle/phi/kernels/primitive/datamover_primitives.h" // NOLINT

using FastDivMod = phi::funcs::FastDivMod<int64_t>;

template <typename ScaleT, bool using_ue8m0_scale>
__device__ __forceinline__ void StoreScaleFleetCustom(ScaleT *ptr, size_t idx,
                                                      float val) {
  if constexpr (using_ue8m0_scale) {
    int exp = (__float_as_int(val) >> 23) & 0xFF;
    reinterpret_cast<uint8_t *>(ptr)[idx] = static_cast<uint8_t>(exp);
  } else {
    ptr[idx] = val;
  }
}

template <typename ArrayT>
__device__ void BlockLoadFleetCustom(ArrayT input_array, __nv_bfloat16 x[8][4],
                                     size_t K, size_t block_y, size_t block_x) {
  const __nv_bfloat16 *input =
      reinterpret_cast<const __nv_bfloat16 *>(input_array.data[blockIdx.z]);

  for (size_t i = 0; i < 8; i++) {
    size_t idx_m = block_y * 128 + static_cast<size_t>(threadIdx.y) + i * 16;
    size_t idx_k = block_x * 128 + static_cast<size_t>(threadIdx.x) * 4;
    size_t idx = idx_m * K + idx_k;

    using LoadT = phi::kps::details::VectorType<__nv_bfloat16, 4>;
    LoadT data = *reinterpret_cast<const LoadT *>(input + idx);
    for (int j = 0; j < 4; j++) {
      x[i][j] = data.val[j];
    }
  }
}

template <int Width = 32>
__device__ __nv_bfloat16 WarpReduceMaxFleetCustom(__nv_bfloat16 x) {
  constexpr unsigned mask = (uint64_t(1) << Width) - 1;
#pragma unroll
  for (int offset = Width / 2; offset > 0; offset /= 2) {
    __nv_bfloat16 t = __shfl_down_sync(mask, x, offset);
    x = __hmax(x, t);
  }
  return x;
}

template <typename OutT, bool Power2Scaling = false>
__device__ float BlockReduceScaleFleetCustom(__nv_bfloat16 x[8][4],
                                             float eps = 1e-10f) {
  __nv_bfloat16 local_max = 0.0;
#pragma unroll
  for (uint32_t i = 0; i < 8; i++) {
    __nv_bfloat162 v0 = *reinterpret_cast<__nv_bfloat162 *>(&x[i][0]);
    __nv_bfloat162 v1 = *reinterpret_cast<__nv_bfloat162 *>(&x[i][2]);
    v0 = __habs2(v0);
    v1 = __habs2(v1);
    v0 = __hmax2(v1, v0);
    local_max = __hmax(__hmax(v0.x, v0.y), local_max);
  }

  __nv_bfloat16 warp_max = WarpReduceMaxFleetCustom<32>(local_max);

  __shared__ __nv_bfloat16 block_max[16];
  __shared__ float block_scale;
  if (threadIdx.x == 0) {
    block_max[threadIdx.y] = warp_max;
  }
  __syncthreads();
  if (threadIdx.y == 0 && threadIdx.x < 16) {
    warp_max = WarpReduceMaxFleetCustom<16>(block_max[threadIdx.x]);
    if (threadIdx.x == 0) {
      block_scale =
          ComputeScale<__nv_bfloat16, OutT, Power2Scaling>(warp_max, eps);
    }
  }
  __syncthreads();

  return block_scale;
}

template <typename OutT, typename ArrayT, typename ScaleT,
          bool using_pow2_scaling, bool using_ue8m0_scale,
          bool output_scale_transpose>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 900))
// Force nvcc to squash register to avoid low occupancy
// in architecture after Hopper
__global__ void __launch_bounds__(512, 4)
#else
__global__ void __launch_bounds__(512)
#endif
    FusedStackQuantGPUKernelFleetCustom(ArrayT input_array,
                                        OutT *__restrict__ out,
                                        ScaleT *__restrict__ scale, size_t M,
                                        size_t K, FastDivMod K_div_128) {
  size_t block_y = K_div_128.Div(blockIdx.x);
  size_t block_x = static_cast<size_t>(blockIdx.x) - block_y * (K / 128);

  // Load 128x128 elements from X
  __nv_bfloat16 x[8][4];
  BlockLoadFleetCustom(input_array, x, K, block_y, block_x);

  // Find the scale of all elements
  float block_scale = BlockReduceScaleFleetCustom < OutT,
        using_pow2_scaling || using_ue8m0_scale > (x);

  // Compute scale and store back
  // For FusedStackQuant, logical layout: Rows=N*M, Cols=K
  // block_y -> idx_m (row block in M/128), block_x -> idx_k (col block in
  // K/128) idx_n -> blockIdx.z
  int tid = threadIdx.y * 32 + threadIdx.x;
  if constexpr (using_ue8m0_scale) {
    if (tid < 128) {
      size_t r = tid;
      size_t global_row =
          (static_cast<size_t>(blockIdx.z) * (M / 128) + block_y) * 128 + r;
      size_t idx;
      if constexpr (output_scale_transpose) {
        // [K/128, N*M]
        // idx = block_x * (static_cast<size_t>(gridDim.z) * M) + global_row;
        size_t total_cols = static_cast<size_t>(gridDim.z) * M;
        idx = (block_x / 4) * (total_cols * 4) + global_row * 4 + (block_x % 4);
      } else {
        // [N*M, K/128]
        idx = global_row * (K / 128) + block_x;
      }
      StoreScaleFleetCustom<ScaleT, using_ue8m0_scale>(scale, idx,
                                                       __frcp_rn(block_scale));
    }
  } else {
    if (tid == 0) {
      size_t idx;
      if constexpr (output_scale_transpose) {
        // [K/128, N*M/128]
        idx = block_x * (static_cast<size_t>(gridDim.z) * (M / 128)) +
              (blockIdx.z * (M / 128) + block_y);
      } else {
        // [N*M/128, K/128]
        idx = (blockIdx.z * (M / 128) + block_y) * (K / 128) + block_x;
      }
      StoreScaleFleetCustom<ScaleT, using_ue8m0_scale>(scale, idx,
                                                       __frcp_rn(block_scale));
    }
  }

  // Scale X and store to out
  for (uint32_t i = 0; i < 8; i++) {
    size_t idx_n = blockIdx.z;
    size_t idx_m = block_y * 128 + static_cast<size_t>(threadIdx.y) + i * 16;
    size_t idx_k = block_x * 128 + static_cast<size_t>(threadIdx.x) * 4;
    size_t idx = (idx_n * M + idx_m) * K + idx_k;

    using StoreT = phi::kps::details::VectorType<OutT, 4>;
    StoreT data;
    for (uint32_t j = 0; j < 4; j++) {
      float x_fp32 = static_cast<float>(x[i][j]);
      float output_scaled = x_fp32 * block_scale;
      data.val[j] = static_cast<OutT>(output_scaled);
    }
    *reinterpret_cast<StoreT *>(out + idx) = data;
  }
}

template <typename OutT, typename ArrayT, typename ScaleT,
          bool using_pow2_scaling, bool using_ue8m0_scale,
          bool output_scale_transpose>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 900))
// Force nvcc to squash register to avoid low occupancy
// in architecture after Hopper
__global__ void __launch_bounds__(512, 4)
#else
__global__ void __launch_bounds__(512)
#endif
    FusedStackTransposeQuantGPUKernelFleetCustom(ArrayT input_array,
                                                 OutT *__restrict__ out,
                                                 ScaleT *__restrict__ scale,
                                                 size_t M, size_t K,
                                                 FastDivMod K_div_128) {
  size_t block_y = K_div_128.Div(blockIdx.x);
  size_t block_x = static_cast<size_t>(blockIdx.x) - block_y * (K / 128);

  // Load 128x128 elements from X
  __nv_bfloat16 x[8][4];
  BlockLoadFleetCustom(input_array, x, K, block_y, block_x);

  // Find the scale of all elements
  float block_scale = BlockReduceScaleFleetCustom < OutT,
        using_pow2_scaling || using_ue8m0_scale > (x);

  // Compute scale and store back
  // For FusedStackTransposeQuant, logical layout: Rows=N*K, Cols=M
  // block_y -> idx_m (col block in M/128), block_x -> idx_k (row block in
  // K/128) idx_n -> blockIdx.z
  int tid = threadIdx.y * 32 + threadIdx.x;
  if constexpr (using_ue8m0_scale) {
    if (tid < 128) {
      size_t r = tid;
      size_t global_row =
          (static_cast<size_t>(blockIdx.z) * (K / 128) + block_x) * 128 + r;
      size_t idx;
      if constexpr (output_scale_transpose) {
        // [M/128, N*K]
        // idx = block_y * (static_cast<size_t>(gridDim.z) * K) + global_row;
        size_t total_rows = static_cast<size_t>(gridDim.z) * K;
        idx = (block_y / 4) * (total_rows * 4) + global_row * 4 + (block_y % 4);
      } else {
        // [N*K, M/128]
        idx = global_row * (M / 128) + block_y;
      }
      StoreScaleFleetCustom<ScaleT, using_ue8m0_scale>(scale, idx,
                                                       __frcp_rn(block_scale));
    }
  } else {
    if (tid == 0) {
      size_t idx;
      if constexpr (output_scale_transpose) {
        // [M/128, N*K/128]
        idx = block_y * (static_cast<size_t>(gridDim.z) * (K / 128)) +
              (blockIdx.z * (K / 128) + block_x);
      } else {
        // [N*K/128, M/128]
        idx = (blockIdx.z * (K / 128) + block_x) * (M / 128) + block_y;
      }
      StoreScaleFleetCustom<ScaleT, using_ue8m0_scale>(scale, idx,
                                                       __frcp_rn(block_scale));
    }
  }

  // Scale X and transpose in shared memory
  __shared__ OutT shm[128][129];
  for (uint32_t i = 0; i < 8; i++) {
    for (uint32_t j = 0; j < 4; j++) {
      float x_fp32 = static_cast<float>(x[i][j]);
      float output_scaled = x_fp32 * block_scale;
      shm[threadIdx.x * 4 + j][i * 16 + threadIdx.y] =
          static_cast<OutT>(output_scaled);
    }
  }
  __syncthreads();

  // Store X back to out
  for (uint32_t i = 0; i < 8; i++) {
    size_t idx_n = blockIdx.z;
    size_t idx_k = block_x * 128 + static_cast<size_t>(threadIdx.y) + i * 16;
    size_t idx_m = block_y * 128 + static_cast<size_t>(threadIdx.x) * 4;
    size_t idx = (idx_n * K + idx_k) * M + idx_m;

    using StoreT = phi::kps::details::VectorType<OutT, 4>;
    StoreT data;
    for (uint32_t j = 0; j < 4; j++) {
      data.val[j] = shm[i * 16 + threadIdx.y][threadIdx.x * 4 + j];
    }
    *reinterpret_cast<StoreT *>(out + idx) = data;
  }
}

std::tuple<int64_t, int64_t, int64_t>
FusedStackQuantCommonCheckFleetCustom(const std::vector<paddle::Tensor> &x) {
  PADDLE_ENFORCE_GT(x.size(), 0UL,
                    common::errors::InvalidArgument(
                        "Number of Inputs(x) must be larger than 0, but"
                        " received value is:%d.",
                        x.size()));
  int64_t N = x.size();
  for (int i = 0; i < N; ++i) {
    PADDLE_ENFORCE_EQ(
        x[i].dtype(), phi::DataType::BFLOAT16,
        common::errors::InvalidArgument(
            "input must be bfloat16, but received dtype: %s", x[i].dtype()));
  }
  auto input_dims = x[0].dims();
  PADDLE_ENFORCE_EQ(
      input_dims.size(), 2U,
      common::errors::InvalidArgument(
          "input must be 2-D, but received dims: %s", input_dims.to_str()));
  int64_t M = input_dims[0];
  int64_t K = input_dims[1];
  for (int i = 1; i < N; ++i) {
    input_dims = x[i].dims();
    PADDLE_ENFORCE_EQ(input_dims.size(), 2U,
                      common::errors::InvalidArgument(
                          "input must be 2-D, but received input[%d] dims: %s",
                          i, input_dims.to_str()));
    PADDLE_ENFORCE_EQ(
        input_dims[0], M,
        common::errors::InvalidArgument(
            "input [%d] must be shape %d, %d, but received dims: %s", i, M, K,
            input_dims.to_str()));
    PADDLE_ENFORCE_EQ(
        input_dims[1], K,
        common::errors::InvalidArgument(
            "input [%d] must be shape %d, %d, but received dims: %s", i, M, K,
            input_dims.to_str()));
  }
  PADDLE_ENFORCE_LE(N, 65535,
                    common::errors::InvalidArgument(
                        "The batch size (N) must be no larger than 65535."));
  PADDLE_ENFORCE_EQ(M % 128, 0,
                    common::errors::InvalidArgument(
                        "The upper dim (M) must be multiple of 128."));
  PADDLE_ENFORCE_EQ(K % 128, 0,
                    common::errors::InvalidArgument(
                        "The lower dim (K) must be multiple of 128."));
  return {N, M, K};
}

/**
 * Stack tensors in X, optionally transpose dim[-1] and dim[-2], and do
 * quantization on both dim[-1] and dim[-2].
 *
 * Inputs:
 *   X    : N tensors of [M, K], bfloat16
 *
 * Outputs:
 *   if Transpose:
 *     out  : [N * K, M], float8_e4m3fn
 *     scale: [N * K / 128, M / 128], float
 *   else:
 *     out  : [N * M, K], float8_e4m3fn
 *     scale: [N * M / 128, K / 128], float
 *
 * Requirements:
 *   1) N <= 65535
 *   2) M % 128 == 0
 *   3) K % 128 == 0
 */
template <bool Transpose>
std::vector<paddle::Tensor> fuse_stack_transpose_fp8_quant_fleet_custom(
    const std::vector<paddle::Tensor> &X, const bool &using_pow2_scaling,
    const bool &using_ue8m0_scale, const bool &output_scale_transpose) {
  int64_t N, M, K;
  std::tie(N, M, K) = FusedStackQuantCommonCheckFleetCustom(X);

  std::vector<int64_t> out_shape;
  std::vector<int64_t> scale_shape;

  if (Transpose) {
    out_shape = {N * K, M};
    if (using_ue8m0_scale) {
      if (output_scale_transpose) {
        scale_shape = {M / 128 / 4, N * K};
      } else {
        scale_shape = {N * K, M / 128 / 4};
      }
    } else {
      if (output_scale_transpose) {
        scale_shape = {M / 128, N * K / 128};
      } else {
        scale_shape = {N * K / 128, M / 128};
      }
    }

  } else {
    out_shape = {N * M, K};
    if (using_ue8m0_scale) {
      if (output_scale_transpose) {
        scale_shape = {K / 128 / 4, N * M};
      } else {
        scale_shape = {N * M, K / 128 / 4};
      }
    } else {
      if (output_scale_transpose) {
        scale_shape = {K / 128, N * M / 128};
      } else {
        scale_shape = {N * M / 128, K / 128};
      }
    }
  }

  // int64_t N = X.size();
  PD_CHECK(N > 0);
  for (int64_t i = 0; i < N; i++) {
    PD_CHECK(X[i].dtype() == paddle::DataType::BFLOAT16);
  }

  std::vector<int64_t> shape = X[0].shape();
  PD_CHECK(shape.size() == 2);
  // int64_t M = shape[0];
  // int64_t K = shape[1];

  for (int64_t i = 1; i < N; i++) {
    std::vector<int64_t> shape = X[i].shape();
    PD_CHECK(shape.size() == 2);
    PD_CHECK(shape[0] == M);
    PD_CHECK(shape[1] == K);
  }

  PADDLE_ENFORCE_LE(N, 65535,
                    common::errors::InvalidArgument(
                        "The batch size (N) must be no larger than 65535."));
  PADDLE_ENFORCE_EQ(M % 128, 0,
                    common::errors::InvalidArgument(
                        "The upper dim (M) must be multiple of 128."));
  PADDLE_ENFORCE_EQ(K % 128, 0,
                    common::errors::InvalidArgument(
                        "The lower dim (K) must be multiple of 128."));

  const auto &place = X[0].place();
  paddle::Tensor out =
      paddle::empty(out_shape, paddle::DataType::FLOAT8_E4M3FN, place);
  paddle::Tensor scale = paddle::empty(
      scale_shape,
      using_ue8m0_scale ? paddle::DataType::INT32 : paddle::DataType::FLOAT32,
      place);

  // Skip 0-size
  if (M == 0 || K == 0) {
    return {out, scale};
  }

  // Launch kernel
  int64_t grid_x = (M / 128) * (K / 128);
  PADDLE_ENFORCE_LE(
      grid_x, static_cast<int64_t>(std::numeric_limits<int>::max()),
      common::errors::InvalidArgument(
          "grid.x exceeds INT_MAX in fuse_stack_transpose_fp8_quant."));
  dim3 grid(static_cast<uint32_t>(grid_x), 1, N);
  dim3 block(32, 16);

  FastDivMod K_div_128(K / 128);
  {
    // NOLINTNEXTLINE(build/namespaces)
    using namespace phi;

#define LAUNCH_KERN(ScaleT, POW2, UE8M0, TRANS)                                \
  if (Transpose) {                                                             \
    FusedStackTransposeQuantGPUKernelFleetCustom<                              \
        phi::float8_e4m3fn, decltype(array), ScaleT, POW2, UE8M0, TRANS>       \
        <<<grid, block, 0, X[0].stream()>>>(                                   \
            array, out.data<phi::float8_e4m3fn>(),                             \
            reinterpret_cast<ScaleT *>(scale.data<ScaleT>()), M, K,            \
            K_div_128);                                                        \
  } else {                                                                     \
    FusedStackQuantGPUKernelFleetCustom<phi::float8_e4m3fn, decltype(array),   \
                                        ScaleT, POW2, UE8M0, TRANS>            \
        <<<grid, block, 0, X[0].stream()>>>(                                   \
            array, out.data<phi::float8_e4m3fn>(),                             \
            reinterpret_cast<ScaleT *>(scale.data<ScaleT>()), M, K,            \
            K_div_128);                                                        \
  }

    switch (funcs::CalcArraySize(N)) {
      SEGMENTED_ARRAY_KERNEL_HELPER({
        funcs::ConstPointerArray<phi::bfloat16, kArraySize> array;
        std::vector<const phi::bfloat16 *> ptrs(X.size());
        for (int i = 0; i < X.size(); ++i) {
          ptrs[i] = X[i].data<phi::bfloat16>();
        }
        const phi::bfloat16 **dev_ptr = nullptr;
        paddle::Tensor ptr_tensor;
        if constexpr (kArraySize ==
                      funcs::SegmentedArraySize::kVariableLength) {
          size_t nbytes = ptrs.size() * sizeof(const phi::bfloat16 *);
          ptr_tensor = paddle::empty({static_cast<int64_t>(nbytes)},
                                     paddle::DataType::UINT8, X[0].place());
          dev_ptr = reinterpret_cast<const phi::bfloat16 **>(ptr_tensor.data());
          auto err = cudaMemcpyAsync(dev_ptr, ptrs.data(), nbytes,
                                     cudaMemcpyHostToDevice, X[0].stream());
          PD_CHECK(err == cudaSuccess,
                   "cudaMemcpyAsync error: ", cudaGetErrorString(err));
        }
        array.Set(ptrs, dev_ptr);

        if (using_pow2_scaling) {
          if (using_ue8m0_scale) {
            if (output_scale_transpose) {
              LAUNCH_KERN(int, true, true, true);
            } else {
              LAUNCH_KERN(int, true, true, false);
            }
          } else {
            if (output_scale_transpose) {
              LAUNCH_KERN(float, true, false, true);
            } else {
              LAUNCH_KERN(float, true, false, false);
            }
          }
        } else {
          if (using_ue8m0_scale) {
            if (output_scale_transpose) {
              LAUNCH_KERN(int, false, true, true);
            } else {
              LAUNCH_KERN(int, false, true, false);
            }
          } else {
            if (output_scale_transpose) {
              LAUNCH_KERN(float, false, false, true);
            } else {
              LAUNCH_KERN(float, false, false, false);
            }
          }
        }
      });
    }

#undef LAUNCH_KERN
  }

  return {out, scale};
}

PD_BUILD_OP(fuse_stack_fp8_quant)
    .Inputs({paddle::Vec("X")})
    .Attrs({"using_pow2_scaling: bool", "using_ue8m0_scale: bool",
            "output_scale_transpose: bool"})
    .Outputs({"output", "scale"})
    .SetKernelFn(PD_KERNEL(fuse_stack_transpose_fp8_quant_fleet_custom<false>));

PD_BUILD_OP(fuse_stack_transpose_fp8_quant)
    .Inputs({paddle::Vec("X")})
    .Attrs({"using_pow2_scaling: bool", "using_ue8m0_scale: bool",
            "output_scale_transpose: bool"})
    .Outputs({"output", "scale"})
    .SetKernelFn(PD_KERNEL(fuse_stack_transpose_fp8_quant_fleet_custom<true>));
