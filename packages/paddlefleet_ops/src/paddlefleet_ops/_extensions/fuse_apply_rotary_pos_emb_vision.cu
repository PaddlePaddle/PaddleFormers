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

// Optimized CUDA custom op: apply_rotary_pos_emb_vision  (forward + backward)
//
// Python reference:
//   def rotate_half(x):
//       x1 = x[..., : x.shape[-1] // 2]
//       x2 = x[..., x.shape[-1] // 2 :]
//       return paddle.concat([-x2, x1], axis=-1)
//
//   def apply_rotary_pos_emb_vision(tensor, freqs):
//       orig_dtype = tensor.dtype
//       tensor = tensor.cast("float32")
//       cos = freqs.cos().unsqueeze(1).tile([1, 1,
//       2]).unsqueeze(0).cast("float32") sin =
//       freqs.sin().unsqueeze(1).tile([1, 1, 2]).unsqueeze(0).cast("float32")
//       output = tensor * cos + rotate_half(tensor) * sin
//       return output.cast(orig_dtype)
//
// Key optimizations vs V1:
//   1. Single kernel: compute cos/sin from freqs inline — no separate kernel,
//      no intermediate global-memory cos/sin table.
//   2. Grid = (seq_len, batch): one block per (s, b) position.
//   3. 2D block = (WARP=32, warps_per_block): threadIdx.y iterates over heads.
//      cos/sin loaded into shared memory ONCE and reused by all heads.
//   4. sincosf(): compute both cos and sin simultaneously (single HW
//   instruction).
//   5. Forward/backward unified as template (IS_FWD = true/false flips sin
//   sign).
//   6. __ldg for freqs (read-only cache, broadcast-friendly).
//
// Input shapes (supports both 3D and 4D):
//   tensor : [seq, heads, dim] or [batch, seq, heads, dim]  (float16 / bfloat16
//   / float32) freqs  : [seq, dim//2]              (any dtype, cast to float32
//   internally)
// Output:
//   Out    : same shape + dtype as tensor

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include "paddle/extension.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"

// ---------------------------------------------------------------------------
// Type converters  (element <-> float32)
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ float elem_to_float(T v) {
  return static_cast<float>(v);
}
template <>
__device__ __forceinline__ float elem_to_float<phi::dtype::float16>(
    phi::dtype::float16 v) {
  return __half2float(*reinterpret_cast<const __half*>(&v));
}
template <>
__device__ __forceinline__ float elem_to_float<phi::dtype::bfloat16>(
    phi::dtype::bfloat16 v) {
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&v));
}

template <typename T>
__device__ __forceinline__ T float_to_elem(float v) {
  return static_cast<T>(v);
}
template <>
__device__ __forceinline__ phi::dtype::float16
float_to_elem<phi::dtype::float16>(float v) {
  __half h = __float2half(v);
  phi::dtype::float16 r;
  *reinterpret_cast<__half*>(&r) = h;
  return r;
}
template <>
__device__ __forceinline__ phi::dtype::bfloat16
float_to_elem<phi::dtype::bfloat16>(float v) {
  __nv_bfloat16 b = __float2bfloat16(v);
  phi::dtype::bfloat16 r;
  *reinterpret_cast<__nv_bfloat16*>(&r) = b;
  return r;
}

// ---------------------------------------------------------------------------
// Unified forward / backward kernel
//
// IS_FWD = true  → forward:  out = x*cos + rotate_half(x)*sin
// IS_FWD = false → backward: out = x*cos - rotate_half(x)*sin
//                            (rotation inverse = negate sin)
//
// Grid  : (seq_len, batch_size)       — one block per (s, b) position
// Block : (WARP_SIZE=32, WPB)         — WPB warps cover multiple heads
// Shmem : cos_s[dim] + sin_s[dim]    — computed once, shared across heads
// ---------------------------------------------------------------------------

template <typename T, bool IS_FWD>
__global__ void RopeVisionKernel(
    const T* __restrict__ in,         // [batch, seq, heads, dim]
    const float* __restrict__ freqs,  // [seq, half_dim]  float32
    T* __restrict__ out,              // [batch, seq, heads, dim]
    int heads,
    int dim,
    int half_dim) {
  int s_id = blockIdx.x;  // seq position
  int b_id = blockIdx.y;  // batch position

  extern __shared__ float smem[];
  float* cos_s = smem;        // [dim]
  float* sin_s = smem + dim;  // [dim]

  // ----- Phase 1: populate cos_s / sin_s from freqs -----
  // All threads in the block collaborate; each covers a strided range of d.
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int blk_size = blockDim.x * blockDim.y;  // 32 * WPB
  int tid = ty * blockDim.x + tx;

  const float* freqs_row = freqs + s_id * half_dim;  // [half_dim]

  for (int d = tid; d < dim; d += blk_size) {
    // Tile: freqs covers [0, half_dim); dim = 2*half_dim.
    float f = __ldg(freqs_row + (d % half_dim));
    sincosf(f, &sin_s[d], &cos_s[d]);
  }
  __syncthreads();

  // ----- Phase 2: apply rotation for each head -----
  // ty loops over heads (step = WPB), tx loops over dim (step = 32).
  // Use int64_t so the index arithmetic is safe for tensors whose total
  // element count exceeds 2^31 (e.g. batch*seq*heads*dim > 2,147,483,647).
  int64_t base_sb = (static_cast<int64_t>(b_id) * gridDim.x + s_id) *
                    (static_cast<int64_t>(heads) * dim);

  for (int h_id = ty; h_id < heads; h_id += blockDim.y) {
    int64_t base_h = base_sb + (int64_t)h_id * dim;

    for (int d = tx; d < dim; d += blockDim.x) {
      float x = elem_to_float(in[base_h + d]);
      // rotate_half partner: d < half → partner = d+half; else partner = d-half
      int partner = (d < half_dim) ? (d + half_dim) : (d - half_dim);
      float x_rot = elem_to_float(in[base_h + partner]);
      // rotate_half sign: first half negates, second half keeps
      float rot = (d < half_dim) ? -x_rot : x_rot;

      float c = cos_s[d];
      float s = sin_s[d];
      // forward:  x*cos + rot*sin
      // backward: x*cos - rot*sin  (IS_FWD=false → subtract)
      float val = IS_FWD ? (x * c + rot * s) : (x * c - rot * s);

      out[base_h + d] = float_to_elem<T>(val);
    }
  }
}

// ---------------------------------------------------------------------------
// Launcher: choose WPB (warps-per-block) and launch the kernel
// ---------------------------------------------------------------------------

// Max shared memory needed = 2 * dim * sizeof(float)
// Max block size = 32 * 8 = 256 threads (safe for most GPUs).
constexpr int WARP_SIZE = 32;
constexpr int MAX_WPB = 8;

template <typename T, bool IS_FWD>
static void LaunchRopeVision(const paddle::Tensor& in,
                             const paddle::Tensor& freqs,
                             paddle::Tensor& out,
                             int batch,
                             int seq_len,
                             int heads,
                             int dim,
                             int half_dim,
                             cudaStream_t stream) {
  // Choose warps-per-block: enough to keep SMs busy; cap at MAX_WPB.
  // Mirror Paddle: use 4 if heads < 16, else 8.
  int wpb = (heads < 16) ? 4 : MAX_WPB;

  dim3 grid(seq_len, batch);
  dim3 block(WARP_SIZE, wpb);
  size_t smem = 2 * dim * sizeof(float);  // cos_s + sin_s

  RopeVisionKernel<T, IS_FWD><<<grid, block, smem, stream>>>(
      in.data<T>(), freqs.data<float>(), out.data<T>(), heads, dim, half_dim);
}

// ---------------------------------------------------------------------------
// Custom op: forward
// ---------------------------------------------------------------------------

std::vector<paddle::Tensor> ApplyRopevisionForward(
    const paddle::Tensor& tensor, const paddle::Tensor& freqs) {
  auto shape = tensor.shape();
  int ndim = static_cast<int>(shape.size());

  PD_CHECK(
      ndim == 3 || ndim == 4,
      "fused_apply_rotary_pos_emb_vision: tensor must be 3D [seq, heads, dim] "
      "or 4D [batch, seq, heads, dim], got ",
      ndim,
      "D");

  // 0-size: return an empty tensor of the same shape/dtype immediately.
  if (tensor.numel() == 0) {
    return {paddle::empty(shape, tensor.dtype(), tensor.place())};
  }

  // Normalise to 4D: [batch, seq, heads, dim]
  int batch, seq_len, heads, dim;
  if (ndim == 3) {
    batch = 1;
    seq_len = static_cast<int>(shape[0]);
    heads = static_cast<int>(shape[1]);
    dim = static_cast<int>(shape[2]);
  } else {
    batch = static_cast<int>(shape[0]);
    seq_len = static_cast<int>(shape[1]);
    heads = static_cast<int>(shape[2]);
    dim = static_cast<int>(shape[3]);
  }

  PD_CHECK(dim % 2 == 0 && dim > 0,
           "fused_apply_rotary_pos_emb_vision: tensor dim must be positive "
           "even number, got ",
           dim);

  int half = dim / 2;

  // Cast freqs to float32 if needed (kernel reads float*)
  paddle::Tensor freqs_f32 = (freqs.dtype() == paddle::DataType::FLOAT32)
                                 ? freqs
                                 : freqs.cast(paddle::DataType::FLOAT32);

  // Output shape matches input shape
  auto stream = tensor.stream();
  auto out = paddle::empty(shape, tensor.dtype(), tensor.place());

  // The kernel always works on contiguous [batch, seq, heads, dim] layout.
  // For 3D input [seq, heads, dim], the memory layout is identical to
  // [1, seq, heads, dim], so we just pass batch=1 to the kernel.
  switch (tensor.dtype()) {
    case paddle::DataType::FLOAT32:
      LaunchRopeVision<float, true>(
          tensor, freqs_f32, out, batch, seq_len, heads, dim, half, stream);
      break;
    case paddle::DataType::FLOAT16:
      LaunchRopeVision<phi::dtype::float16, true>(
          tensor, freqs_f32, out, batch, seq_len, heads, dim, half, stream);
      break;
    case paddle::DataType::BFLOAT16:
      LaunchRopeVision<phi::dtype::bfloat16, true>(
          tensor, freqs_f32, out, batch, seq_len, heads, dim, half, stream);
      break;
    default:
      PD_THROW("fused_apply_rotary_pos_emb_vision: unsupported dtype");
  }
  return {out};
}

// ---------------------------------------------------------------------------
// Custom op: backward
// ---------------------------------------------------------------------------

std::vector<paddle::Tensor> ApplyRopevisionBackward(
    const paddle::Tensor& d_out, const paddle::Tensor& freqs) {
  auto shape = d_out.shape();
  int ndim = static_cast<int>(shape.size());

  PD_CHECK(ndim == 3 || ndim == 4,
           "fused_apply_rotary_pos_emb_vision backward: tensor must be 3D or "
           "4D, got ",
           ndim,
           "D");

  // 0-size: return an empty gradient tensor immediately.
  if (d_out.numel() == 0) {
    return {paddle::empty(shape, d_out.dtype(), d_out.place())};
  }

  int batch, seq_len, heads, dim;
  if (ndim == 3) {
    batch = 1;
    seq_len = static_cast<int>(shape[0]);
    heads = static_cast<int>(shape[1]);
    dim = static_cast<int>(shape[2]);
  } else {
    batch = static_cast<int>(shape[0]);
    seq_len = static_cast<int>(shape[1]);
    heads = static_cast<int>(shape[2]);
    dim = static_cast<int>(shape[3]);
  }

  PD_CHECK(dim % 2 == 0 && dim > 0,
           "fused_apply_rotary_pos_emb_vision: tensor dim must be positive "
           "even number, got ",
           dim);

  int half = dim / 2;

  // Cast freqs to float32 if needed
  paddle::Tensor freqs_f32 = (freqs.dtype() == paddle::DataType::FLOAT32)
                                 ? freqs
                                 : freqs.cast(paddle::DataType::FLOAT32);

  auto stream = d_out.stream();
  // Output shape matches input shape
  auto d_tensor = paddle::empty(shape, d_out.dtype(), d_out.place());

  // Same as forward: kernel works on contiguous data with batch=1 for 3D
  switch (d_out.dtype()) {
    case paddle::DataType::FLOAT32:
      LaunchRopeVision<float, false>(
          d_out, freqs_f32, d_tensor, batch, seq_len, heads, dim, half, stream);
      break;
    case paddle::DataType::FLOAT16:
      LaunchRopeVision<phi::dtype::float16, false>(
          d_out, freqs_f32, d_tensor, batch, seq_len, heads, dim, half, stream);
      break;
    case paddle::DataType::BFLOAT16:
      LaunchRopeVision<phi::dtype::bfloat16, false>(
          d_out, freqs_f32, d_tensor, batch, seq_len, heads, dim, half, stream);
      break;
    default:
      PD_THROW("fused_apply_rotary_pos_emb_vision backward: unsupported dtype");
  }
  return {d_tensor};
}

// ---------------------------------------------------------------------------
// Shape / dtype inference
// ---------------------------------------------------------------------------

std::vector<std::vector<int64_t>> FwdInferShape(
    std::vector<int64_t> tensor_shape, std::vector<int64_t> freqs_shape) {
  return {tensor_shape};
}

std::vector<paddle::DataType> FwdInferDtype(paddle::DataType tensor_dtype,
                                            paddle::DataType freqs_dtype) {
  return {tensor_dtype};
}

std::vector<std::vector<int64_t>> BwdInferShape(
    std::vector<int64_t> d_out_shape, std::vector<int64_t> freqs_shape) {
  return {d_out_shape};
}

std::vector<paddle::DataType> BwdInferDtype(paddle::DataType d_out_dtype,
                                            paddle::DataType freqs_dtype) {
  return {d_out_dtype};
}

// ---------------------------------------------------------------------------
// Op registration
// ---------------------------------------------------------------------------

PD_BUILD_OP(fused_apply_rotary_pos_emb_vision)
    .Inputs({"Tensor", "Freqs"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(ApplyRopevisionForward))
    .SetInferShapeFn(PD_INFER_SHAPE(FwdInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FwdInferDtype));

PD_BUILD_GRAD_OP(fused_apply_rotary_pos_emb_vision)
    .Inputs({paddle::Grad("Out"), "Freqs"})
    .Outputs({paddle::Grad("Tensor")})
    .SetKernelFn(PD_KERNEL(ApplyRopevisionBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(BwdInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(BwdInferDtype));
