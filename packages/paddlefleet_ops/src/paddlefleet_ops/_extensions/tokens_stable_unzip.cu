/*
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
*/
#include "paddle/common/array.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "utils.h"  // NOLINT

#define CUMSUM_BLOCK_SIZE 48
#define CUMSUM_INVALID_TAG -1

template <int MAX_NUM_EXPERTS>
struct __align__(16) expert_base_offset {
  int data[MAX_NUM_EXPERTS];
};

template <typename X_T,
          typename routemap_T,
          typename probs_T,
          bool has_scale,
          bool fill_x,
          int MAX_NUM_EXPERTS_C>
__global__ void tokens_unzip_stable_kernel(
    const X_T* __restrict__ X,
    const routemap_T* __restrict__ routemap_topk,
    const probs_T* __restrict__ probs_topk,
    const float* __restrict__ XScale,
    const expert_base_offset<MAX_NUM_EXPERTS_C> expert_base_offset,
    X_T* __restrict__ X_unzipped,
    int* __restrict__ zipped_expertwise_rowmap,
    probs_T* __restrict__ probs_unzipped,
    float* __restrict__ XScale_unzipped,
    int* global_expertwise_block_cumsum,
    const int total_zipped_tokens_num,
    const int token_length,
    const int scale_length,
    const int num_experts,
    const int topk) {
  const int block_row_base = blockIdx.x * CUMSUM_BLOCK_SIZE;
  int cumsum_offset[MAX_NUM_EXPERTS_C];
  int local_expert_offsets[MAX_NUM_EXPERTS_C];
  int local_cumsum[MAX_NUM_EXPERTS_C];
#pragma unroll
  for (int i = 0; i < num_experts; i++) {
    cumsum_offset[i] = (blockIdx.x == 0) ? 0 : CUMSUM_INVALID_TAG;
    local_expert_offsets[i] = expert_base_offset.data[i];
    local_cumsum[i] = 0;
  }
  const int base_row_idx = blockIdx.x * CUMSUM_BLOCK_SIZE;
  __shared__ int shared_expert_rowmap[CUMSUM_BLOCK_SIZE][MAX_NUM_EXPERTS_C];
  __shared__ probs_T
      shared_expert_probmap[CUMSUM_BLOCK_SIZE][MAX_NUM_EXPERTS_C];

  if (threadIdx.x == 0) [[unlikely]] {  // NOLINT
    int local_expert_rowmap[CUMSUM_BLOCK_SIZE][MAX_NUM_EXPERTS_C];
    probs_T local_expert_probs[CUMSUM_BLOCK_SIZE][MAX_NUM_EXPERTS_C];
#pragma unroll
    for (int i = 0; i < CUMSUM_BLOCK_SIZE; i++) {
#pragma unroll
      for (int j = 0; j < num_experts; j++) {
        local_expert_rowmap[i][j] = -1;
        local_expert_probs[i][j] = (probs_T)0;
      }
    }

    for (int row = block_row_base; row < block_row_base + CUMSUM_BLOCK_SIZE;
         row++) {
      if (row >= total_zipped_tokens_num) break;
      const int internal_row = row - block_row_base;
#pragma unroll
      for (int k = 0; k < topk; k++) {
        const int expert = routemap_topk[row * topk + k];
        if (expert == -1) continue;
        local_expert_rowmap[internal_row][expert] =
            local_cumsum[expert] + local_expert_offsets[expert];
        local_expert_probs[internal_row][expert] = probs_topk[row * topk + k];
        local_cumsum[expert] += 1;
      }
    }

#pragma unroll
    for (int i = 0; i < num_experts; i++) {
      if (blockIdx.x != 0) [[likely]] {  // NOLINT
        while (cumsum_offset[i] == CUMSUM_INVALID_TAG) [[likely]] {
          cumsum_offset[i] = atomicExch(
              &global_expertwise_block_cumsum[blockIdx.x * num_experts + i],
              CUMSUM_INVALID_TAG);
        }
      }
      const int proposed_offset = cumsum_offset[i] + local_cumsum[i];
      global_expertwise_block_cumsum[(blockIdx.x + 1) * num_experts + i] =
          proposed_offset;
    }

#pragma unroll
    for (int i = 0; i < CUMSUM_BLOCK_SIZE; i++) {
#pragma unroll
      for (int j = 0; j < num_experts; j++) {
        const int proposed_row =
            (local_expert_rowmap[i][j] == -1)
                ? -1
                : (local_expert_rowmap[i][j] + cumsum_offset[j]);
        shared_expert_rowmap[i][j] = proposed_row;
        shared_expert_probmap[i][j] = local_expert_probs[i][j];
      }
    }
  }
  __syncthreads();

  for (int row = block_row_base; row < block_row_base + CUMSUM_BLOCK_SIZE;
       row++) {
    if (row >= total_zipped_tokens_num) return;
    const int internal_row = row - block_row_base;
#pragma unroll
    for (int expert = 0; expert < num_experts; expert++) {
      const int unzipped_row_idx = shared_expert_rowmap[internal_row][expert];
      if (threadIdx.x == 0) {
        zipped_expertwise_rowmap[row * num_experts + expert] = unzipped_row_idx;
      }
      if (unzipped_row_idx == -1) continue;

      if (threadIdx.x == 0) {
        probs_unzipped[unzipped_row_idx] =
            shared_expert_probmap[internal_row][expert];
      }
      if (fill_x) {
        if constexpr (has_scale) {
          vectorized_memcpy(&XScale[(int64_t)row * (int64_t)scale_length],
                            &XScale_unzipped[(int64_t)unzipped_row_idx *
                                             (int64_t)scale_length],
                            scale_length);
        }
        vectorized_memcpy(
            &X[(int64_t)row * (int64_t)token_length],
            &X_unzipped[(int64_t)unzipped_row_idx * (int64_t)token_length],
            token_length);
      }
    }
  }
}
// ---------------------------- Dispatch ---------------------------------
template <int MAX_NUM_EXPERTS_C>
void dispatch_tokens_unzip_stable(
    const paddle::Tensor& X,
    const paddle::Tensor& expert_routemap_topk,
    const paddle::Tensor& expert_prob_topk,
    const paddle::optional<paddle::Tensor>& XScale,
    const expert_base_offset<MAX_NUM_EXPERTS_C>& expert_offsets,
    paddle::Tensor& X_unzipped,                      // NOLINT
    paddle::Tensor& zipped_expertwise_rowmap,        // NOLINT
    paddle::Tensor& token_prob_unzipped,             // NOLINT
    paddle::Tensor& XScale_unzipped,                 // NOLINT
    paddle::Tensor& global_expertwise_block_cumsum,  // NOLINT
    const int total_zipped_tokens_num,
    const int token_length,
    const int topk,  // deprecated
    const int num_experts,
    const int scale_length,
    const bool fill_x) {
  dim3 grid, block;
  grid.x =
      (total_zipped_tokens_num + CUMSUM_BLOCK_SIZE - 1) / CUMSUM_BLOCK_SIZE;
  block.x = 256;

  if (grid.x <= 0) return;

#define DTYPE_CASE(dtype, type) dtype == paddle::DataType::type
#define GET_DATA(tensor, type) tensor.data<type>()

#define DISPATCH_CASE_IMPL(TOKEN_T, PROB_T, INT_T, HAS_SCALE, FILL_X) \
  do {                                                                \
    auto kernel = tokens_unzip_stable_kernel<TOKEN_T,                 \
                                             INT_T,                   \
                                             PROB_T,                  \
                                             HAS_SCALE,               \
                                             FILL_X,                  \
                                             MAX_NUM_EXPERTS_C>;      \
    kernel<<<grid, block, 0, X.stream()>>>(                           \
        GET_DATA(X, TOKEN_T),                                         \
        GET_DATA(expert_routemap_topk, INT_T),                        \
        GET_DATA(expert_prob_topk, PROB_T),                           \
        XScale ? XScale->data<float>() : nullptr,                     \
        expert_offsets,                                               \
        GET_DATA(X_unzipped, TOKEN_T),                                \
        GET_DATA(zipped_expertwise_rowmap, INT_T),                    \
        GET_DATA(token_prob_unzipped, PROB_T),                        \
        XScale_unzipped.data<float>(),                                \
        global_expertwise_block_cumsum.data<int>(),                   \
        total_zipped_tokens_num,                                      \
        token_length,                                                 \
        scale_length,                                                 \
        num_experts,                                                  \
        topk);                                                        \
  } while (0)

#define DISPATCH_CASE(TOKEN_T, PROB_T, INT_T, HAS_SCALE)            \
  do {                                                              \
    if (fill_x) {                                                   \
      DISPATCH_CASE_IMPL(TOKEN_T, PROB_T, INT_T, HAS_SCALE, true);  \
    } else {                                                        \
      DISPATCH_CASE_IMPL(TOKEN_T, PROB_T, INT_T, HAS_SCALE, false); \
    }                                                               \
  } while (0)

// 可扩展：处理特定的topk和num_experts组合,可根据之后需求进行扩展
#define HANDLE_EXPERT_CASE(TOKEN_T, PROB_T, INT_T, HAS_SCALE) \
  DISPATCH_CASE(TOKEN_T, PROB_T, INT_T, HAS_SCALE);

#define HANDLE_TOKEN_TYPE(PROB_T, INT_T)                           \
  do {                                                             \
    if (DTYPE_CASE(X.dtype(), BFLOAT16)) {                         \
      HANDLE_EXPERT_CASE(phi::bfloat16, PROB_T, INT_T, false);     \
    } else if (DTYPE_CASE(X.dtype(), FLOAT8_E4M3FN)) {             \
      HANDLE_EXPERT_CASE(phi::float8_e4m3fn, PROB_T, INT_T, true); \
    }                                                              \
  } while (0)

#define HANDLE_PROB_TYPE(INT_T)                                 \
  do {                                                          \
    if (DTYPE_CASE(expert_prob_topk.dtype(), BFLOAT16)) {       \
      HANDLE_TOKEN_TYPE(phi::bfloat16, INT_T);                  \
    } else if (DTYPE_CASE(expert_prob_topk.dtype(), FLOAT32)) { \
      HANDLE_TOKEN_TYPE(float, INT_T);                          \
    }                                                           \
  } while (0)

  if (DTYPE_CASE(zipped_expertwise_rowmap.dtype(), INT32)) {
    HANDLE_PROB_TYPE(int);
  }

#undef DTYPE_CASE
#undef GET_DATA
#undef DISPATCH_CASE
#undef HANDLE_EXPERT_CASE
#undef HANDLE_TOKEN_TYPE
#undef HANDLE_PROB_TYPE
}

std::vector<paddle::Tensor> tokens_unzip_stable(
    const paddle::Tensor& X,
    const paddle::optional<paddle::Tensor>& XScale,
    const paddle::Tensor& expert_routemap_topk,
    const paddle::Tensor& expert_prob_topk,
    const int& topk,
    const int& num_experts,
    const std::vector<int>& tokens_per_expert,
    const int padding_multiplex,
    const bool fill_x) {
  PD_CHECK(X.dtype() == paddle::DataType::BFLOAT16 ||
           X.dtype() == paddle::DataType::FLOAT8_E4M3FN);
  PD_CHECK(expert_routemap_topk.dtype() == paddle::DataType::INT32);
  PD_CHECK(expert_prob_topk.dtype() == paddle::DataType::BFLOAT16 ||
           expert_prob_topk.dtype() == paddle::DataType::FLOAT32);
  if (XScale) {
    PD_CHECK(XScale->dtype() == paddle::DataType::FLOAT32);
  }
  // TODO(large-tensor): downstream functors may still use int; guard until
  // upgraded.
  int64_t rows = X.shape()[0];

  // TODO(large-tensor): downstream functors may still use int; guard until
  // upgraded.
  int64_t cols = X.shape()[1];

  const int quanted_cols = (XScale) ? XScale->shape()[1] : 0;
  /*
  const int max_tokens_per_expert =
      ((max_tokens_per_expert_in + 127) / 128) * 128;
  const int output_rows = num_experts * max_tokens_per_expert;
  */
  //------------------------ 输出缓冲区分配  ------------------------
  paddle::Tensor X_unzipped, XScale_unzipped, zipped_expertwise_rowmap,
      token_prob_unzipped;

  PD_SWITCH_NUM_EXPERTS(
      num_experts, ([&] {
        expert_base_offset<MAX_NUM_EXPERTS_C> expert_offset;
        int tokens_cumulated = 0;
        for (int i = 0; i < MAX_NUM_EXPERTS_C; i++) {
          if (i < num_experts) {
            expert_offset.data[i] = tokens_cumulated;
            tokens_cumulated +=
                ((tokens_per_expert[i] + padding_multiplex - 1) /
                 padding_multiplex) *
                padding_multiplex;
          } else {
            expert_offset.data[i] = 0;
          }
        }

        const int output_rows = tokens_cumulated;
        // TODO(large-tensor): downstream functors may still use int; guard
        // until upgraded.
        int64_t topk_calculated = expert_routemap_topk.shape()[1];

        if (XScale && fill_x) {
          XScale_unzipped = paddle::empty(
              {output_rows, quanted_cols}, XScale->dtype(), XScale->place());
        } else {
          XScale_unzipped =
              paddle::empty({0}, paddle::DataType::FLOAT32, X.place());
        }

        if (fill_x) {
          X_unzipped = paddle::empty({output_rows, cols}, X.dtype(), X.place());
        } else {
          X_unzipped = paddle::empty({0, cols}, X.dtype(), X.place());
        }
        zipped_expertwise_rowmap = paddle::empty(
            {rows, num_experts}, paddle::DataType::INT32, X.place());
        token_prob_unzipped = paddle::empty(
            {output_rows}, expert_prob_topk.dtype(), expert_prob_topk.place());

        if (fill_x) {
          if (X.dtype() == paddle::DataType::BFLOAT16) {
            auto X_unzipped_ptr =
                reinterpret_cast<void*>(X_unzipped.data<phi::bfloat16>());
            cudaMemsetAsync(X_unzipped_ptr,
                            0,
                            sizeof(phi::bfloat16) * output_rows * cols,
                            X.stream());
          } else if (X.dtype() == paddle::DataType::FLOAT8_E4M3FN) {
            auto X_unzipped_ptr =
                reinterpret_cast<void*>(X_unzipped.data<phi::float8_e4m3fn>());
            cudaMemsetAsync(X_unzipped_ptr,
                            0,
                            sizeof(phi::float8_e4m3fn) * output_rows * cols,
                            X.stream());
          }
          if (XScale) {
            auto XScale_unzipped_ptr =
                reinterpret_cast<void*>(XScale_unzipped.data<float>());
            cudaMemsetAsync(XScale_unzipped_ptr,
                            0,
                            sizeof(float) * output_rows * quanted_cols,
                            XScale_unzipped.stream());
          }
        }
        if (expert_prob_topk.dtype() == paddle::DataType::BFLOAT16) {
          auto token_prob_unzipped_ptr = reinterpret_cast<void*>(
              token_prob_unzipped.data<phi::bfloat16>());
          cudaMemsetAsync(token_prob_unzipped_ptr,
                          0,
                          sizeof(phi::bfloat16) * output_rows,
                          token_prob_unzipped.stream());
        } else if (expert_prob_topk.dtype() == paddle::DataType::FLOAT32) {
          auto token_prob_unzipped_ptr =
              reinterpret_cast<void*>(token_prob_unzipped.data<float>());
          cudaMemsetAsync(token_prob_unzipped_ptr,
                          0,
                          sizeof(float) * output_rows,
                          token_prob_unzipped.stream());
        }

        const int cumsum_blocknum =
            (rows + CUMSUM_BLOCK_SIZE - 1) / CUMSUM_BLOCK_SIZE;
        auto global_expertwise_block_cumsum =
            paddle::empty({cumsum_blocknum + 1, num_experts},
                          paddle::DataType::INT32,
                          X.place());
        auto global_expertwise_block_cumsum_ptr =
            reinterpret_cast<void*>(global_expertwise_block_cumsum.data<int>());

        cudaMemsetAsync(global_expertwise_block_cumsum_ptr,
                        CUMSUM_INVALID_TAG,
                        sizeof(int) * (cumsum_blocknum + 1) * num_experts,
                        global_expertwise_block_cumsum.stream());
        if (rows != 0) {
          dispatch_tokens_unzip_stable<MAX_NUM_EXPERTS_C>(
              X,
              expert_routemap_topk,
              expert_prob_topk,
              XScale,
              expert_offset,
              X_unzipped,
              zipped_expertwise_rowmap,
              token_prob_unzipped,
              XScale_unzipped,
              global_expertwise_block_cumsum,
              rows,
              cols,
              topk_calculated,
              num_experts,
              quanted_cols,
              fill_x);
        }
      }));
  return {X_unzipped,
          zipped_expertwise_rowmap,
          token_prob_unzipped,
          XScale_unzipped};
}

PD_BUILD_OP(tokens_unzip_stable)
    .Inputs({"X",
             paddle::Optional("Xscale"),
             "expert_routemap_topk",
             "expert_prob_topk"})
    .Outputs({"X_unzipped",
              "zipped_expertwise_rowmap",
              "token_prob_unzipped",
              paddle::Optional("XScale_unzipped")})
    .Attrs({"topk: int",
            "num_experts: int",
            "tokens_per_expert: std::vector<int>",
            "padding_multiplex: int",
            "fill_output: bool"})
    .SetKernelFn(PD_KERNEL(tokens_unzip_stable));

#undef CUMSUM_BLOCK_SIZE
