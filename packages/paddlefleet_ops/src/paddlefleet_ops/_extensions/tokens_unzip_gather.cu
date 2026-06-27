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

#include "paddle/common/array.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "utils.h"  // NOLINT

template <typename T, typename ScaleT, bool has_scale>
__global__ void tokens_unzip_gather_kernel(
    const T* __restrict__ x,
    const ScaleT* __restrict__ x_scale,
    const int* __restrict__ zipped_expertwise_rowmap,
    T* __restrict__ x_unzipped,
    ScaleT* __restrict__ x_scale_unzipped,
    int64_t* __restrict__ index_unzipped,
    int64_t unzipped_rows,
    int64_t zipped_rows,
    int token_length,
    int scale_length,
    int num_experts,
    int expert_id,
    int64_t offset) {
  for (int64_t row = blockIdx.x; row < zipped_rows; row += gridDim.x) {
    int64_t unzipped_row_idx =
        zipped_expertwise_rowmap[row * num_experts + expert_id];
    if (unzipped_row_idx < 0) continue;

    unzipped_row_idx -= offset;
    // Guard against out-of-bounds access on index_unzipped and *_unzipped
    // buffers.
    if (unzipped_row_idx < 0 || unzipped_row_idx >= unzipped_rows) continue;
    index_unzipped[unzipped_row_idx] = row;
    if constexpr (has_scale) {
      try_vectorized_memcpy(x_scale + row * scale_length,
                            x_scale_unzipped + unzipped_row_idx * scale_length,
                            scale_length);
    }
    vectorized_memcpy(x + row * token_length,
                      x_unzipped + unzipped_row_idx * token_length,
                      token_length);
  }
}

std::vector<paddle::Tensor> tokens_unzip_gather(
    const paddle::Tensor& x,
    const paddle::optional<paddle::Tensor>& x_scale,
    const paddle::Tensor& zipped_expertwise_rowmap,
    const int expert_id,
    const std::vector<int64_t>& tokens_per_expert,
    const int padding_multiplex) {
  int num_experts = tokens_per_expert.size();
  PD_CHECK(expert_id >= 0 && expert_id < num_experts);
  std::vector<int64_t> cumsum_tokens(num_experts + 1);
  cumsum_tokens[0] = 0;
  for (int i = 0; i < num_experts; ++i) {
    auto padded = (tokens_per_expert[i] + padding_multiplex - 1) /
                  padding_multiplex * padding_multiplex;
    cumsum_tokens[i + 1] = cumsum_tokens[i] + padded;
  }

  int64_t padded_num_tokens =
      cumsum_tokens[expert_id + 1] - cumsum_tokens[expert_id];
  int64_t offset = cumsum_tokens[expert_id];

  auto dtype = x.dtype();
  auto place = x.place();
  auto stream = x.stream();
  auto x_shape = x.shape();
  PD_CHECK(x_shape.size() == 2);
  int64_t zipped_rows = x_shape[0];
  int hidden_size = x_shape[1];

  bool has_scale = (x_scale.get_ptr() != nullptr);
  int quanted_hidden_size = 0;
  if (has_scale) {
    auto x_scale_shape = x_scale.get().shape();
    PD_CHECK(x_scale_shape.size() == 2);
    PD_CHECK(x_scale_shape[0] == x_shape[0]);
    quanted_hidden_size = x_scale_shape[1];
  } else {
    PD_CHECK(hidden_size % 128 == 0);
    quanted_hidden_size = hidden_size / 128;
  }

  auto x_unzipped =
      paddle::zeros({padded_num_tokens, hidden_size}, dtype, place);
  paddle::Tensor x_scale_unzipped;
  if (has_scale) {
    x_scale_unzipped = paddle::zeros(
        {padded_num_tokens, quanted_hidden_size}, x_scale.get().dtype(), place);
  } else {
    x_scale_unzipped = paddle::empty(
        {0, quanted_hidden_size}, paddle::DataType::FLOAT32, place);
  }
  auto index_unzipped = paddle::empty(
      {tokens_per_expert[expert_id]}, paddle::DataType::INT64, place);

  int block = 1024;
  int grid = LimitGridDim(zipped_rows);

#define LAUNCH_TOKENS_UNZIP_GATHER_KERNEL_IMPL(                             \
    __cpp_dtype, __scale_dtype, __has_scale)                                \
  do {                                                                      \
    tokens_unzip_gather_kernel<__cpp_dtype, __scale_dtype, __has_scale>     \
        <<<grid, block, 0, stream>>>(                                       \
            x.data<__cpp_dtype>(),                                          \
            __has_scale ? x_scale.get().data<__scale_dtype>() : nullptr,    \
            zipped_expertwise_rowmap.data<int>(),                           \
            x_unzipped.data<__cpp_dtype>(),                                 \
            __has_scale ? x_scale_unzipped.data<__scale_dtype>() : nullptr, \
            index_unzipped.data<int64_t>(),                                 \
            tokens_per_expert[expert_id],                                   \
            zipped_rows,                                                    \
            hidden_size,                                                    \
            quanted_hidden_size,                                            \
            num_experts,                                                    \
            expert_id,                                                      \
            offset);                                                        \
  } while (0)

#define LAUNCH_TOKENS_UNZIP_GATHER_KERNEL(__cpp_dtype)                    \
  do {                                                                    \
    if (has_scale) {                                                      \
      if (x_scale.get().dtype() == paddle::DataType::FLOAT32) {           \
        LAUNCH_TOKENS_UNZIP_GATHER_KERNEL_IMPL(__cpp_dtype, float, true); \
      } else if (x_scale.get().dtype() == paddle::DataType::INT32) {      \
        LAUNCH_TOKENS_UNZIP_GATHER_KERNEL_IMPL(__cpp_dtype, int, true);   \
      } else {                                                            \
        PD_CHECK(false, "Unsupported scale dtype");                       \
      }                                                                   \
    } else {                                                              \
      LAUNCH_TOKENS_UNZIP_GATHER_KERNEL_IMPL(__cpp_dtype, float, false);  \
    }                                                                     \
  } while (0)

  // Skip kernel when expert has no tokens: unzipped_rows=0 causes the kernel to
  // always hit `unzipped_row_idx >= unzipped_rows` and continue, so avoid the
  // wasteful launch.
  if (grid > 0 && tokens_per_expert[expert_id] > 0) {
    if (has_scale) {
      LAUNCH_TOKENS_UNZIP_GATHER_KERNEL(phi::float8_e4m3fn);
    } else {
      LAUNCH_TOKENS_UNZIP_GATHER_KERNEL(phi::bfloat16);
    }
  }
  return {x_unzipped, x_scale_unzipped, index_unzipped};
}

PD_BUILD_OP(tokens_unzip_gather)
    .Inputs({"x", paddle::Optional("x_scale"), "zipped_expertwise_rowmap"})
    .Outputs({"x_unzipped",
              paddle::Optional("x_scale_unzipped"),
              "idx_unzipped"})
    .Attrs({"expert_id: int",
            "tokens_per_expert: std::vector<int64_t>",
            "padding_multiplex: int"})
    .SetKernelFn(PD_KERNEL(tokens_unzip_gather));
