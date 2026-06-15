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

template <typename T>
__global__ void tokens_unzip_slice_kernel(
    const T* __restrict__ x,
    const int* __restrict__ zipped_expertwise_rowmap,
    int64_t* __restrict__ index_out,
    int64_t total_zipped_rows,
    int num_experts,
    int start_idx,
    int end_idx) {
  const int64_t slice_len = end_idx - start_idx;
  if (slice_len <= 0) return;

  const int64_t total = total_zipped_rows * static_cast<int64_t>(num_experts);

  for (int64_t elem =
           static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
           static_cast<int64_t>(threadIdx.x);
       elem < total;
       elem += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t row = elem / num_experts;
    int64_t u = zipped_expertwise_rowmap[elem];
    if (u < 0) continue;
    if (u < start_idx || u >= end_idx) continue;
    int64_t out = static_cast<int64_t>(u) - start_idx;
    index_out[out] = row;
  }
}

std::vector<paddle::Tensor> tokens_unzip_slice(
    const paddle::Tensor& x,
    const paddle::Tensor& zipped_expertwise_rowmap,
    const int num_experts,
    const int total_unzipped_rows,
    const int start_idx,
    const int end_idx) {
  auto dtype = x.dtype();
  auto place = x.place();
  auto stream = x.stream();
  auto x_shape = x.shape();
  PD_CHECK(x_shape.size() == 2);
  int64_t total_zipped_rows = x_shape[0];

  auto index_unzipped =
      paddle::full({total_unzipped_rows}, -1, paddle::DataType::INT64, place);
  if (total_zipped_rows == 0) {
    return {index_unzipped};
  }
  int block = 1024;
  int grid = LimitGridDim(total_zipped_rows);

#define LAUNCH_TOKENS_UNZIP_SLICE_KERNEL_IMPL(__cpp_dtype)                 \
  do {                                                                     \
    tokens_unzip_slice_kernel<__cpp_dtype>                                 \
        <<<grid, block, 0, stream>>>(x.data<__cpp_dtype>(),                \
                                     zipped_expertwise_rowmap.data<int>(), \
                                     index_unzipped.data<int64_t>(),       \
                                     total_zipped_rows,                    \
                                     num_experts,                          \
                                     start_idx,                            \
                                     end_idx);                             \
  } while (0)

#define LAUNCH_TOKENS_UNZIP_SLICE_KERNEL(__cpp_dtype)   \
  do {                                                  \
    LAUNCH_TOKENS_UNZIP_SLICE_KERNEL_IMPL(__cpp_dtype); \
  } while (0)

  if (grid > 0) {
    LAUNCH_TOKENS_UNZIP_SLICE_KERNEL(phi::float8_e4m3fn);
  }

  return {index_unzipped};
}

PD_BUILD_OP(tokens_unzip_slice)
    .Inputs({"x", "zipped_expertwise_rowmap"})
    .Outputs({"idx_unzipped"})
    .Attrs({"num_experts: int",
            "total_unzipped_rows: int",
            "start_idx: int",
            "end_idx: int"})
    .SetKernelFn(PD_KERNEL(tokens_unzip_slice));
