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

template <typename InT, typename OutT, typename InPtrsT, int VecSize>
__global__ void merge_subbatch_cast_kernel(const InPtrsT in_ptrs,
                                           OutT* __restrict__ out,
                                           int64_t total_num,
                                           int64_t subbatch_num) {
  int64_t idx =
      (threadIdx.x + static_cast<int64_t>(blockDim.x) * blockIdx.x) * VecSize;
  int64_t stride = (static_cast<int64_t>(blockDim.x) * gridDim.x) * VecSize;

  while (idx < total_num) {
    const InT* x_ptr = in_ptrs[idx / subbatch_num] + idx % subbatch_num;
    phi::AlignedVector<InT, VecSize> in_data;
    phi::Load(x_ptr, &in_data);
    if constexpr (std::is_same<InT, OutT>::value) {
      phi::Store(in_data, out + idx);
    } else {
      phi::AlignedVector<OutT, VecSize> out_data;
#pragma unroll
      for (int i = 0; i < VecSize; ++i) {
        out_data[i] = static_cast<OutT>(in_data[i]);
      }
      phi::Store(out_data, out + idx);
    }
    idx += stride;
  }
}

std::vector<paddle::Tensor> merge_subbatch_cast(
    const std::vector<paddle::Tensor>& x, int64_t int_dtype) {
  if (x.empty()) return {};

  auto in_dtype = x[0].dtype();
  auto merged_dtype = TransToDataType(int_dtype);

  auto place = x[0].place();
  auto merged_shape = x[0].shape();
  int64_t subbatch_rows = merged_shape[0];
  for (size_t i = 1; i < x.size(); ++i) {
    auto tmp_shape = x[i].shape();
    PD_CHECK(tmp_shape.size() == merged_shape.size());
    for (size_t j = 1; j < tmp_shape.size(); ++j) {
      PD_CHECK(tmp_shape[j] == merged_shape[j]);
    }
    if (i + 1 != x.size()) {
      PD_CHECK(tmp_shape[0] == subbatch_rows);
    } else {
      PD_CHECK(tmp_shape[0] <= subbatch_rows);
    }
    merged_shape[0] += tmp_shape[0];

    PD_CHECK(x[i].dtype() == in_dtype);
  }

  auto output = paddle::empty(merged_shape, merged_dtype, place);
  int64_t hidden_size = 1;
  for (size_t i = 1; i < merged_shape.size(); ++i) {
    hidden_size *= merged_shape[i];
  }

  int64_t total_num = merged_shape[0] * hidden_size;
  int64_t subbatch_num = subbatch_rows * hidden_size;

  constexpr int kVecSize = 4;
  PD_CHECK(total_num % kVecSize == 0);
  PD_CHECK(subbatch_num % kVecSize == 0);
  auto stream = output.stream();

  int thread = 1024;
  int grid = LimitGridDim((total_num / kVecSize + thread - 1) / thread);
  auto num_split = static_cast<int64_t>(x.size());

#define LAUNCH_MERGE_SUBBATCH_CAST_CASE_IMPL(__InT, __OutT, __in_ptrs) \
  do {                                                                 \
    merge_subbatch_cast_kernel<                                        \
        __InT,                                                         \
        __OutT,                                                        \
        typename std::remove_reference<decltype(__in_ptrs)>::type,     \
        kVecSize><<<grid, thread, 0, stream>>>(                        \
        __in_ptrs, output.data<__OutT>(), total_num, subbatch_num);    \
  } while (0)

#define LAUNCH_MERGE_SUBBATCH_CAST_FIX_CASE(__InT, __OutT, __num_split) \
  if (num_split <= __num_split) {                                       \
    phi::Array<const __InT*, __num_split> array;                        \
    for (int64_t i = 0; i < num_split; ++i) {                           \
      array[i] = x[i].data<__InT>();                                    \
    }                                                                   \
    LAUNCH_MERGE_SUBBATCH_CAST_CASE_IMPL(__InT, __OutT, array);         \
    break;                                                              \
  }

#define LAUNCH_MERGE_SUBBATCH_CAST_DYNAMIC_CASE(__InT, __OutT)      \
  paddle::Tensor ptr_tensor;                                        \
  auto device_ptrs =                                                \
      GetTensorDevicePtrs<__InT>(x, &ptr_tensor, stream, place);    \
  LAUNCH_MERGE_SUBBATCH_CAST_CASE_IMPL(__InT, __OutT, device_ptrs); \
  break

#define LAUNCH_MERGE_SUBBATCH_CAST(__InT, __OutT)           \
  do {                                                      \
    LAUNCH_MERGE_SUBBATCH_CAST_FIX_CASE(__InT, __OutT, 1);  \
    LAUNCH_MERGE_SUBBATCH_CAST_FIX_CASE(__InT, __OutT, 2);  \
    LAUNCH_MERGE_SUBBATCH_CAST_FIX_CASE(__InT, __OutT, 4);  \
    LAUNCH_MERGE_SUBBATCH_CAST_FIX_CASE(__InT, __OutT, 8);  \
    LAUNCH_MERGE_SUBBATCH_CAST_FIX_CASE(__InT, __OutT, 16); \
    LAUNCH_MERGE_SUBBATCH_CAST_DYNAMIC_CASE(__InT, __OutT); \
  } while (0)

  if (grid > 0) {
    if (in_dtype == paddle::DataType::FLOAT32 &&
        merged_dtype == paddle::DataType::BFLOAT16) {
      LAUNCH_MERGE_SUBBATCH_CAST(float, paddle::bfloat16);
    } else if (in_dtype == paddle::DataType::BFLOAT16 &&
               merged_dtype == paddle::DataType::FLOAT32) {
      LAUNCH_MERGE_SUBBATCH_CAST(paddle::bfloat16, float);
    } else if (in_dtype == paddle::DataType::FLOAT32 &&
               merged_dtype == paddle::DataType::FLOAT32) {
      LAUNCH_MERGE_SUBBATCH_CAST(float, float);
    } else if (in_dtype == paddle::DataType::BFLOAT16 &&
               merged_dtype == paddle::DataType::BFLOAT16) {
      LAUNCH_MERGE_SUBBATCH_CAST(paddle::bfloat16, paddle::bfloat16);
    } else {
      PD_THROW("Unsupported data type");
    }
  }

  return {output};
}

PD_BUILD_OP(merge_subbatch_cast)
    .Inputs({paddle::Vec("x")})
    .Outputs({"y"})
    .Attrs({"dtype: int64_t"})
    .SetKernelFn(PD_KERNEL(merge_subbatch_cast));
