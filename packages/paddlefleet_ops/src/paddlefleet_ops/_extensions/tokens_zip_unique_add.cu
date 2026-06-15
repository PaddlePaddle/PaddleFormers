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

template <typename ZipT, typename UnzipT, typename ZipPtrsT, int VecSize>
__global__ void tokens_zip_unique_add_kernel(
    ZipPtrsT zipped_ptrs,
    const UnzipT* __restrict__ unzipped,
    const int64_t* __restrict__ index_unzipped,
    const int64_t unzipped_rows,
    const int64_t subbatch_rows,
    const int hidden_size) {
  for (int64_t unzipped_row = blockIdx.x; unzipped_row < unzipped_rows;
       unzipped_row += gridDim.x) {
    int64_t zipped_row = index_unzipped[unzipped_row];
    if (zipped_row < 0) continue;
    auto* zipped_ptr = zipped_ptrs[zipped_row / subbatch_rows] +
                       (zipped_row % subbatch_rows) * hidden_size;
    const auto* unzipped_ptr = unzipped + unzipped_row * hidden_size;
    for (int64_t i = static_cast<int64_t>(threadIdx.x) * VecSize;
         i < hidden_size;
         i += static_cast<int64_t>(blockDim.x) * VecSize) {
      phi::AlignedVector<ZipT, VecSize> zipped_tmp;
      phi::AlignedVector<UnzipT, VecSize> unzipped_tmp;
      phi::Load(zipped_ptr + i, &zipped_tmp);
      phi::Load(unzipped_ptr + i, &unzipped_tmp);
#pragma unroll
      for (int j = 0; j < VecSize; ++j) {
        zipped_tmp[j] += static_cast<ZipT>(unzipped_tmp[j]);
      }
      phi::Store(zipped_tmp, zipped_ptr + i);
    }
  }
}

std::vector<paddle::Tensor> tokens_zip_unique_add_impl(
    const std::vector<paddle::Tensor>& zipped_origin,
    const paddle::Tensor& unzipped,
    const paddle::Tensor& index_unzipped,
    int64_t zipped_rows,
    int64_t subbatch_rows) {
  int64_t num_split = static_cast<int64_t>(zipped_origin.size());
  PD_CHECK(num_split >= 1, "num_split should be larger than or equal to 1");

  auto zipped_shape = zipped_origin[0].shape();
  auto unzipped_shape = unzipped.shape();
  PD_CHECK(zipped_shape.size() == 2);
  PD_CHECK(unzipped_shape.size() == 2);
  PD_CHECK(zipped_shape[1] == unzipped_shape[1]);

  auto hidden_size = zipped_shape[1];

  auto out_dtype = zipped_origin[0].dtype();
  auto in_dtype = unzipped.dtype();
  auto place = zipped_origin[0].place();

  if (zipped_rows <= 0) {
    return zipped_origin;
  }

  if (subbatch_rows <= 0) {
    subbatch_rows = zipped_rows;
  }
  subbatch_rows = std::min(zipped_rows, subbatch_rows);
  auto desired_num_split = (zipped_rows + subbatch_rows - 1) / subbatch_rows;
  auto remainder_rows = zipped_rows - (desired_num_split - 1) * subbatch_rows;

  std::vector<paddle::Tensor> zipped;
  zipped.reserve(desired_num_split);
  if (zipped_shape[0] == 0) {
    PD_CHECK(num_split == 1,
             "When input is 0-size tensor, it should be a single tensor "
             "instead of a tensor list");
    for (int64_t i = 0; i < desired_num_split; ++i) {
      auto tmp_rows =
          (i + 1 == desired_num_split ? remainder_rows : subbatch_rows);
      zipped.emplace_back(
          paddle::zeros({tmp_rows, hidden_size}, out_dtype, place));
    }
    num_split = desired_num_split;
  } else {
    PD_CHECK(num_split == desired_num_split);
    for (int64_t i = 0; i < desired_num_split; ++i) {
      auto tmp_shape = zipped_origin[i].shape();
      auto tmp_dtype = zipped_origin[i].dtype();
      PD_CHECK(tmp_shape.size() == 2);
      if (i + 1 == desired_num_split) {
        PD_CHECK(tmp_shape[0] == remainder_rows);
      } else {
        PD_CHECK(tmp_shape[0] == subbatch_rows);
      }
      PD_CHECK(tmp_shape[1] == hidden_size);
      PD_CHECK(tmp_dtype == out_dtype);

      zipped.emplace_back(zipped_origin[i]);
    }
  }

  auto index_shape = index_unzipped.shape();
  PD_CHECK(index_shape.size() == 1);
  auto unzipped_rows = index_shape[0];
  PD_CHECK(unzipped_rows <= zipped_rows);
  PD_CHECK(unzipped_rows <= unzipped_shape[0]);

  constexpr int kVecSize = 4;
  PD_CHECK(hidden_size % kVecSize == 0);

  int block = 1024;
  int grid = LimitGridDim(unzipped_rows);

  auto stream = unzipped.stream();
  paddle::Tensor ptr_tensor;

#define LAUNCH_TOKENS_ZIP_UNIQUE_ADD_CASE_IMPL(__ZipT, __UnzipT, __out_ptrs)  \
  do {                                                                        \
    auto stream = unzipped.stream();                                          \
    tokens_zip_unique_add_kernel<                                             \
        __ZipT,                                                               \
        __UnzipT,                                                             \
        typename std::remove_reference<decltype(__out_ptrs)>::type,           \
        kVecSize><<<grid, block, 0, stream>>>(__out_ptrs,                     \
                                              unzipped.data<__UnzipT>(),      \
                                              index_unzipped.data<int64_t>(), \
                                              unzipped_rows,                  \
                                              subbatch_rows,                  \
                                              hidden_size);                   \
  } while (0)

#define LAUNCH_TOKENS_ZIP_UNIQUE_ADD_FIX_CASE(__ZipT, __UnzipT, __num_split) \
  if (num_split <= __num_split) {                                            \
    phi::Array<__ZipT*, __num_split> array;                                  \
    for (int64_t i = 0; i < num_split; ++i) {                                \
      array[i] = zipped[i].data<__ZipT>();                                   \
    }                                                                        \
    LAUNCH_TOKENS_ZIP_UNIQUE_ADD_CASE_IMPL(__ZipT, __UnzipT, array);         \
    break;                                                                   \
  }

#define LAUNCH_TOKENS_ZIP_UNIQUE_ADD_DYNAMIC_CASE(__ZipT, __UnzipT)      \
  paddle::Tensor ptr_tensor;                                             \
  auto device_ptrs =                                                     \
      GetTensorDevicePtrs<__ZipT>(zipped, &ptr_tensor, stream, place);   \
  LAUNCH_TOKENS_ZIP_UNIQUE_ADD_CASE_IMPL(__ZipT, __UnzipT, device_ptrs); \
  break

#define LAUNCH_TOKENS_ZIP_UNIQUE_ADD(__ZipT, __UnzipT)           \
  do {                                                           \
    LAUNCH_TOKENS_ZIP_UNIQUE_ADD_FIX_CASE(__ZipT, __UnzipT, 1);  \
    LAUNCH_TOKENS_ZIP_UNIQUE_ADD_FIX_CASE(__ZipT, __UnzipT, 2);  \
    LAUNCH_TOKENS_ZIP_UNIQUE_ADD_FIX_CASE(__ZipT, __UnzipT, 4);  \
    LAUNCH_TOKENS_ZIP_UNIQUE_ADD_FIX_CASE(__ZipT, __UnzipT, 8);  \
    LAUNCH_TOKENS_ZIP_UNIQUE_ADD_FIX_CASE(__ZipT, __UnzipT, 16); \
    LAUNCH_TOKENS_ZIP_UNIQUE_ADD_DYNAMIC_CASE(__ZipT, __UnzipT); \
  } while (0)

  if (grid > 0) {
    if (out_dtype == paddle::DataType::FLOAT32 &&
        in_dtype == paddle::DataType::BFLOAT16) {
      LAUNCH_TOKENS_ZIP_UNIQUE_ADD(float, phi::bfloat16);
    } else if (out_dtype == paddle::DataType::BFLOAT16 &&
               in_dtype == out_dtype) {
      LAUNCH_TOKENS_ZIP_UNIQUE_ADD(phi::bfloat16, phi::bfloat16);
    } else if (out_dtype == paddle::DataType::FLOAT32 &&
               in_dtype == out_dtype) {
      LAUNCH_TOKENS_ZIP_UNIQUE_ADD(float, float);
    } else {
      PD_THROW("Unsupported data type");
    }
  }
  return zipped;
}

std::vector<paddle::Tensor> tokens_zip_unique_add(
    const paddle::Tensor& zipped_origin,
    const paddle::Tensor& unzipped,
    const paddle::Tensor& index_unzipped,
    int64_t zipped_rows) {
  return tokens_zip_unique_add_impl(
      {zipped_origin}, unzipped, index_unzipped, zipped_rows, 0);
}

void tokens_zip_unique_add_subbatch(
    const std::vector<paddle::Tensor>& zipped_origin,
    const paddle::Tensor& unzipped,
    const paddle::Tensor& index_unzipped,
    int64_t zipped_rows,
    int64_t subbatch_rows) {
  tokens_zip_unique_add_impl(
      zipped_origin, unzipped, index_unzipped, zipped_rows, subbatch_rows);
}

PD_BUILD_OP(tokens_zip_unique_add)
    .Inputs({"x_zipped", "x_unzipped", "idx_unzipped"})
    .Outputs({"y_zipped"})
    .Attrs({"zipped_rows: int64_t"})
    .SetKernelFn(PD_KERNEL(tokens_zip_unique_add));

PD_BUILD_OP(tokens_zip_unique_add_subbatch)
    .Inputs({paddle::Vec("x_zipped"), "x_unzipped", "idx_unzipped"})
    .Outputs({paddle::Vec("y_zipped")})
    .SetInplaceMap({{paddle::Vec("x_zipped"), paddle::Vec("y_zipped")}})
    .Attrs({"zipped_rows: int64_t", "subbatch_rows: int64_t"})
    .SetKernelFn(PD_KERNEL(tokens_zip_unique_add_subbatch));
