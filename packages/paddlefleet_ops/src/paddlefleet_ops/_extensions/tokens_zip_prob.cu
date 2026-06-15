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

#include <limits>

#include "paddle/common/array.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "utils.h"  // NOLINT

template <typename T>
struct UnzippedProbInfo {
  const T* __restrict__ data;
  int64_t offset;
};

template <typename T, int MAX_NUM_EXPERTS_C>
__global__ void tokens_zip_prob_kernel(
    phi::Array<UnzippedProbInfo<T>, MAX_NUM_EXPERTS_C> unzipped_probs,
    const int* __restrict__ zipped_expertwise_rowmap,
    const int* __restrict__ dispatched_indices,
    T* zipped_probs,
    int64_t zipped_rows,
    int topk,
    int num_expert) {
  int64_t idx = threadIdx.x + static_cast<int64_t>(blockDim.x) * blockIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t limit = zipped_rows * topk;
  while (idx < limit) {
    auto zipped_row = idx / topk;
    auto topk_idx = idx % topk;
    auto expert_id = dispatched_indices[idx];
    T value = static_cast<T>(0);
    if (expert_id >= 0) {
      auto unzipped_row =
          zipped_expertwise_rowmap[zipped_row * num_expert + expert_id];
      if (unzipped_row >= 0) {
        unzipped_row -= unzipped_probs[expert_id].offset;
        value = unzipped_probs[expert_id].data[unzipped_row];
      }
    }
    zipped_probs[idx] = value;
    idx += stride;
  }
}

template <typename T>
std::vector<paddle::Tensor> tokens_zip_prob_impl(
    const std::vector<paddle::Tensor>& unzipped_probs,
    const paddle::Tensor& zipped_expertwise_rowmap,
    const paddle::Tensor& dispatched_indices,
    paddle::DataType dtype) {
  auto zipped_expertwise_rowmap_shape = zipped_expertwise_rowmap.shape();
  auto dispatched_indices_shape = dispatched_indices.shape();
  PD_CHECK(zipped_expertwise_rowmap_shape.size() == 2);
  PD_CHECK(dispatched_indices_shape.size() == 2);
  PD_CHECK(zipped_expertwise_rowmap_shape[0] == dispatched_indices_shape[0]);

  int64_t zipped_rows = zipped_expertwise_rowmap_shape[0];
  int64_t num_expert = zipped_expertwise_rowmap_shape[1];
  int64_t topk = dispatched_indices_shape[1];
  PD_CHECK(num_expert <= static_cast<int64_t>(std::numeric_limits<int>::max()),
           "num_expert must be <= INT_MAX for tokens_zip_prob.");
  PD_CHECK(topk <= static_cast<int64_t>(std::numeric_limits<int>::max()),
           "topk must be <= INT_MAX for tokens_zip_prob.");
  PD_CHECK(unzipped_probs.size() == static_cast<size_t>(num_expert) &&
               num_expert > 0,
           "unzipped_probs.size() must equal num_expert.");
  int num_expert_int = static_cast<int>(num_expert);
  int topk_int = static_cast<int>(topk);

  auto zipped_probs =
      paddle::empty({zipped_rows, topk}, dtype, unzipped_probs[0].place());

  PD_SWITCH_NUM_EXPERTS(
      static_cast<int>(num_expert), ([&] {
        phi::Array<UnzippedProbInfo<T>, MAX_NUM_EXPERTS_C> unzipped_probs_info;
        int64_t offset = 0;
        for (int i = 0; i < num_expert_int; ++i) {
          auto shape = unzipped_probs[i].shape();
          PD_CHECK(shape.size() == 1);
          unzipped_probs_info[i].data = unzipped_probs[i].data<T>();
          unzipped_probs_info[i].offset = offset;
          offset += shape[0];
        }

        int thread = 1024;
        int64_t total_items = zipped_rows * topk;
        int grid = LimitGridDim((total_items + thread - 1) / thread);

        if (grid > 0) {
          tokens_zip_prob_kernel<T, MAX_NUM_EXPERTS_C>
              <<<grid, thread, 0, zipped_probs.stream()>>>(
                  unzipped_probs_info,
                  zipped_expertwise_rowmap.data<int>(),
                  dispatched_indices.data<int>(),
                  zipped_probs.data<T>(),
                  zipped_rows,
                  topk_int,
                  num_expert_int);
        }
      }));
  return {zipped_probs};
}

std::vector<paddle::Tensor> tokens_zip_prob(
    const std::vector<paddle::Tensor>& unzipped_probs,
    const paddle::Tensor& zipped_expertwise_rowmap,
    const paddle::Tensor& dispatched_indices) {
  PD_CHECK(zipped_expertwise_rowmap.dtype() == paddle::DataType::INT32);
  PD_CHECK(dispatched_indices.dtype() == paddle::DataType::INT32);

  auto dtype = unzipped_probs[0].dtype();
  if (dtype == paddle::DataType::FLOAT32) {
    return tokens_zip_prob_impl<float>(
        unzipped_probs, zipped_expertwise_rowmap, dispatched_indices, dtype);
  } else if (dtype == paddle::DataType::BFLOAT16) {
    return tokens_zip_prob_impl<phi::bfloat16>(
        unzipped_probs, zipped_expertwise_rowmap, dispatched_indices, dtype);
  } else {
    PD_THROW("Unsupported data type: %s", dtype);
  }
}

template <typename T, typename UnZipProbPtrsT>
__global__ void tokens_zip_prob_seq_subbatch_kernel(
    UnZipProbPtrsT unzipped_probs,
    const int* __restrict__ zipped_expertwise_rowmap,
    const int* __restrict__ dispatched_indices,
    T* zipped_probs,
    int64_t zipped_rows,
    int topk,
    int num_expert,
    int64_t subbatch_rows) {
  int64_t idx = threadIdx.x + static_cast<int64_t>(blockDim.x) * blockIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t limit = zipped_rows * topk;
  while (idx < limit) {
    int64_t zipped_row = idx / topk;
    int64_t topk_idx = idx % topk;
    int64_t expert_id = dispatched_indices[idx];
    T value = static_cast<T>(0);
    if (expert_id >= 0) {
      int64_t unzipped_row =
          zipped_expertwise_rowmap[zipped_row * num_expert + expert_id];
      int64_t i = unzipped_row / subbatch_rows;
      int64_t j = unzipped_row % subbatch_rows;
      if (unzipped_row >= 0) {
        value = unzipped_probs[i][j];
      }
    }
    zipped_probs[idx] = value;
    idx += stride;
  }
}

template <typename T>
std::vector<paddle::Tensor> tokens_zip_prob_seq_subbatch_impl(
    const std::vector<paddle::Tensor>& unzipped_probs,
    const paddle::Tensor& zipped_expertwise_rowmap,
    const paddle::Tensor& dispatched_indices,
    int64_t subbatch_rows,
    paddle::DataType dtype) {
  auto zipped_expertwise_rowmap_shape = zipped_expertwise_rowmap.shape();
  auto dispatched_indices_shape = dispatched_indices.shape();
  PD_CHECK(zipped_expertwise_rowmap_shape.size() == 2);
  PD_CHECK(dispatched_indices_shape.size() == 2);
  PD_CHECK(zipped_expertwise_rowmap_shape[0] == dispatched_indices_shape[0]);

  int64_t zipped_rows = zipped_expertwise_rowmap_shape[0];
  int64_t num_expert = zipped_expertwise_rowmap_shape[1];
  int64_t topk = dispatched_indices_shape[1];
  PD_CHECK(num_expert > 0,
           "num_expert must be > 0 for tokens_zip_prob_seq_subbatch.");
  PD_CHECK(num_expert <= static_cast<int64_t>(std::numeric_limits<int>::max()),
           "num_expert must be <= INT_MAX for tokens_zip_prob_seq_subbatch.");
  PD_CHECK(topk <= static_cast<int64_t>(std::numeric_limits<int>::max()),
           "topk must be <= INT_MAX for tokens_zip_prob_seq_subbatch.");
  PD_CHECK(subbatch_rows > 0,
           "subbatch_rows must be > 0 for tokens_zip_prob_seq_subbatch.");
  int num_expert_int = static_cast<int>(num_expert);
  int topk_int = static_cast<int>(topk);

  auto zipped_probs =
      paddle::empty({zipped_rows, topk}, dtype, unzipped_probs[0].place());
  int thread = 1024;
  int64_t total_items = zipped_rows * topk;
  int grid = LimitGridDim((total_items + thread - 1) / thread);

#define LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_FIX_CASE(__T, __num_split)       \
  if (unzipped_probs.size() <= __num_split) {                                \
    phi::Array<const __T*, __num_split> unzipped_probs_info;                 \
    for (size_t i = 0; i < unzipped_probs.size(); ++i) {                     \
      unzipped_probs_info[i] = unzipped_probs[i].data<__T>();                \
    }                                                                        \
    LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_CASE_IMPL(__T, unzipped_probs_info); \
    break;                                                                   \
  }

#define LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_DYNAMIC_CASE(__T)              \
  paddle::Tensor ptr_tensor;                                               \
  auto unzipped_probs_info =                                               \
      GetTensorDevicePtrs<const __T>(unzipped_probs,                       \
                                     &ptr_tensor,                          \
                                     zipped_probs.stream(),                \
                                     zipped_probs.place());                \
  LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_CASE_IMPL(__T, unzipped_probs_info); \
  break

#define LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_CASE_IMPL(__T,                   \
                                                      __unzipped_probs_info) \
  do {                                                                       \
    if (grid > 0) {                                                          \
      tokens_zip_prob_seq_subbatch_kernel<                                   \
          __T,                                                               \
          typename std::remove_reference<                                    \
              decltype(__unzipped_probs_info)>::type>                        \
          <<<grid, thread, 0, zipped_probs.stream()>>>(                      \
              __unzipped_probs_info,                                         \
              zipped_expertwise_rowmap.data<int>(),                          \
              dispatched_indices.data<int>(),                                \
              zipped_probs.data<__T>(),                                      \
              zipped_rows,                                                   \
              topk_int,                                                      \
              num_expert_int,                                                \
              subbatch_rows);                                                \
    }                                                                        \
  } while (0)

#define LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH(__T)           \
  do {                                                     \
    LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_FIX_CASE(__T, 1);  \
    LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_FIX_CASE(__T, 2);  \
    LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_FIX_CASE(__T, 3);  \
    LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_FIX_CASE(__T, 4);  \
    LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_FIX_CASE(__T, 5);  \
    LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_FIX_CASE(__T, 6);  \
    LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_FIX_CASE(__T, 7);  \
    LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_FIX_CASE(__T, 8);  \
    LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_FIX_CASE(__T, 9);  \
    LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_FIX_CASE(__T, 10); \
    LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_FIX_CASE(__T, 11); \
    LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_FIX_CASE(__T, 12); \
    LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_FIX_CASE(__T, 13); \
    LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_FIX_CASE(__T, 14); \
    LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_FIX_CASE(__T, 15); \
    LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_FIX_CASE(__T, 16); \
    LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_FIX_CASE(__T, 17); \
    LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_FIX_CASE(__T, 18); \
    LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_FIX_CASE(__T, 19); \
    LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_FIX_CASE(__T, 20); \
    LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH_DYNAMIC_CASE(__T); \
  } while (0)

  LAUNCH_TOKENS_ZIP_PROB_SEQ_SUBBATCH(T);
  return {zipped_probs};
}

std::vector<paddle::Tensor> tokens_zip_prob_seq_subbatch(
    const std::vector<paddle::Tensor>& unzipped_probs,
    const paddle::Tensor& zipped_expertwise_rowmap,
    const paddle::Tensor& dispatched_indices,
    int64_t subbatch_rows) {
  PD_CHECK(zipped_expertwise_rowmap.dtype() == paddle::DataType::INT32);
  PD_CHECK(dispatched_indices.dtype() == paddle::DataType::INT32);

  auto dtype = unzipped_probs[0].dtype();
  if (dtype == paddle::DataType::FLOAT32) {
    return tokens_zip_prob_seq_subbatch_impl<float>(unzipped_probs,
                                                    zipped_expertwise_rowmap,
                                                    dispatched_indices,
                                                    subbatch_rows,
                                                    dtype);
  } else if (dtype == paddle::DataType::BFLOAT16) {
    return tokens_zip_prob_seq_subbatch_impl<phi::bfloat16>(
        unzipped_probs,
        zipped_expertwise_rowmap,
        dispatched_indices,
        subbatch_rows,
        dtype);
  } else {
    PD_THROW("Unsupported data type: %s", dtype);
  }
}

PD_BUILD_OP(tokens_zip_prob)
    .Inputs({paddle::Vec("unzipped_prob"),
             "zipped_expertwise_rowmap",
             "dispatched_indices"})
    .Outputs({"zipped_prob"})
    .SetKernelFn(PD_KERNEL(tokens_zip_prob));

PD_BUILD_OP(tokens_zip_prob_seq_subbatch)
    .Inputs({paddle::Vec("unzipped_prob"),
             "zipped_expertwise_rowmap",
             "dispatched_indices"})
    .Outputs({"zipped_prob"})
    .Attrs({"subbatch_rows: int64_t"})
    .SetKernelFn(PD_KERNEL(tokens_zip_prob_seq_subbatch));
