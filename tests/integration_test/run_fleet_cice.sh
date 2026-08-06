# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
#
# 统一集成测试入口脚本。用于替代 tests/integration_test/ 下按模型/模式拆分的
# glm45_*.sh / qwen*.sh / qwen3vl_*.sh 等脚本，合并训练执行、失败重试检查、
# loss 精度比对、精度审批兜底流程等公共逻辑，通过 <model> <mode> [machine]
# 三个参数驱动具体 case。
#
# 用法:
#   run_fleet_cice.sh <model> <mode> [machine]
#
# 示例:
#   run_fleet_cice.sh glm45 pt
#   run_fleet_cice.sh glm45 sft
#   run_fleet_cice.sh glm45 lora
#   run_fleet_cice.sh glm45 dpo a100
#   run_fleet_cice.sh qwen sft
#   run_fleet_cice.sh qwen3vl lora h20
#
# model/mode 完整映射表及 checkpoint 依赖关系见:
#   tests/integration_test/design_docs/case_matrix.md

set -exo pipefail
export root_dir=$(pwd)

model=$1
mode=$2
machine=${3:-h20}

if [[ -z "$model" || -z "$mode" ]]; then
  echo -e "::error:: \033[31mUsage: run_fleet_cice.sh <model> <mode> [machine]\033[0m"
  exit 1
fi

if [ -f 'PaddleFleet/.venv/bin/activate' ]; then
   source PaddleFleet/.venv/bin/activate
fi

config_ci_dir=$root_dir/PaddleFormers/tests/config/ci
fixtures_dir=$root_dir/PaddleFormers/tests/fixtures/dummy

# ---------------------------------------------------------------------------
# 公共函数
# ---------------------------------------------------------------------------

# 校验前置阶段产出的 checkpoint 是否存在，防止某一步没有产出时静默地用错误路径继续跑。
require_checkpoint() {
  local path=$1
  local desc=$2
  if [[ ! -d "$path" ]] || [[ -z "$(ls -A "$path" 2>/dev/null)" ]]; then
    echo -e "::error:: \033[31m[$model/$mode] 依赖的前置 checkpoint 不存在或为空: $path ($desc)\033[0m"
    echo -e "\033[31m请先运行对应的前置 case 生成该 checkpoint。\033[0m"
    exit 1
  fi
}

# 执行训练命令，失败时通过日志内容二次确认（兼容偶发的非零退出码但训练实际已完成的情况）。
# $1: 训练用的 config yaml 路径
# $2: 日志文件路径
# $3: 用于在日志里确认训练完成的匹配字符串
run_training() {
  local config_yaml=$1
  local log_file=$2
  local check_string=$3

  set +e
  NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port coverage run $(which paddleformers-cli) train "$config_yaml" 2>&1 | tee "./${log_file}"

  local exit_code=$?
  if [ $exit_code -ne 0 ]; then
    echo "[$model/$mode] training failed, try to check the log file"
    python "$root_dir/PaddleFormers/tests/check_log_for_exitcode.py" "./${log_file}" "$check_string"
    local check_exit_code=$?
    if [ $check_exit_code -ne 0 ]; then
      echo "Failed to find '$check_string' in log file."
      exit 1
    else
      echo "Log check passed."
    fi
  else
    echo "Test passed."
  fi
}

# 下载 ground-truth loss、执行 loss 比对，比对失败时走精度变更审批兜底流程。
# $1: 日志文件路径
# $2: 是否传 --compare_step 10（"true"/"false"）
check_precision() {
  local log_file=$1
  local use_compare_step=$2
  local gt_loss_file=${log_file%.*}_gt_loss.txt

  export repo_name=PaddleFleet
  export REPO_NAME=$(echo $GITHUB_REPO_NAME | awk -F'/' '{print $2}')

  wget --no-proxy --no-check-certificate "https://xly-devops.cdn.bcebos.com/PaddleFleet/precision/${REPO_NAME}${pfpatch}${pppatch}_latest/${gt_loss_file}"
  if [ $? -ne 0 ]; then
    echo "To request precision checks for new models, please contact swgu98."
    exit 1
  fi

  local log_loss_file=${log_file%.*}_loss.${log_file##*.}
  local compare_args=()
  if [[ "$use_compare_step" == "true" ]]; then
    compare_args=(--compare_step 10)
  fi

  python "$root_dir/PaddleFormers/tests/integration_test/check_loss.py" \
    "${compare_args[@]}" \
    --log_file "./${log_file}" \
    --log_loss_file "./${log_loss_file}" \
    --gt_file "./${gt_loss_file}"

  if [ $? -ne 0 ]; then
    if [ "${BRANCH}" != "develop" ]; then
      echo "please update precision in develop and rerun this workflow"
      exit 1
    fi
    pushd "$root_dir/PaddleFormers"
    source /root/proxy
    bash "$root_dir/PaddleFormers/tests/integration_test/check_precision_approval.sh"
    if [ $? -ne 0 ]; then
      echo -e "\033[31mThe precision has been changed and requires approvals.\033[0m"
      exit 1
    fi
    popd
    rm "${gt_loss_file}" && mv "${log_loss_file}" "${gt_loss_file}"
    if [ ! -f precision_list.txt ]; then
      wget --no-proxy --no-check-certificate "https://paddle-github-action.cdn.bcebos.com/PaddleFleet/precision/${REPO_NAME}${pfpatch}${pppatch}/${PR_ID}/precision_list.txt"
      if [ $? -ne 0 ]; then
        wget --no-proxy --no-check-certificate "https://xly-devops.cdn.bcebos.com/PaddleFleet/precision/${repo_name}${pfpatch}${pppatch}_latest/precision_list.txt"
        python "$root_dir/bos/BosClient.py" precision_list.txt "paddle-github-action/PaddleFleet/precision/${REPO_NAME}${pfpatch}${pppatch}/${PR_ID}"
      fi
    fi
    python "$root_dir/bos/BosClient.py" "${gt_loss_file}" "paddle-github-action/PaddleFleet/precision/${REPO_NAME}${pfpatch}${pppatch}/${PR_ID}"
  fi
}

yq_write() {
  local config_yaml=$1
  local expr=$2
  yq eval "$expr" "$config_yaml" > "${config_yaml}.tmp"
  mv "${config_yaml}.tmp" "$config_yaml"
}

# ---------------------------------------------------------------------------
# model/mode 分支：只负责准备 config_yaml / model_name_or_path / output_dir /
# checkpoint 依赖校验 / log_file 命名，具体训练与校验逻辑统一在后面执行。
# ---------------------------------------------------------------------------

# 训练完成后用哪个字符串确认日志里出现了预期的结束标记，默认 train metrics。
check_string="***** train metrics *****"
# loss 比对时是否传 --compare_step 10，默认传（多卡训练场景）。
use_compare_step=true
# 是否需要提前解压 glm45_fleet.12-18.tar 并 cd 进去（GLM4.5 多卡系列的公共前置步骤）。
need_glm45_fleet=false

case "$model" in
  glm45|glm45_ep4)
    need_glm45_fleet=true
    if [[ "$mode" == "pt" ]]; then
      config_yaml=$config_ci_dir/glm45_pt.yaml
      data_dir=$fixtures_dir/pt
    elif [[ "$mode" == "sft" ]]; then
      config_yaml=$config_ci_dir/glm45_sft.yaml
      data_dir=$fixtures_dir/sft
    elif [[ "$mode" == "sft_cp" ]]; then
      config_yaml=$config_ci_dir/glm45_sft_cp.yaml
      data_dir=$fixtures_dir/sft
    elif [[ "$mode" == "lora" ]]; then
      config_yaml=$config_ci_dir/glm45_lora.yaml
      data_dir=$fixtures_dir/sft
    elif [[ "$mode" == "dpo" ]]; then
      config_yaml=$config_ci_dir/glm45_dpo.yaml
      data_dir=$fixtures_dir/dpo
      check_string="***** eval metrics *****"
    elif [[ "$mode" == "dpo_lora" ]]; then
      config_yaml=$config_ci_dir/glm45_dpo_lora.yaml
      data_dir=$fixtures_dir/dpo
      check_string="***** eval metrics *****"
    else
      echo -e "::error:: unsupported mode '$mode' for model '$model'"
      exit 1
    fi
    ;;
  glm45_fp8)
    config_yaml=$config_ci_dir/glm45_pt_fp8.yaml
    data_dir=$fixtures_dir/pt
    ;;
  glm45_grouped_gemm)
    config_yaml=$config_ci_dir/glm45_pt_grouped_gemm.yaml
    data_dir=$fixtures_dir/pt
    ;;
  glm45_single)
    config_yaml=$config_ci_dir/glm45_single_pt-test.yaml
    use_compare_step=false
    ;;
  qwen)
    if [[ "$mode" != "pt" && "$mode" != "sft" && "$mode" != "lora" ]]; then
      echo -e "::error:: unsupported mode '$mode' for model '$model'"
      exit 1
    fi
    if [[ ! -d $CACHE_DIR/Qwen3-30B-A3B ]]; then
      pushd $CACHE_DIR
      wget -q --tries=5 --no-proxy https://xly-devops.cdn.bcebos.com/PaddleFleet/Qwen/Qwen3-30B-A3B.tar.gz --no-check-certificate
      tar xf Qwen3-30B-A3B.tar.gz
      popd
    fi
    if [[ "$mode" == "pt" ]]; then
      config_yaml=$config_ci_dir/qwen3_multicard_pt.yaml
      data_dir=$fixtures_dir/pt
      model_name_or_path=$CACHE_DIR/Qwen3-30B-A3B
      output_dir=$root_dir/checkpoints/qwen-pt
    elif [[ "$mode" == "sft" ]]; then
      config_yaml=$config_ci_dir/qwen3_multicard_sft.yaml
      data_dir=$fixtures_dir/sft
      model_name_or_path=$root_dir/checkpoints/qwen-pt
      output_dir=$root_dir/checkpoints/qwen-sft
    else
      config_yaml=$config_ci_dir/qwen3_multicard_lora.yaml
      data_dir=$fixtures_dir/sft
      model_name_or_path=$root_dir/checkpoints/qwen-sft
      output_dir=$root_dir/checkpoints/qwen-lora
    fi
    if [[ "$machine" == "a100" && ( "$mode" == "pt" || "$mode" == "sft" ) ]]; then
      yq_write "$config_yaml" '.moe_expert_fusion = false | .stage1_overlap = false'
    fi
    ;;
  qwen_single)
    config_yaml=$config_ci_dir/qwen3_pt.yaml
    use_compare_step=false
    ;;
  qwen3vl|qwen3vl_moe|qwen3vl_fsdp)
    if [[ "$model" == "qwen3vl" && "$mode" == "sft" ]]; then
      config_yaml=$config_ci_dir/qwen3vl_sft.yaml
      data_dir=$fixtures_dir/sft-vl
      model_name_or_path=$CACHE_DIR/qwen3vl/tiny-random-qwen3vlv2
      output_dir=$root_dir/checkpoints/qwen3vl-sft
    elif [[ "$model" == "qwen3vl" && "$mode" == "lora" ]]; then
      config_yaml=$config_ci_dir/qwen3vl_lora.yaml
      data_dir=$fixtures_dir/sft-vl
      model_name_or_path=$root_dir/checkpoints/qwen3vl-sft
      output_dir=$root_dir/checkpoints/qwen3vl-lora
    elif [[ "$model" == "qwen3vl_moe" && "$mode" == "sft" ]]; then
      if [[ "$machine" == "a100" ]]; then
        config_yaml=$config_ci_dir/qwen3vl_sft_moe_a100.yaml
      else
        config_yaml=$config_ci_dir/qwen3vl_sft_moe.yaml
      fi
      data_dir=$fixtures_dir/sft-vl
      model_name_or_path=$CACHE_DIR/qwen3vl/tiny-random-qwen3vlmoev2
      output_dir=$root_dir/checkpoints/qwen3vl-moe
    elif [[ "$model" == "qwen3vl_fsdp" && "$mode" == "sft" ]]; then
      config_yaml=$config_ci_dir/qwen3vl_sft_fsdp.yaml
      data_dir=$fixtures_dir/sft-vl
      model_name_or_path=$CACHE_DIR/qwen3vl/tiny-random-qwen3vlv2
      output_dir=$root_dir/checkpoints/qwen3vl-fsdp
    else
      echo -e "::error:: unsupported mode '$mode' for model '$model'"
      exit 1
    fi
    if [[ ! -d $data_dir/DoclingMatix ]]; then
      wget https://paddleformers.bj.bcebos.com/datasets/DoclingMatix.tar.gz
      tar -xf DoclingMatix.tar.gz -C $data_dir
      rm -rf DoclingMatix.tar.gz
    fi
    ;;
  qwen3vl_single)
    config_yaml=$config_ci_dir/qwen3vl_sft_single.yaml
    data_dir=$fixtures_dir/sft-vl
    model_name_or_path=$CACHE_DIR/qwen3vl/tiny-random-qwen3vlv2
    output_dir=$root_dir/checkpoints/qwen3vl-single
    if [[ ! -d $data_dir/DoclingMatix ]]; then
      wget https://paddleformers.bj.bcebos.com/datasets/DoclingMatix.tar.gz
      tar -xf DoclingMatix.tar.gz -C $data_dir
      rm -rf DoclingMatix.tar.gz
    fi
    ;;
  *)
    echo -e "::error:: \033[31munsupported model '$model'\033[0m"
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# GLM4.5 多卡系列公共前置：下载/解压 glm45_fleet，cd 进入并计算路径
# （pt/sft/sft_cp/lora/dpo/dpo_lora/ep4 都基于这份目录布局）
# ---------------------------------------------------------------------------
if [[ "$need_glm45_fleet" == "true" ]]; then
  if [[ ! -d "$root_dir/glm45_fleet" ]]; then
    wget -q --tries=5 --no-proxy https://xly-devops.cdn.bcebos.com/PaddleFleet/glm45/glm45_fleet.12-18.tar --no-check-certificate
    tar -xf glm45_fleet.12-18.tar
  fi
  cd "$root_dir/glm45_fleet"
  export cur_dir=$(pwd)

  case "$mode" in
    pt)
      model_name_or_path=$cur_dir/GLM-4.5-Air
      output_dir=$cur_dir/checkpoints/pretrain
      logging_dir=$cur_dir/vdl_log
      ;;
    sft)
      model_name_or_path=$cur_dir/checkpoints/pretrain
      output_dir=$cur_dir/checkpoints/glm_full_pp_ckpts
      logging_dir=$cur_dir/glm_full_pp_vdl_log
      ;;
    sft_cp)
      model_name_or_path=$cur_dir/checkpoints/pretrain
      output_dir=$cur_dir/checkpoints/glm_full_pp_cp_ckpts
      logging_dir=$cur_dir/glm_full_pp_cp_vdl_log
      ;;
    lora)
      model_name_or_path=$cur_dir/checkpoints/glm_full_pp_ckpts
      output_dir=$cur_dir/checkpoints/glm_single_lora_ckps
      logging_dir=$cur_dir/glm_full_single_lora_log
      ;;
    dpo)
      model_name_or_path=$cur_dir/checkpoints/glm_full_pp_ckpts
      output_dir=$cur_dir/checkpoints/glm_full_dpo_ckpts
      logging_dir=$cur_dir/glm_full_dpo_vdl_log
      ;;
    dpo_lora)
      # dpo_lora 不依赖本地 checkpoint，直接从 CACHE_DIR 下预置的 base 模型开始训练
      model_name_or_path=$CACHE_DIR/zai-org/GLM-4.5-Air-Base
      output_dir=$cur_dir/checkpoints/glm_full_dpo_lora_ckpts
      logging_dir=$cur_dir/glm_full_dpo_lora_vdl_log
      ;;
  esac

  # sft/sft_cp/lora/dpo 均读取前一阶段产出的 checkpoint，需要先校验其存在
  case "$mode" in
    sft|sft_cp)
      require_checkpoint "$model_name_or_path" "glm45 pt 产出"
      ;;
    lora|dpo)
      require_checkpoint "$model_name_or_path" "glm45 sft 产出"
      ;;
  esac

  yq_expr='.train_dataset_path = strenv(data_dir) + "/train.jsonl"
    | .eval_dataset_path = strenv(data_dir) + "/eval.jsonl"
    | .model_name_or_path = strenv(model_name_or_path)
    | .logging_dir = strenv(logging_dir)
    | .output_dir = strenv(output_dir)'
  if [[ "$mode" == "pt" && "$model" == "glm45_ep4" ]]; then
    yq_expr="$yq_expr"' | .expert_model_parallel_size = 4'
  fi
  if [[ "$machine" == "a100" ]]; then
    case "$mode" in
      pt)
        yq_expr="$yq_expr"' | .expert_model_parallel_size = 1 | .num_hidden_layers = 2 | .per_device_eval_batch_size = 1 | .per_device_train_batch_size = 1 | .use_expert_parallel = false | .stage1_overlap = false'
        ;;
      sft)
        yq_expr="$yq_expr"' | .use_expert_parallel = false | .expert_model_parallel_size = 1 | .per_device_train_batch_size = 1 | .tensorwise_offload_optimizer = true | .stage1_overlap = false | .num_empty_layers_add_in_head = 0'
        ;;
      lora)
        yq_expr="$yq_expr"' | .num_empty_layers_add_in_tail = 0 | .use_expert_parallel = false | .expert_model_parallel_size = 1 | del(.moe_token_dispatcher_type)'
        ;;
      dpo)
        yq_expr="$yq_expr"' | .num_empty_layers_add_in_tail = 0 | .use_expert_parallel = false | .tensorwise_offload_optimizer = true | .expert_model_parallel_size = 1'
        ;;
      dpo_lora)
        yq_expr="$yq_expr"' | .use_expert_parallel = false | .expert_model_parallel_size = 1'
        ;;
    esac
  fi
  export data_dir model_name_or_path output_dir logging_dir
  yq_write "$config_yaml" "$yq_expr"
fi

# ---------------------------------------------------------------------------
# GLM4.5 独立 pt 变体 (fp8 / grouped_gemm)：不依赖 glm45_fleet 目录，
# model_name_or_path 直接读 CACHE_DIR 下预置的 base 模型。
# ---------------------------------------------------------------------------
if [[ "$model" == "glm45_fp8" || "$model" == "glm45_grouped_gemm" ]]; then
  model_name_or_path=$CACHE_DIR/glm45/GLM-4.5-Air
  output_dir=$data_dir/checkpoints
  export data_dir model_name_or_path
  yq_write "$config_yaml" '.train_dataset_path = strenv(data_dir) + "/train.jsonl"
    | .eval_dataset_path = strenv(data_dir) + "/eval.jsonl"
    | .model_name_or_path = strenv(model_name_or_path)
    | .logging_dir = strenv(data_dir) + "/vdl_log"
    | .output_dir = strenv(data_dir) + "/checkpoints"'
fi

# ---------------------------------------------------------------------------
# Qwen3-30B-A3B 多卡系列 (pt/sft/lora)：统一 yq 覆盖
# ---------------------------------------------------------------------------
if [[ "$model" == "qwen" ]]; then
  export data_dir model_name_or_path output_dir
  yq_expr='.train_dataset_path = strenv(data_dir) + "/train.jsonl"
    | .eval_dataset_path = strenv(data_dir) + "/eval.jsonl"
    | .model_name_or_path = strenv(model_name_or_path)
    | .output_dir = strenv(output_dir)'
  if [[ "$machine" == "a100" ]]; then
    yq_expr="$yq_expr"' | .use_expert_parallel = false | .expert_model_parallel_size = 1'
  fi
  yq_write "$config_yaml" "$yq_expr"
  if [[ "$mode" == "sft" ]]; then
    require_checkpoint "$model_name_or_path" "qwen pt 产出"
  elif [[ "$mode" == "lora" ]]; then
    require_checkpoint "$model_name_or_path" "qwen sft 产出"
  fi
fi

# ---------------------------------------------------------------------------
# Qwen3-30B-A3B 单卡 pt：沿用原脚本的字段名与数据路径（历史行为，见
# design_docs/implementation_notes.md 第 3 条，未做修复）
# ---------------------------------------------------------------------------
if [[ "$model" == "qwen_single" ]]; then
  yq eval '
    .save_steps = 100 |
    .input_dir = "1.0 '"${CACHE_DIR}"'/glm45/data/pre-training/llama_openwebtext_100k" |
    .model_name_or_path = "'"${CACHE_DIR}"'/qwen/Qwen3-30B-A3B-Base"
  ' "$config_yaml" -i
  cat "$config_yaml"
fi

# ---------------------------------------------------------------------------
# Qwen3-VL 系列 (sft/lora/moe/fsdp/single)：统一 yq 覆盖
# ---------------------------------------------------------------------------
if [[ "$model" == qwen3vl* ]]; then
  export data_dir model_name_or_path output_dir
  yq_write "$config_yaml" '.train_dataset_path = strenv(data_dir) + "/train.jsonl"
    | .eval_dataset_path = strenv(data_dir) + "/train.jsonl"
    | .model_name_or_path = strenv(model_name_or_path)
    | .output_dir = strenv(output_dir)'
  if [[ "$model" == "qwen3vl" && "$mode" == "lora" ]]; then
    require_checkpoint "$model_name_or_path" "qwen3vl sft 产出"
  fi
fi

# ---------------------------------------------------------------------------
# glm45_single / qwen_single 不需要任何 yq 覆盖，直接使用 config 文件里的固定值
# （与原 glm45_pt_single_card.sh / qwen3_single_card.sh 行为一致）
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 公共执行部分：清理产物、设置确定性/性能 flags、跑训练、跑精度比对
# ---------------------------------------------------------------------------
rm -rf ./outputs
rm -rf checkpoint/
rm -rf paddleformers_dist_log
master=$(hostname -i)
port=36677

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export FLAGS_use_stride_compute_kernel=False
# 不设置 CUDA_VISIBLE_DEVICES 的 case：单卡 case，以及 qwen3vl 的 lora/moe/single
# （与原脚本行为一致：qwen3vl_sft.sh 的 tp8/fsdp 分支设置了 8 卡，但 qwen3vl_lora.sh 和
# qwen3vl_sft_single_card.sh 没有设置）。
case "$model.$mode" in
  glm45_single.*|qwen_single.*|qwen3vl_single.*|qwen3vl.lora|qwen3vl_moe.*)
    ;;
  *)
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    ;;
esac

unset http_proxy https_proxy

log_file=${model}_${mode}_${machine}.txt

run_training "$config_yaml" "$log_file" "$check_string"
check_precision "$log_file" "$use_compare_step"