#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(pwd)

mkdir -p "${ROOT_DIR}/coveragedata"
if [ ! -e "${ROOT_DIR}/PaddleFormers" ]; then
  ln -s . "${ROOT_DIR}/PaddleFormers"
fi

bash -x PaddleFormers/tests/integration_test/glm45_pt_single_card.sh
bash -x PaddleFormers/tests/integration_test/qwen3vl_sft_single_card.sh single
