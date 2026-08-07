#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(pwd)

mkdir -p "${ROOT_DIR}/coveragedata"
if [ ! -e "${ROOT_DIR}/PaddleFormers" ]; then
  ln -s . "${ROOT_DIR}/PaddleFormers"
fi

bash -x PaddleFormers/tests/integration_test/glm45_pt.sh
bash -x PaddleFormers/tests/integration_test/qwen.sh pt
bash -x PaddleFormers/tests/integration_test/qwen.sh sft
bash -x PaddleFormers/tests/integration_test/qwen.sh lora
