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

if [ -z "${BRANCH:-}" ]; then
    BRANCH="develop"
fi

PADDLE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"

UPSTREAM_BRANCH="upstream/${BRANCH}"
if ! DIFF_BASE=$(git merge-base HEAD "${UPSTREAM_BRANCH}"); then
    echo "Unable to find merge base between HEAD and ${UPSTREAM_BRANCH}." >&2
    exit 1
fi

approval_line=$(curl -H "Authorization: token ${GITHUB_TOKEN}" "https://api.github.com/repos/PaddlePaddle/PaddleFormers/pulls/${PR_ID}/reviews?per_page=10000")
failed_num=1
echo_list=()


function check_approval(){
    APPROVALS=$(echo "${approval_line}" | python "${PADDLE_ROOT}/ci/check_pr_approval.py" "$@")
    if [[ "${APPROVALS}" == "FALSE" && "${echo_line}" != "" ]]; then
        add_failed "${failed_num}. ${echo_line}"
    fi
}


function add_failed(){
    failed_num=`expr $failed_num + 1`
    echo_list="${echo_list[@]}$1"
}


PADDLEFORMERS_TRAINER_APPROVERS="From00"
PADDLEFORMERS_TRAINER_FILES=(
    "paddleformers/trainer/training_args.py"
    "paddleformers/cli/hparams/"
)
for FILE in "${PADDLEFORMERS_TRAINER_FILES[@]}"; do
    HAS_MODIFIED=$(git diff --name-only "${DIFF_BASE}" HEAD -- | grep "^${FILE}" || true)
    if [ "${HAS_MODIFIED}" != "" ] && [ "${PR_ID}" != "" ]; then
        echo_line="You must be approved by From00 for changes in ${FILE}.\n"
        APPROVER_LIST=(${PADDLEFORMERS_TRAINER_APPROVERS})
        check_approval 1 "${APPROVER_LIST[@]}"
    fi
done


if [ -n "${echo_list}" ];then
  echo "****************"
  echo -e "${echo_list[@]}"
  echo "There are `expr $failed_num - 1` approved errors."
  echo "****************"
fi

if [ -n "${echo_list}" ]; then
  exit 6
fi
