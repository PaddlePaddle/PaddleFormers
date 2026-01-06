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

if [ -z ${BRANCH} ]; then
    BRANCH="develop"
fi

PADDLE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../../" && pwd )"

approval_line=`curl -H "Authorization: token ${GITHUB_TOKEN}" https://api.github.com/repos/PaddlePaddle/${repo_name}/pulls/${PR_ID}/reviews?per_page=10000`
# git_files=`git diff --numstat upstream/$BRANCH| wc -l`
# git_count=`git diff --numstat upstream/$BRANCH| awk '{sum+=$1}END{print sum}'`
failed_num=1
echo_list=()


function check_approval(){
    person_num=`echo $@|awk '{for (i=2;i<=NF;i++)print $i}'`
    APPROVALS=`echo ${approval_line}|python ${PADDLE_ROOT}/tests/integration_test/check_pr_approval.py $1 $person_num`
    if [[ "${APPROVALS}" == "FALSE" && "${echo_line}" != "" ]]; then
        add_failed "${failed_num}. ${echo_line}"
    fi
}


function add_failed(){
    failed_num=`expr $failed_num + 1`
    echo_list="${echo_list[@]}$1"
}

function run_tools_test() {
    CUR_PWD=$(pwd)
    cd ${PADDLE_ROOT}/tools
    python $1
    cd ${CUR_PWD}
}


PRECISION_APPROVERS="XieYunshen From00 risemeup1 tianlef zjjlivein"
echo_line="You must be approved by all of ${PRECISION_APPROVERS} for changing precision.\n"
APPROVER_LIST=(${PRECISION_APPROVERS})
NEED_APPROVALS=5
for user in "${APPROVER_LIST[@]}"; do
    if [[ "$user" == "$PR_USER" ]]; then
        NEED_APPROVALS=$((NEED_APPROVALS - 1))
        break
    fi
done
check_approval $NEED_APPROVALS "${APPROVER_LIST[@]}"


if [ -n "${echo_list}" ];then
  echo "****************"
  echo -e "${echo_list[@]}"
  echo "There are `expr $failed_num - 1` approved errors."
  echo "****************"
fi

if [ -n "${echo_list}" ]; then
  exit 6
fi
