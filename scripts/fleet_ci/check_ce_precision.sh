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

case_name=$1
base_name=$2
case_gt_file=${base_name}_${case_name}_gt.txt
wget --no-proxy --no-check-certificate https://paddle-github-action.cdn.bcebos.com/PaddleFleet/ce/${case_gt_file} -O ${case_gt_file}
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "Failed to download ground truth file ${case_gt_file} for precision check."
    exit $exit_code
fi
if [[ ! -f "${case_name}.txt" && "$case_name" == *glm* ]]; then
    cp glm45_fleet/${case_name}.txt ./
fi
python  PaddleFormers/tests/integration_test/check_loss.py --log_file ${case_name}.txt --gt_file ${case_gt_file}
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "Failed to check precision for ${case_name}."
    exit $exit_code
else
    echo "Precision check passed for ${case_name}."
fi