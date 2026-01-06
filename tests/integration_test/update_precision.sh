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

response=$(curl -L \
-H "Accept: application/vnd.github+json" \
-H "Authorization: Bearer ${GITHUB_TOKEN}" \
-H "X-GitHub-Api-Version: 2022-11-28" \
https://api.github.com/repos/PaddlePaddle/PaddleFleet/commits/${COMMIT_ID}/pulls)
pr_number=$(echo $response | jq -r '.[0].url' | awk -F'/' '{print $NF}')
repo_name=$(echo $GITHUB_REPO_NAME | awk -F'/' '{print $2}')
wget --no-proxy --no-check-certificate https://xly-devops.cdn.bcebos.com/PaddleFleet/precision/latest/precision_list.txt
pr_precision_url_base="https://paddle-github-action.cdn.bcebos.com/PaddleFleet/precision/${repo_name}_${pr_number}"
while read -r fname; do
    [ -z "$fname" ] && continue
    url="${pr_precision_url_base}/${fname}"
    echo "try update: $url"
    wget -q --no-proxy --no-check-certificate "$url" -O "$fname"
    if [ $? -ne 0 ]; then
        echo "No update is needed: $fname"
    else
        echo "Update required: $fname"
        python /workspace/bos/BosClient.py $fname xly-devops/PaddleFleet/precision/latest
    fi
    echo -e "\n"
done < precision_list.txt
