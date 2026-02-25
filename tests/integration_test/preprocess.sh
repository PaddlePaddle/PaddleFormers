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

set -exo pipefail

export root_dir=$(pwd)

if [ ! -f $CACHE_DIR/glm45/data.tar ]; then
  mkdir -p $CACHE_DIR/glm45 && cd $CACHE_DIR/glm45
  wget -q --tries=5 --no-proxy https://xly-devops.cdn.bcebos.com/PaddleFleet/glm45_dataset/data.tar --no-check-certificate
  tar -xf data.tar
fi
if [ ! -f $CACHE_DIR/glm45/GLM-4.5-Air.tar ]; then
  mkdir -p $CACHE_DIR/glm45 && cd $CACHE_DIR/glm45
  wget -q --tries=5 --no-proxy https://xly-devops.cdn.bcebos.com/PaddleFleet/zai-org/GLM-4.5-Air.tar --no-check-certificate
  tar -xf GLM-4.5-Air.tar
fi
if [ ! -f $CACHE_DIR/qwen/Qwen3-30B-A3B-Base.tar ]; then
  mkdir -p $CACHE_DIR/qwen && cd $CACHE_DIR/qwen
  wget -q --tries=5 --no-proxy https://xly-devops.cdn.bcebos.com/PaddleFleet/Qwen/Qwen3-30B-A3B-Base.tar --no-check-certificate
  tar -xf Qwen3-30B-A3B-Base.tar
fi

cd $root_dir/PaddleFormers/examples/experiments/paddlefleet

apt-get update
apt-get install jq -y

sed -i '/if not int(os.getenv("test_ci_no_save_model", 0)):/s/^/# /' run_pretrain.py
sed -i '/trainer.save_model()/s/^/# /' run_pretrain.py
# sed -i 's/num_layers: int = 10/num_layers: int = 5/' glm45_provider.py


export FLAGS_use_stride_compute_kernel=False

python -c "
infile = '$root_dir/PaddleFormers/paddleformers/trainer/training_args.py'
outfile = infile + '.new'
with open(infile) as fin, open(outfile, 'w') as fout:
    for line in fin:
        if line.strip() == 'self.use_paddlefleet = False':
            pad = line[:len(line) - len(line.lstrip())]
            fout.write(pad + 'self.use_paddlefleet = True\n')
        else:
            fout.write(line)
"
mv $root_dir/PaddleFormers/paddleformers/trainer/training_args.py.new $root_dir/PaddleFormers/paddleformers/trainer/training_args.py
