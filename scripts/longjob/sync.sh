#!/bin/bash
set -x
echo "lzx debug sync"
# if [[ -z "$PDC_YAML_PATH" ]]; then
#   # conf模式下需要
#   rm -f /root/paddlejob/workspace/env_run/longjob/train.conf
#   if [[ -z $PDC_TRAIN_CONF ]]; then
#     PDC_TRAIN_CONF=$1
#   fi
#   train_conf_path=`realpath $PDC_TRAIN_CONF`
#   ln -s $train_conf_path /root/paddlejob/workspace/env_run/longjob/train.conf
# fi

# source /root/paddlejob/workspace/env_run/longjob/train.conf

# tar_name="dbt-code.tar.zst"
# mpirun rm -f ${tar_name}
# # 压缩：tar + zstd（多线程处理 5.2G文件压缩3.8s）
# tar -I 'zstd -T0' --exclude 'longjob/train.conf' -cf ${tar_name} ./ernie5 ./ernie4 ./utils ./model_config ./conf ./script/ema ./script/ema_moe ./script/*.sh  ./script/*.py ./third_party/ ./longjob  $@
# python3 script/sync_new.py file ${tar_name}
# EXECUTE_EXCEPT_SELF="python script/execute_except_self.py"
# $EXECUTE_EXCEPT_SELF rm -rf third_party
# # 解压：tar + zstd（5.2文件压缩后解压4.8s）
# $EXECUTE_EXCEPT_SELF tar --zstd -xf ${tar_name}

# if [ "${enable_use_ema}" == "true" ]; then
#      nohup bash longjob/userfiles/start_ema.sh >>start_ema.log 2>&1 &
#      echo "start ema OK"
# fi
