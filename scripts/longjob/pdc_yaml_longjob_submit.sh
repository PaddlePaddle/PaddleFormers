#!/bin/bash
set -x

function parse_yaml2 {
  local file="$1"
  local key="$2"
  awk -F': *' -v k="$key" '$1 == k {sub(/^[^:]+:[ \t]*/, "", $0); gsub(/^"|"$/,"", $0); print $0}' "$file"
}

function pdc_submit_paddle_longjob() {
    SUBMIT_FILE=$1
    CONF_FILE=$2
    task_name=$3
    echo $1
    echo $2
    echo $3
    echo "---------------"


    # ------------- k8s gpu group on paddlecloud -----------------------------------#
    job_name=`parse_yaml2 ${SUBMIT_FILE} job_name`
    group_name=`parse_yaml2 ${SUBMIT_FILE} group_name`
    algo_id=`parse_yaml2 ${SUBMIT_FILE} algo_id`
    nnode=`parse_yaml2 ${SUBMIT_FILE} trainer_count`
    gpus=`parse_yaml2 ${SUBMIT_FILE} gpu_card`
    priority=`parse_yaml2 ${SUBMIT_FILE} priority`
    job_scene=`parse_yaml2 ${SUBMIT_FILE} job-scene`
    model_domain=`parse_yaml2 ${SUBMIT_FILE} model-domain`

    start_cmd=`parse_yaml2 ${SUBMIT_FILE} start_cmd`
    recovery_mode=`parse_yaml2 ${SUBMIT_FILE} recovery_mode`
    pdc_watch_log=`parse_yaml2 ${SUBMIT_FILE} pdc_watch_log`
    pdc_init_cmd=`parse_yaml2 ${SUBMIT_FILE} pdc_init_cmd`
    remote_checkpoint_dir=`parse_yaml2 ${SUBMIT_FILE} remote_checkpoint_dir`
    local_checkpoint_dir=`parse_yaml2 ${SUBMIT_FILE} local_checkpoint_dir`

    fs_name=`parse_yaml2 ${SUBMIT_FILE} fs_name`
    fs_ugi=`parse_yaml2 ${SUBMIT_FILE} fs_ugi`
    output_path=`parse_yaml2 ${SUBMIT_FILE} output_path`
    image_addr=`parse_yaml2  ${SUBMIT_FILE} image_addr`
    ak=`parse_yaml2  ${SUBMIT_FILE} ak`
    sk=`parse_yaml2  ${SUBMIT_FILE} sk`
    pdc_server=`parse_yaml2  ${SUBMIT_FILE} pdc_server`
    pdc_port=`parse_yaml2  ${SUBMIT_FILE} pdc_port`
    pdc_env_conf=$(parse_yaml2 ${SUBMIT_FILE} pdc_env_conf)

    # 新的配置文件
    new_config_file="./final_config.ini"
    temp_config_file="./temp_config.ini"

echo "
fs_name=${fs_name}
fs_ugi=${fs_ugi}
output_path=${output_path}
CORDON_IP_LIST=""
mpi_on_k8s=1
SYS_NCCL_CHECK=0
ENABLE_MOUNT_DEV_SHM=true
pdc_cordon_mode=on
is_output_auto_upload=0
is_fault_tolerant_on=0
pdc_watch_log=${pdc_watch_log}
pdc_init_cmd=${pdc_init_cmd}
remote_checkpoint_dir=${remote_checkpoint_dir}
local_checkpoint_dir=${local_checkpoint_dir}
SUBMIT_FILE=${SUBMIT_FILE}
" > ${temp_config_file}

    # 如果原配置文件存在，合并两个文件
    if [[ -n "$pdc_env_conf" && -f "$pdc_env_conf" ]]; then
        cat ${pdc_env_conf} ${temp_config_file} > ${new_config_file}
    else
        cp ${temp_config_file} ${new_config_file}
    fi

    if (( ${nnode} == 1 ));then
        sdl=1
    else
        sdl=0
    fi

    paddlecloud longjob --debug \
        --server ${pdc_server} \
        --port ${pdc_port} \
        --ak ${ak} \
        --sk ${sk} \
        train \
        --job-name ${job_name} \
        --job-version "paddle-fluid-custom" \
        --job-conf ${new_config_file} \
        --permission "group" \
        --group-name $group_name \
        --file-dir ./ \
        --k8s-priority ${priority} \
        --job-scene ${job_scene} \
        --model-domain ${model_domain} \
        --k8s-trainers ${nnode} \
        --k8s-gpu-cards ${gpus} \
        --image-addr ${image_addr} \
        --start-cmd "${start_cmd}" \
        --is-standalone $sdl \
        --enable-replace \
        --recovery-mode ${recovery_mode} \
        --algo-id ${algo_id} \
        --infoflow-group 9311534 \
        --infoflow-access-token dd0d46a2025aa1e663953ddf5bc6ef568 \
        --config-file ${CONF_FILE} \
        --task-name ${task_name}
}

pdc_submit_paddle_longjob $1 $2 $3