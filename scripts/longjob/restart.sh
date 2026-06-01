#!/bin/bash
set -x

# train.conf 可选，有则加载，没有也能跑
LONGJOB_CONF="scripts/longjob/train.conf"
if [[ -f "${LONGJOB_CONF}" ]]; then
    source ${LONGJOB_CONF}
else
    echo "No ${LONGJOB_CONF} found, using default settings."
fi

# 默认值（train.conf 不存在时生效）
local_checkpoint_dir="${local_checkpoint_dir:-output/checkpoints/glm45}"
pdc_sync_cmd="${pdc_sync_cmd:-}"

ERNIE_MPI_SH="scripts/longjob/mpi.sh"
TRAIN_SH="bash scripts/longjob/FT_test/run_glm45.sh"

if [[ -z "${PDC_FC_INIT_STEP}" ]]; then
    # 手动重跑, 非ZZC情况下需要清理FC的checkpoint
    bash scripts/longjob/utils/clean_flash_checkpoint.sh
fi

function restart() {
    done_save_step=`ls ${local_checkpoint_dir} 2>/dev/null | grep checkpoint- | sed 's/.$//' | grep -E '^checkpoint-[0-9]+$' | sed -n 's/[^0-9]*\([0-9]\+\).*/\1/p' | sort -n | tail -n 1`
    if [ -z "$done_save_step" ]; then
        done_save_step=0
        echo "No checkpoint found, starting from step 0"
    fi

    restart_from_step $done_save_step
}

function restart_from_step() {
    bash scripts/longjob/kill.sh
    done_save_step=$1
    mpirun python scripts/longjob/utils/deperacate_checkpoints.py ${local_checkpoint_dir} ${done_save_step}
    is_same_data=""
    if [ ! -z "$2" ] ;then
        if [ "$2" == "False" ] || [ "$2" == "false" ] ;then
            is_same_data="False"
        elif [ "$2" == "True" ] || [ "$2" == "true" ] ;then
            is_same_data="True"
        fi
    fi
    ckpt_path=${local_checkpoint_dir}/checkpoint-${done_save_step}
    echo $ckpt_path
    if [ $done_save_step -ne 0 ]; then
        mpirun /root/paddlejob/tools/agent -mode command -type download_checkpoint -config '{"download_step":'\"$done_save_step\"'}' || echo "agent download skipped (local env)"
    fi
    if [ -f watch_error_clean_data_seq ]; then
        echo "clean  ${local_checkpoint_dir}/data_seq*"
        mpirun rm -f "${local_checkpoint_dir}/data_seq*"
        touch watch_error_clean_data_seq.done
        rm watch_error_clean_data_seq
    fi
    if [ -n "$pdc_sync_cmd" ]; then
        eval "$pdc_sync_cmd"
    fi
    # 一键起飞
    CONF_FILE=$(./scripts/longjob/utils/get_yaml_path.sh)
    export PDC_INIT_STEP=${done_save_step}
    nohup sh ${ERNIE_MPI_SH} ${TRAIN_SH} &>restart.log &
    # 更新ckptwatcher的步数
    bash scripts/longjob/utils/reset_init_step.sh ${done_save_step}
    if [ -z $is_same_data ]; then
        is_same_data="auto" # for notice msg
    fi
    message="##[Status]:Starting\nQueue:$QUEUE\nJob-id:$PADDLE_JOB_NAME\nUpload_Dir:${remote_checkpoint_dir}\nInfo:从step_${done_save_step}重新启动中,samedata:${is_same_data}"
    python scripts/longjob/utils/create_msg_event.py error UserAction $message || true
}

function restart_from_fc_step() {
    bash scripts/longjob/kill.sh
    done_save_step=$1
    flash_device_path="/shared/dev/shm/flash"
    mpirun python scripts/longjob/utils/deperacate_checkpoints.py ${local_checkpoint_dir} ${done_save_step}
    is_same_data=""
    if [ ! -z "$2" ] ;then
        if [ "$2" == "False" ] || [ "$2" == "false" ] ;then
            is_same_data="False"
        elif [ "$2" == "True" ] || [ "$2" == "true" ] ;then
            is_same_data="True"
        fi
    fi
    ckpt_path=${flash_device_path}/checkpoint-${done_save_step}
    echo $ckpt_path
    if [ -n "$pdc_sync_cmd" ]; then
        eval "$pdc_sync_cmd"
    fi
    # 一键起飞
    export PDC_FC_INIT_STEP=${done_save_step}
    export PDC_INIT_STEP=${done_save_step}
    nohup sh ${ERNIE_MPI_SH} ${TRAIN_SH} &>restart.log &
    # 更新ckptwatcher的步数
    bash scripts/longjob/utils/reset_init_step.sh ${done_save_step}
    if [ -z $is_same_data ]; then
        is_same_data="auto" # for notice msg
    fi
    message="##[Status]:Starting-from-zcc \nQueue:$QUEUE\nJob-id:$PADDLE_JOB_NAME\nUpload_Dir:${remote_checkpoint_dir}\nInfo:从step_${done_save_step}重新启动中,samedata:${is_same_data}"
    python scripts/longjob/utils/create_msg_event.py error UserAction $message || true
}

if [ ! -z "${PDC_FC_INIT_STEP}" ]; then
    restart_from_fc_step ${PDC_FC_INIT_STEP}
elif [ -z "$1" ]; then
    restart
elif  [ ! -z "$1" ] && [ -z "$2" ]; then
    restart_from_step $1
elif [ ! -z "$1" ] && [ ! -z "$2" ]; then
    restart_from_step $1 $2
fi
