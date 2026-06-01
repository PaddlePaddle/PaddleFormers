#!/bin/bash
export master=`head -1 ${TRAIN_WORKSPACE}/hostfile |awk '{print $1}'`

if [[ ${IS_STANDALONE:-1} -eq 0 ]]; then
    mpirun \
        --allow-run-as-root \
        -tag-output -timestamp-output \
        -mca btl_tcp_if_exclude docker0,lo,matrixdummy0,matrix0 \
        -pernode \
        --bind-to none \
        -x iplist=${TRAINER_IP_LIST} \
        -x PATH \
        -x LD_LIBRARY_PATH \
        -x NCCL_DEBUG=INFO  \
        -x NCCL_ERROR_FILE=/root/paddlejob/workspace/log/err.nccl.%p.log \
        -x script \
        -x pt_args \
        -x PYTHONPATH \
        -x PAIMON_CONFIG \
        -x restore_ckpt \
        -x RANDOM_PORT=100 \
        -x gpus=8 \
        -x master=${master} \
        -x restore_state \
        -x copyrun_root_dir \
        -x expr_name \
        -x outdir \
        $@
    echo 'mpirun finished at' `date '+%Y-%m-%d %T'`
else
    $@
fi
