#!/bin/bash
set -x

skip_kill_time=${1:-"False"}

function kill_impl() {
    skip_kill_time=$1
    if [[ $skip_kill_time == "True" ]];then
        for((i=1;i<=60;i++));
        do
            pids=`ps -ef | grep 'time_2023_8888.py' | grep -v grep | awk '{print $2}'`
            if [[ "$pids" == "" ]] ; then
                echo "no process found for speed-testing. stop waiting and kill other scripts."
                break
            fi
            echo "wait 10 seconds for finishing the speed-testing scripts."
            sleep 10s
        done
    fi

    # kill aadiff test finally.
    ps -ef | grep -E "check_aadiff.sh|run_aadiff_matmul.sh|test_matmul.py" | awk '{print $2}' | xargs kill -9

    pids=`ps -ef | grep train.py | grep -v grep | awk '{print $2}'`
    if [[ "$pids" != "" ]] ; then
        echo $pids
        echo $pids | xargs kill -9
    fi

    if [[ $TRAININGJOB_REPLICA_NAME == "trainer" ]]; then
        echo "Killing processes on gpu"
        lsof /dev/nvidia* | awk '{print $2}' | xargs -I {} kill -9 {}
    elif [[ $TRAININGJOB_REPLICA_NAME == "trainerxpu" ]]; then
        echo "Killing processes on xpu"
        lsof /dev/xpu* | awk '{print $2}' | xargs -I {} kill -9 {}
    else
        echo "[FATAL] unsupported training job type: ${TRAININGJOB_REPLICA_NAME}"
        exit 1
    fi
}

kill_impl $skip_kill_time || true