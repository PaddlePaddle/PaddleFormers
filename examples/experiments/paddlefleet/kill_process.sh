max_attempts=20
attempt=0

while [[ $attempt -lt $max_attempts ]]; do
    pids=$(ps -ef | grep -E 'train.py|trainer.py' | grep -v grep | awk '{print $2}')
    if [[ "$pids" != "" ]]; then
        echo "Attempt $((attempt + 1)): Killing processes: $pids"
        echo $pids | xargs kill -9
        sleep 1
        ((attempt++))
    else
        echo "No more processes to kill."
        break
    fi
done

if [[ $attempt -eq $max_attempts ]]; then
    echo "Error: Not all processes could be killed after $max_attempts attempts. Manual intervention required."
    echo "PADDLE_TRAINER_ID: $PADDLE_TRAINER_ID"
    exit 1
fi

attempt=0  # Reset attempt counter for GPU process killing
while [[ $attempt -lt $max_attempts ]]; do
    if [[ $TRAININGJOB_REPLICA_NAME == "trainer" ]]; then
        gpu_pids=$(lsof /dev/nvidia* | awk 'NR>1 {print $2}' | sort -u)
        if [[ "$gpu_pids" != "" ]]; then
            echo "Attempt $((attempt + 1)): Killing GPU processes: $gpu_pids"
            echo $gpu_pids | xargs kill -9
            sleep 1
            ((attempt++))
        else
            echo "No more GPU processes to kill."
            break
        fi
    elif [[ $TRAININGJOB_REPLICA_NAME == "trainerxpu" ]]; then
        gpu_pids=$(lsof /dev/xpu* | awk 'NR>1 {print $2}' | sort -u)
        if [[ "$gpu_pids" != "" ]]; then
            echo "Attempt $((attempt + 1)): Killing XPU processes: $gpu_pids"
            echo $gpu_pids | xargs kill -9
            sleep 10
            ((attempt++))
        else
            echo "No more XPU processes to kill."
            break
        fi
    else
        echo "[FATAL] unsupported training job type: ${TRAININGJOB_REPLICA_NAME}"
        exit 1
    fi
done

if [[ $attempt -eq $max_attempts ]]; then
    echo "Error: Not all GPU processes could be killed after $max_attempts attempts. Manual intervention required."
    echo "PADDLE_TRAINER_ID: $PADDLE_TRAINER_ID"
    exit 1
fi