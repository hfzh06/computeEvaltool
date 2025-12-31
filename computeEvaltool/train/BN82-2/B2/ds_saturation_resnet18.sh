#!/bin/bash

MASTER_ADDR="192.168.100.6"
MASTER_PORT=29503
NNODES=2
GPUS_PER_NODE=8
unset TORCH_DISTRIBUTED_DEBUG
# 强制清理环境（防止上一次卡死的残留进程）
# pdsh -w node1,node2 "pkill -9 python; pkill -9 deepspeed" 

export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export GLOO_SOCKET_IFNAME=bond1
# 定义你要测试的 Batch Size 列表
BATCH_SIZES=(64 128 256 512 1024 2048 4096)

cd ./resnet18
# 清理旧结果（可选）
rm -f /root/computeEvaltool/train/B2/resnet18/training_resnet18_2.xlsx

for BS in "${BATCH_SIZES[@]}"
do
    echo "=================================================="
    echo "开始测试 Batch Size: $BS"
    echo "=================================================="

    deepspeed \
      train_resnet18_deepspeed1.py \
      --bs $BS
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "DeepSpeed run failed with exit code $EXIT_CODE"
        # 你可以在这里决定是继续还是停止
        # break 
    fi
    
    # 稍微休息一下，等待端口释放
    sleep 5
done
