#!/bin/bash
# ==============================
#  ViT-Large 极限性能测试脚本 (记录后3轮)
# ==============================

NODE_RANK=0
MASTER_ADDR="192.168.100.6"
MASTER_PORT=29577
NNODES=2
GPUS_PER_NODE=8

export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_DISABLE=1
unset TORCH_DISTRIBUTED_DEBUG
export GLOO_SOCKET_IFNAME=bond1

# 测试列表
BATCH_LIST=( 64 128 256)

# 确保 Epoch 数量大于 3，以便取最后三次
EPOCHS=4

# 结果保存路径
RESULT_PATH="/root/computeEvaltool/train/B1/results/training_vit-large_2.xlsx"
MODEL_DIR="/root/computeEvaltool/train/B1/vit-large/vit-large"
DATA_PATH="/root/computeEvaltool/train/B1/vit-large/cifar10"
echo "开始测试，结果将保存至: $RESULT_PATH"

for BS in ${BATCH_LIST[@]}; do
  echo ""
  echo "========================================"
  echo " Running Batch Size = $BS"
  echo "========================================"

  torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /root/computeEvaltool/train/B1/vit-large/train_vit_cifar10_ddp1.py \
      --epochs $EPOCHS \
      --batch-size $BS \
      --result-path $RESULT_PATH \
      --data-path $DATA_PATH 
done

echo ""
echo "所有 BatchSize 测试完成。"
