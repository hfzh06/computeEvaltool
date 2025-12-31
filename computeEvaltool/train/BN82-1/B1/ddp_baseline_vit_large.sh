#!/bin/bash
NODE_RANK=0

###############################################
# 2. 固定 DDP 配置（保持原参数）
###############################################
MASTER_ADDR="192.168.100.6"
MASTER_PORT=29577
NNODES=2
GPUS_PER_NODE=8

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=bond1
unset TORCH_DISTRIBUTED_DEBUG
export GLOO_SOCKET_IFNAME=bond1
#export NCCL_DEBUG=INFO

echo "启动 ViT-Large DDP 训练"

###############################################
# 4. 启动训练
###############################################
torchrun \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --nproc_per_node=${GPUS_PER_NODE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  /root/computeEvaltool/train/B1/vit-large/train_vit_cifar10_ddp.py \
  --batch-size 128 \
  --epochs 10
