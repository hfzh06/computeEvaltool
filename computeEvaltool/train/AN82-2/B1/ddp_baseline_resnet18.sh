#!/bin/bash
source /home/wtc/anaconda3/bin/activate pytorch
NODE_RANK=1
MASTER_ADDR="10.1.73.17"
MASTER_PORT=29503
NNODES=2
GPUS_PER_NODE=8

export NCCL_SOCKET_IFNAME=enp46s0np0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO

echo "启动 ResNet18 DDP 训练"



torchrun \
  --nnodes=${NNODES} \
  --node-rank=${NODE_RANK} \
  --nproc_per_node=${GPUS_PER_NODE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  /root/computeEvaltool/train/B1/resnet18/train_ddp_resnet18.py \
  --epochs 30 \
  --batch-size 128 \
  --data-root /root/computeEvaltool/train/B1/resnet18/cifar10 \
  --save-dir ./checkpoints
