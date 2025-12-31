#!/bin/bash
NODE_RANK=0
MASTER_ADDR="192.168.100.6"
MASTER_PORT=29500
NNODES=2
GPUS_PER_NODE=8

export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
unset TORCH_DISTRIBUTED_DEBUG
export GLOO_SOCKET_IFNAME=bond1
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
