#!/bin/bash

NODE_RANK=1
NNODES=2
GPUS_PER_NODE=8
MASTER_ADDR="10.1.73.17"
MASTER_PORT=29612

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=enp46s0np0
export NCCL_DEBUG=INFO

echo "启动 YOLOv10-S DDP 训练，batch_size=${BATCH_SIZE}"

BATCH_SIZE=$BATCH_SIZE torchrun \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --nproc_per_node=${GPUS_PER_NODE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  /root/computeEvaltool/train/B1/yolov10/test_ddp.py
