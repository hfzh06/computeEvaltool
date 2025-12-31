#!/bin/bash

MASTER_ADDR="10.1.73.17"
MASTER_PORT=29500
NNODES=2
GPUS_PER_NODE=8


export NCCL_DEBUG=INFO


echo "启动 ViT-Large  DeepSpeed 基准性能测试训练"


cd /root/computeEvaltool/train/B2/vit-large

  deepspeed \
  --hostfile hostfile \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_vit_cifar10_deepspeed.py
