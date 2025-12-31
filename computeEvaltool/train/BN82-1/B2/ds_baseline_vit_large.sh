#!/bin/bash

export MASTER_ADDR="192.168.100.6"
export MASTER_PORT=29500
export NNODES=1
export GPUS_PER_NODE=8
export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
unset TORCH_DISTRIBUTED_DEBUG
export NCCL_DEBUG=INFO


echo "启动 ViT-Large  DeepSpeed 基准性能测试训练"


cd /root/computeEvaltool/train/B2/vit-large

deepspeed train_vit_cifar10_deepspeed.py
