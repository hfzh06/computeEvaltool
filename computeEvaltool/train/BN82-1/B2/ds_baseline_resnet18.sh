#!/bin/bash

export MASTER_ADDR="192.168.100.6"
export MASTER_PORT=29500
export NNODES=2
export GPUS_PER_NODE=8
export 
export GLOO_SOCKET_IFNAME=bond1
export NCCL_SOCKET_IFNAME=bond1
unset TORCH_DISTRIBUTED_DEBUG
# 如果你的环境没有 InfiniBand (IB) 设备，建议显式禁用，防止自动探测出错
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO


echo "启动 ResNet18 deepspeed 训练"


cd ./resnet18

deepspeed train_resnet18_deepspeed.py 

