#!/bin/bash


MASTER_PORT=29500
NNODES=2
GPUS_PER_NODE=8

export NCCL_SOCKET_IFNAME=enp46s0np0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO


echo "启动 ResNet18 deepspeed 训练"


cd ./resnet18

deepspeed \
  train_resnet18_deepspeed.py

