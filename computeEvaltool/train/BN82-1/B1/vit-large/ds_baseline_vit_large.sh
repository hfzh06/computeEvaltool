#!/bin/bash

###############################################
# 1. 节点信息 (主机配置)
###############################################
NODE_RANK=1      # 主机填 0
MASTER_ADDR="10.1.73.17"  # 主节点 IP (VM-18121)
MASTER_PORT=29503
NNODES=2
GPUS_PER_NODE=8

# 假设从机 NCCL 实际通信 IP 为 10.1.73.32
REMOTE_NCCL_IP="10.1.73.25" 

# NCCL 配置 (强制使用已知 IP 进行 P2P 通信)
export NCCL_SOCKET_IFNAME=enp46s0np0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
# 🌟 关键：手动指定两个节点的 NCCL 通信 IP 地址 (使用静态初始化) 

###############################################
# 2. 启动 torchrun 训练
###############################################
echo "=============================="
echo "启动 DeepSpeed ViT (Rank ${NODE_RANK})"
echo "NCCL_P2P_ADDRS: ${NCCL_P2P_NET_ADDRS}"
echo "=============================="

cd ~/vit-large

torchrun \
  --nnodes=${NNODES} \
  --nproc_per_node=${GPUS_PER_NODE} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train_vit_cifar10_deepspeed1.py \
  --batch-size 64 \
  --epochs 3 \
  --ds_config ds_config_vit.json \
  --data-path /mnt/ray_share/cifar10
