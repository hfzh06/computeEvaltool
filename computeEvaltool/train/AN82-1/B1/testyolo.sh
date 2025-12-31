#!/bin/bash
# 激活环境
source /home/wtc/anaconda3/bin/activate pytorch

# ================= 配置区 =================
export MASTER_ADDR=10.1.73.17   # 主节点 IP
export MASTER_PORT=29500        # 通信端口
export NNODES=2                 # 总共多少台机器
export NPROC_PER_NODE=8         # 每台机器多少张卡
export NODE_RANK=0              # ★★★ 主节点 RANK 必须是 0

# NCCL 优化参数 (解决超时和网络检测问题)
export NCCL_IB_DISABLE=0                # 如果有 IB 网络则设为 0，没有则设为 1
export NCCL_SOCKET_IFNAME=enp46s0np0    # ★ 请确认这是你的真实网卡名称，用 ifconfig 查看
export NCCL_DEBUG=INFO                # 如果卡住，取消注释这一行查看调试信息

echo "============================================================"
echo "Master Node Launching..."
echo "Master Addr: ${MASTER_ADDR}"
echo "Node Rank:   ${NODE_RANK}"
echo "============================================================"


torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    testyolo.py   