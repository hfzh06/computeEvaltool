#!/bin/bash
# 激活环境 (注意路径是否和主节点一致，不一致请修改)
source /home/wtc/miniconda3/bin/activate pytorch

# ================= 配置区 =================
export MASTER_ADDR=10.1.73.17   # ★ 指向主节点 IP
export MASTER_PORT=29500        # 端口必须一致
export NNODES=2
export NPROC_PER_NODE=8
export NODE_RANK=1              # ★★★ 从节点 RANK 是 1 (如果是第三台机器则是 2)

# NCCL 参数
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=enp46s0np0    # ★ 请确认从节点的网卡名称，可能与主节点不同！

echo "============================================================"
echo "Slave Node Launching..."
echo "Connecting to Master: ${MASTER_ADDR}"
echo "Node Rank:            ${NODE_RANK}"
echo "============================================================"

# ★★★ 启动命令 ★★★
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    testyolo.py