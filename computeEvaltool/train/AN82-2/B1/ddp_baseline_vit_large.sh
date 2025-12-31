#!/bin/bash
source /home/wtc/anaconda3/bin/activate pytorch
NODE_RANK=1

###############################################
# 2. 固定 DDP 配置（保持原参数）
###############################################
MASTER_ADDR="10.1.73.17"
MASTER_PORT=29577
NNODES=2
GPUS_PER_NODE=8

export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=enp46s0np0
#export NCCL_DEBUG=INFO

echo "启动 ViT-Large DDP 训练"

###############################################
# 4. 启动训练
###############################################
torchrun \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --nproc_per_node=${GPUS_PER_NODE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  /root/computeEvaltool/train/B1/vit-large/train_vit_cifar10_ddp.py \
  --batch-size 128 \
  --epochs 10
echo ""
echo "==============================================="
echo "    vit-large 基准性能测试完成！"
echo "    结果已保存至: /root/computeEvaltool/train/B1/results/training_vit-large_1.xlsx"
echo "==============================================="
