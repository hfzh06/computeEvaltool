#!/bin/bash
source /home/wtc/miniconda3/bin/activate pytorch
NODE_RANK=0
MASTER_ADDR="192.168.100.6"
MASTER_PORT=29566
NNODES=2
GPUS_PER_NODE=8

#export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_DISABLE=1
unset TORCH_DISTRIBUTED_DEBUG
export GLOO_SOCKET_IFNAME=bond1

echo "==============================================="
echo "      启动 ResNet18 极限性能测试      "
echo "==============================================="

# 待测试 batch size 列表
BATCH_LIST=(256 512 1024 2048)

# --- 修改处：进入正确的工作目录 ---
cd /root/computeEvaltool/train/B1/resnet18

for BS in ${BATCH_LIST[@]}; do
  echo ""
  echo "-----------------------------------------------"
  echo "   开始测试 Batch Size = ${BS}"
  echo "-----------------------------------------------"

  torchrun \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train_ddp_resnet18_s.py \
      --epochs 10 \
      --batch-size ${BS} \
      --data-root /data/cifar10  # 确保这里指向包含数据的子目录

done

echo ""
echo "==============================================="
echo "    ResNet18 极限性能测试完成！"
echo "    结果已保存至: /root/computeEvaltool/train/B1/results/training_resnet18_2.xlsx"
echo "==============================================="
