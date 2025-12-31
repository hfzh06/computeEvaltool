#!/bin/bash

MASTER_ADDR="192.168.100.6"
MASTER_PORT=29500

# 可以按需修改 batch size sweep 范围
BATCH_LIST=(128 256)

cd /root/computeEvaltool/train/B2/vit-large

echo "===== 启动 ViT-Large  DeepSpeed 极限性能测试训练 ====="

for BS in "${BATCH_LIST[@]}"; do
    echo ">>> 当前 Batch Size = $BS 每 GPU"

    deepspeed train_vit_cifar10_deepspeed1.py \
    --dynamic_bs $BS
done

echo "===== 测试完成！结果保存在 training_vit-large_2.xlsx ====="
