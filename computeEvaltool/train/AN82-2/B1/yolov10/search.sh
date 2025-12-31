#!/bin/bash

# 要测试的 batch size 列表
BATCH_LIST=(32 64 96 128 160 192 256 320 384 448 512)

for BS in "${BATCH_LIST[@]}"
do
    echo "===================================="
    echo "Running batch size = $BS"
    echo "===================================="

    BATCH_SIZE=$BS bash run.sh

    echo "Finished batch size = $BS"
    echo ""
done
