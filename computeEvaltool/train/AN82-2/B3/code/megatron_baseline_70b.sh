#!/bin/bash

##########################
# 环境变量配置
##########################
export MODELSCOPE_CACHE='/share'
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export MEGATRON_LM_PATH=/root/computeEvaltool/train/B3/Megatron-LM
export PYTHONPATH=$PYTHONPATH:$MEGATRON_LM_PATH

MASTER_ADDR="10.1.73.17"
MASTER_PORT=29500
NNODES=2
NODE_RANK=1
NPROC_PER_NODE=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

SAVE_DIR="/share/megatron_output/deepseek70B-lora"
RESULT_DIR="/root/computeEvaltool/train/B3/results"
OUTPUT_XLSX="${RESULT_DIR}/training_deepseek70B_1.xlsx"

mkdir -p ${RESULT_DIR}

##########################
# Step 1: 启动 Megatron-LM 训练
##########################

echo "===== 启动 Megatron-LM DeepSeek-70B LoRA 训练 ====="

NNODES=$NNODES \
NODE_RANK=$NODE_RANK \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
NPROC_PER_NODE=$NPROC_PER_NODE \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
megatron sft \
    --load megatron_deepseek70B \
    --dataset '/root/computeEvaltool/train/B3/datasets/openr1_math_220k.jsonl' \
    --train_type lora \
    --lora_rank 8 \
    --model_type qwen2 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 2 \
    --sequence_parallel true \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --train-iters 30 \
    --save ${SAVE_DIR} \
    --cross_entropy_loss_fusion true \
    --model_name deepseek70B-loraR

echo "===== Megatron 训练已结束 ====="

##########################
# Step 2: 自动定位最新输出目录
##########################

echo "===== 搜索最新训练输出目录 ====="

LATEST_DIR=$(ls -dt ${SAVE_DIR}/* | head -n 1)

if [ ! -d "$LATEST_DIR" ]; then
    echo "错误：未找到训练输出目录 ${LATEST_DIR}"
    exit 1
fi

echo "找到最新目录: $LATEST_DIR"

LOG_FILE="${LATEST_DIR}/logging.jsonl"

if [ ! -f "$LOG_FILE" ]; then
    echo "错误：未找到 logging.jsonl 文件"
    exit 1
fi

echo "logging.jsonl 路径为：$LOG_FILE"

##########################
# Step 3: 转换 JSONL 为 Excel
##########################

echo "===== 将 logging.jsonl 转换为 Excel ====="

python3 <<EOF
import json
import pandas as pd

log_file = "${LOG_FILE}"
output_file = "${OUTPUT_XLSX}"

rows = []
with open(log_file, "r") as f:
    for line in f:
        rows.append(json.loads(line.strip()))

df = pd.DataFrame(rows)
df.to_excel(output_file, index=False)

print("已成功写入 Excel：", output_file)
EOF

echo "===== 全流程完成 ====="
echo "Excel 文件已输出到：${OUTPUT_XLSX}"
