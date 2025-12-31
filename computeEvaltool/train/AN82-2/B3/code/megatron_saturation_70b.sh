#!/bin/bash

##############################################
#   主节点：NODE_RANK=0
#   从节点：NODE_RANK=1
##############################################
NODE_RANK=1    

##############################################
# 环境变量
##############################################
export MODELSCOPE_CACHE='/share'
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export MEGATRON_LM_PATH=/root/computeEvaltool/train/B3/Megatron-LM
export PYTHONPATH=$PYTHONPATH:$MEGATRON_LM_PATH

MASTER_ADDR="10.1.73.17"
MASTER_PORT=29500
NNODES=2
NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

SAVE_DIR="/share/megatron_output/deepseek70B-lora"
RESULT_DIR="/root/computeEvaltool/train/B3/results"
OUTPUT_XLSX="${RESULT_DIR}/training_deepseek70B_2.xlsx"
CONTROL_FILE="/share/current_bs_70B.txt"

mkdir -p "${RESULT_DIR}"
mkdir -p "${SAVE_DIR}"

##############################################
# sweep 参数
##############################################
MAX_GPU_MEM_GB=70
START_BS=1
STEP=1

##############################################
# 显存测试函数（仅在 NODE_RANK=0 用）
##############################################
test_batch_mem() {
python3 <<EOF
import torch
import sys

bs = int(sys.argv[1])

# DeepSeek-70B 的激活尺寸非常巨大，直接用较小尺寸估计即可
H = 4096
S = 2048

try:
    x = torch.randn(bs, S, H, device="cuda")
    y = torch.relu(x)
    mem = torch.cuda.max_memory_allocated()/1024/1024/1024
except Exception:
    mem = 9999

print(mem)
EOF
}

echo "===== 节点 NODE_RANK=${NODE_RANK} 启动 batch sweep ====="

##############################################
# 所有节点共同进入 sweep 循环
##############################################
BS=${START_BS}

while true; do

    #######################################################
    # 1) 节点 0 确定当前 batch size 或 STOP
    #######################################################
    if [ "${NODE_RANK}" -eq 0 ]; then
        rm -f "${CONTROL_FILE}"

        MEM=$(test_batch_mem "${BS}")
        MEM_FLOAT=$(printf "%.4f" "${MEM}")
        echo "[NODE 0] 测试 batch_size=${BS}, 估计显存=${MEM_FLOAT}GB"

        # 浮点比较（超过限制则停止）
        awk -v mem="${MEM_FLOAT}" -v max="${MAX_GPU_MEM_GB}" 'BEGIN { exit !(mem > max) }'
        if [ $? -eq 0 ]; then
            echo "STOP" > "${CONTROL_FILE}"
            echo "[NODE 0] 显存超限，写入 STOP"
            break
        fi

        echo "${BS}" > "${CONTROL_FILE}"
        echo "[NODE 0] 写入 micro_batch_size=${BS}"
    fi

    #######################################################
    # 2) 所有节点等待控制文件
    #######################################################
    while [ ! -f "${CONTROL_FILE}" ]; do
        sleep 1
    done

    CUR_VAL=$(cat "${CONTROL_FILE}")

    if [ "${CUR_VAL}" = "STOP" ]; then
        echo "[NODE ${NODE_RANK}] 收到 STOP，结束 sweep"
        break
    fi

    CUR_BS="${CUR_VAL}"
    echo "[NODE ${NODE_RANK}] 使用 batch_size=${CUR_BS} 进行训练"

    #######################################################
    # 3) 所有节点启动一次分布式 Megatron 训练
    #######################################################
    NNODES=${NNODES} \
    NODE_RANK=${NODE_RANK} \
    MASTER_ADDR=${MASTER_ADDR} \
    MASTER_PORT=${MASTER_PORT} \
    NPROC_PER_NODE=${NPROC_PER_NODE} \
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    megatron sft \
        --load megatron_deepseek70B \
        --dataset '/root/computeEvaltool/train/B3/datasets/openr1_math_220k.jsonl' \
        --train_type lora \
        --lora_rank 8 \
        --model_type llama \
        --lora_alpha 32 \
        --target_modules all-linear \
        --tensor_model_parallel_size 1 \
        --pipeline_model_parallel_size 2 \
        --sequence_parallel true \
        --micro_batch_size ${CUR_BS} \
        --global_batch_size $((CUR_BS * NNODES * NPROC_PER_NODE)) \
        --recompute_granularity full \
        --recompute_method uniform \
        --recompute_num_layers 1 \
        --finetune true \
        --train-iters 30 \
        --save "${SAVE_DIR}" \
        --cross_entropy_loss_fusion true \
        --model_name deepseek70B-loraR

    #######################################################
    # 4) 节点 0 解析日志并追加到 Excel
    #######################################################
    if [ "${NODE_RANK}" -eq 0 ]; then
        LATEST_DIR=$(ls -dt "${SAVE_DIR}"/* | head -n 1)
        LOG_FILE="${LATEST_DIR}/logging.jsonl"

        python3 <<EOF
import json, pandas as pd, os

log_file = "${LOG_FILE}"
output_file = "${OUTPUT_XLSX}"
bs = int("${CUR_BS}")

lines = open(log_file).read().strip().split("\n")[-5:]
records = []
for ln in lines:
    d = json.loads(ln)
    d["micro_batch_size"] = bs
    records.append(d)

df_new = pd.DataFrame(records)

if os.path.exists(output_file):
    df_old = pd.read_excel(output_file)
    df_all = pd.concat([df_old, df_new], ignore_index=True)
else:
    df_all = df_new

df_all.to_excel(output_file, index=False)
print(f"[PY] 已将 batch_size={bs} 的 5 条日志追加写入到 {output_file}")
EOF

        rm -f "${CONTROL_FILE}"
        BS=$((BS + STEP))
    fi

    sleep 2
done

echo "===== 节点 NODE_RANK=${NODE_RANK}：70B sweep 完成 ====="
