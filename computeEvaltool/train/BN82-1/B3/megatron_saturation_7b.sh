#!/bin/bash

##############################################
#   主节点：NODE_RANK=0
#   从节点：NODE_RANK=1
##############################################
NODE_RANK=0    

##############################################
# 环境变量
##############################################
export MODELSCOPE_CACHE='/share'
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export MEGATRON_LM_PATH='/root/Megatron-LM'
export PYTHONPATH=$PYTHONPATH:$MEGATRON_LM_PATH

export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

MASTER_ADDR="192.168.100.6"
MASTER_PORT=29500
NNODES=2
NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

SAVE_DIR="/share/megatron_output/deepseek-ocr-lora"
RESULT_DIR="/root/computeEvaltool/train/B3/results"
OUTPUT_XLSX="${RESULT_DIR}/training_deepseek7B_2.xlsx"
CONTROL_FILE="/share/current_bs.txt"

mkdir -p "${RESULT_DIR}"
mkdir -p "${SAVE_DIR}"

##############################################
# sweep 参数
##############################################
MAX_GPU_MEM_GB=35   # 80G 显卡，预留一点
START_BS=1
STEP=3

##############################################
# 显存测试函数（仅在 NODE_RANK=0 用）
##############################################
test_batch_mem() {
python3 <<EOF
import torch
import sys

bs = int(sys.argv[1])

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

echo "===== 节点 NODE_RANK=${NODE_RANK} 启动，准备进入 batch sweep 循环 ====="

##############################################
# 所有节点共同进入外层循环
##############################################
BS=${START_BS}

while true; do
    #######################################################
    # 1) 只有 NODE_RANK=0 负责选择当前 BS 或决定 STOP
    #######################################################
    if [ "${NODE_RANK}" -eq 1 ]; then
        # 确保控制文件是新的
        rm -f "${CONTROL_FILE}"

        MEM=$(test_batch_mem "${BS}")
        MEM_FLOAT=$(printf "%.4f" "${MEM}")
        echo "[NODE 0] 测试 batch_size=${BS}, 估计显存=${MEM_FLOAT} GB"

        # 如果显存超出上限，则写 STOP 并退出循环
        awk -v mem="${MEM_FLOAT}" -v max="${MAX_GPU_MEM_GB}" 'BEGIN { exit !(mem > max) }'
        if [ $? -eq 0 ]; then
            echo "STOP" > "${CONTROL_FILE}"
            echo "[NODE 0] 显存超出限制（${MEM_FLOAT} GB > ${MAX_GPU_MEM_GB} GB），写入 STOP，结束 sweep"
            break
        fi

        # 将当前 BS 写入控制文件，通知其他节点
        echo "${BS}" > "${CONTROL_FILE}"
        echo "[NODE 0] 确定本轮 batch_size=${BS}，已写入 ${CONTROL_FILE}"
    fi

    #######################################################
    # 2) 所有节点等待控制文件出现
    #######################################################
    while [ ! -f "${CONTROL_FILE}" ]; do
        sleep 1
    done

    CUR_VAL=$(cat "${CONTROL_FILE}")

    # 如果是 STOP，所有节点都退出
    if [ "${CUR_VAL}" = "STOP" ]; then
        echo "[NODE ${NODE_RANK}] 检测到 STOP 标记，退出 sweep 循环"
        break
    fi

    CUR_BS="${CUR_VAL}"
    echo "[NODE ${NODE_RANK}] 本轮使用 micro_batch_size=${CUR_BS} 进入训练"

    #######################################################
    # 3) 所有节点启动一次训练（相同 BS）
    #    这里用 megatron sft（你之前验证可用）
    #######################################################
    NNODES=${NNODES} \
    NODE_RANK=${NODE_RANK} \
    MASTER_ADDR=${MASTER_ADDR} \
    MASTER_PORT=${MASTER_PORT} \
    NPROC_PER_NODE=${NPROC_PER_NODE} \
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    megatron sft \
        --load /data/megatron_deepseek7B \
        --dataset '/data/datasets/openr1_math_220k.jsonl#300' \
        --train_type lora \
        --lora_rank 8 \
        --model_type qwen2 \
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
        --model_name deepseek-ocr-loraR

    #######################################################
    # 4) 只有 NODE_RANK=0 做日志解析并追加到 Excel
    #######################################################
    if [ "${NODE_RANK}" -eq 1 ]; then
        LATEST_DIR=$(ls -dt "${SAVE_DIR}"/* | head -n 1)
        LOG_FILE="${LATEST_DIR}/logging.jsonl"

        echo "[NODE 0] 本轮训练输出目录: ${LATEST_DIR}"
        echo "[NODE 0] 解析日志文件: ${LOG_FILE}"

        python3 <<EOF
import json, pandas as pd, os

log_file = "${LOG_FILE}"
output_file = "${OUTPUT_XLSX}"
bs = int("${CUR_BS}")

with open(log_file, "r") as f:
    lines = f.read().strip().split("\n")

lines = lines[-5:]  # 取最后 5 行
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
print(f"[PY] 已将 batch_size={bs} 的 5 条日志追加到 {output_file}")
EOF

        # 为下一轮做准备：删除控制文件，BS 自增
        rm -f "${CONTROL_FILE}"
        BS=$((BS + STEP))
    fi

    # 所有节点等待 NODE_RANK=0 完成本轮收尾（避免太抢跑）
    sleep 2
done

echo "===== 节点 NODE_RANK=${NODE_RANK}：所有 batch sweep 结束 ====="
