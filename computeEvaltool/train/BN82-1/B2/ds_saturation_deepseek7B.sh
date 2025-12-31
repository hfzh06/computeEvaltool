#!/bin/bash

BATCH_LIST=(1 2 4 8 16)

CONFIG_DIR="/root/computeEvaltool/train/B2/deepseek/configs"
TRAIN_SCRIPT="/root/computeEvaltool/train/B2/deepseek/train.py"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
mkdir -p ${CONFIG_DIR}

for BS in ${BATCH_LIST[@]}
do
    echo ">>>> 生成 DeepSpeed 配置: batch size = $BS <<<<"

    CONFIG_FILE="${CONFIG_DIR}/ds_bs${BS}.json"

    # ==========================
    # 生成专用 DeepSpeed config
    # ==========================
    cat > ${CONFIG_FILE} <<EOF
{
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 50000000,
    "allgather_bucket_size": 50000000
  },

  "bf16": { "enabled": true },

  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-5,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },

  "train_micro_batch_size_per_gpu": ${BS},
  "gradient_accumulation_steps": 1,
  "gradient_clipping": 1.0,
  "wall_clock_breakdown": false
}
EOF

    echo ">>>> 启动训练 batch_size = $BS <<<<"

    deepspeed --launcher pdsh ${TRAIN_SCRIPT} \
        --deepspeed ${CONFIG_FILE} \
        --batch_size ${BS}

done
