#!/usr/bin/env bash
set -euo pipefail

MODEL="/data/models/deepseek-7b-chat"
HTTP_PORT=30000
DP_RPC_PORT=13345
GPU_UTIL=0.90
API_SERVER_COUNT=8

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1

# uv run vllm serve "${MODEL}" \
#   --enable-log-requests \
vllm serve "${MODEL}" \
  --port "${HTTP_PORT}" \
  --api-server-count "${API_SERVER_COUNT}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --tensor-parallel-size 8 \
  --load-format dummy \
  --max-model-len 4096