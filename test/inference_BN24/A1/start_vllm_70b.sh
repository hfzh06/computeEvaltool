#!/usr/bin/env bash
set -euo pipefail

MODEL="/data/models/deepseek-70b"
HTTP_PORT=30000
GPU_UTIL=0.90
API_SERVER_COUNT=2

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1

# --data-parallel-size 2
# --data-parallel-size-local 2

vllm serve "${MODEL}" \
  --port "${HTTP_PORT}" \
  --api-server-count "${API_SERVER_COUNT}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --tensor-parallel-size 2 \
  --max-model-len 2048 \
  --load-format dummy
