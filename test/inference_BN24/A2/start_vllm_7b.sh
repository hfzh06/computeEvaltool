#!/usr/bin/env bash
set -euo pipefail

MODEL="/data/models/deepseek-7b-chat"
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
  --api-server-count "${API_SERVER_COUNT}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --data-parallel-size 8 \
  --data-parallel-size-local 2 \
  --data-parallel-address 9.0.2.60 \
  --data-parallel-rpc-port 13345 \
  --host 0.0.0.0 \
  --port 30000 \
  --load-format dummy