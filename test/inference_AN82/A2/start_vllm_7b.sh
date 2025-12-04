#!/usr/bin/env bash
set -euo pipefail

MODEL="/data/models/deepseek-7b-chat"
HTTP_PORT=30000
DP_RPC_PORT=13345
GPU_UTIL=0.90
API_SERVER_COUNT=8

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_5,mlx5_6,mlx5_7,mlx5_8
export NCCL_IB_GID_INDEX=3,3,3,3,3,3,3,3
#export NCCL_SHM_DISABLE=0
#export NCCL_NVLAN_DISABLE=0
export NCCL_SOCKET_IFNAME=enp46s0np0
export GLOO_SOCKET_IFNAME=enp46s0np0
export NCCL_IB_DISABLE=0


# uv run python -m vllm.entrypoints.openai.api_server \
uv run vllm serve "${MODEL}" \
  --port "${HTTP_PORT}" \
  --api-server-count "${API_SERVER_COUNT}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-address 10.1.73.17\
  --data-parallel-rpc-port "${DP_RPC_PORT}" \
  --tensor-parallel-size 8 \
  --max-model-len 4096 
