#!/bin/bash

echo "deepspeed deepseek7b训练启动"

export NCCL_SOCKET_IFNAME=enp46s0np0
export GLOO_SOCKET_IFNAME=enp46s0np0
export TP_SOCKET_IFNAME=enp46s0np0
export NCCL_P2P_LEVEL=NVL


deepspeed --launcher pdsh train.py --deepspeed ds_config.json
