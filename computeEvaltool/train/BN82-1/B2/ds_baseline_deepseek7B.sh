#!/bin/bash

echo "deepspeed deepseek7b训练启动"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
export TP_SOCKET_IFNAME=bond1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
deepspeed  /root/computeEvaltool/train/B2/deepseek/train1.py --deepspeed ds_config.json
