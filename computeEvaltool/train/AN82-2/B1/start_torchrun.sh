#!/bin/bash
source /home/wtc/anaconda3/bin/activate yolo

export LOCAL_IP=10.1.73.17
export NODE_RANK=1              
export MASTER_ADDR=10.1.73.17   
export MASTER_PORT=29500
export NNODES=2                 
export NPROC_PER_NODE=8         
echo "============================================================"
echo "LOCAL IP:${LOCAL_IP}  NODE_RANK ${NODE_RANK}"
echo "MASTER_ADDR: ${MASTER_ADDR}; MASTER_PORT: ${MASTER_PORT}; NNODES: ${NNODES}"
echo "PyTorch DDP 分布式通信已成功启动"
echo "============================================================"

