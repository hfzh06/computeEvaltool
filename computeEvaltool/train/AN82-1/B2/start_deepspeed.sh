#!/bin/bash
docker exec -it deepspeed1 /bin/bash -c "cd /root/computeEvaltool/train/B2"

export LOCAL_IP=10.1.73.17
export NODE_RANK=0
export MASTER_ADDR=10.1.73.17
export MASTER_PORT=29500
export NNODES=2
export NPROC_PER_NODE=8

echo "============================================================"
echo "LOCAL IP:${LOCAL_IP}  NODE_RANK ${NODE_RANK}"
echo "MASTER_ADDR: ${MASTER_ADDR}; MASTER_PORT: ${MASTER_PORT}; NNODES: ${NNODES}"
echo "============================================================"


docker start deepspeed1 > /dev/null 2>&1


docker exec -it \
  -e LOCAL_IP=$LOCAL_IP \
  -e NODE_RANK=$NODE_RANK \
  -e MASTER_ADDR=$MASTER_ADDR \
  -e MASTER_PORT=$MASTER_PORT \
  -e NNODES=$NNODES \
  -e NPROC_PER_NODE=$NPROC_PER_NODE \
  -w /root/computeEvaltool/train/B2 \
  deepspeed1 \
  /bin/bash
