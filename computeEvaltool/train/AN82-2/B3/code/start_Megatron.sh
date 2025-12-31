#!/bin/bash


LOCAL_IP="10.1.73.17"
MASTER_ADDR="10.1.73.17"
MASTER_PORT=29500
NNODES=2
NODE_RANK=1   
NPROC_PER_NODE=8


echo "LOCAL_IP:        ${LOCAL_IP}"
echo "NODE_RANK:       ${NODE_RANK}"
echo "MASTER_ADDR:     ${MASTER_ADDR}"
echo "MASTER_PORT:     ${MASTER_PORT}"
echo "NNODES:          ${NNODES}"
echo "NPROC_PER_NODE:  ${NPROC_PER_NODE}"
CONTAINER_NAME="deepseek"

docker start ${CONTAINER_NAME}


docker exec -it ${CONTAINER_NAME} bash -c "cd /root/computeEvaltool/train/B3 && exec bash"



