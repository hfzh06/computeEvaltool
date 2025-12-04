# A LightWeight Computility Evalution Framework
## Install Inference Framework

There are two famous inference frameworks [`vllm`](https://docs.vllm.ai/en/v0.10.1/index.html) and [`SGLang`](https://docs.sglang.ai/)


### Install `vllm` or `SGLang` 

### Using Python 

Firstly, create a new Python environment, we recommend to use uv, a very fast Python environment manager, to create and manage Python environments.

```bash
sudo apt update
apt install pip
pip install uv
```

Then,Install the vllm buy pre-built wheels

```bash
uv pip install vllm==0.10.1 --torch-backend=auto
uv pip install "sglang" --prerelease=allow
```

### Using Docker 

The simple way is to use pre-built images,
```bash
docker pull vllm/vllm-openai:v0.10.1
```

## Startup

### Run on A-N-8-1
On this machine, we have 8 NVIDIA A100(80G) linked with NVlink.

Run the `start.sh` 

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL="/path/to/models/deepseek-7b-chat"
HTTP_PORT=8000
GPU_UTIL=0.90
API_SERVER_COUNT=8

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=enp46s0np0 #your exact IFNAME
export GLOO_SOCKET_IFNAME=enp46s0np0
export NCCL_IB_DISABLE=1

vllm serve "${MODEL}" \
  --port "${HTTP_PORT}" \
  --api-server-count "${API_SERVER_COUNT}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --data-parallel-size 1 \
  --data-parallel-size-local 1 \
  --tensor-parallel-size 8
```

### Run on A-N-8-2
This cluster have two machine, which has 8 NVIDIA A100(80G) linked with NVlink.
#### Run the `start.sh` on each machine, use the same paramters
```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL="/path/to/models/deepseek-7b-chat"
HTTP_PORT=8000
GPU_UTIL=0.90
API_SERVER_COUNT=8

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=enp46s0np0 #your exact IFNAME
export GLOO_SOCKET_IFNAME=enp46s0np0
export NCCL_IB_DISABLE=1

vllm serve "${MODEL}" \
  --port "${HTTP_PORT}" \
  --api-server-count "${API_SERVER_COUNT}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --data-parallel-size 1 \
  --data-parallel-size-local 1 \
  --tensor-parallel-size 8
```

How does `vllm` start successfully, If you could see the logs like this
```bash
INFO 10-31 00:07:55 [api_server.py:1880] Starting vLLM API server 3 on http://0.0.0.0:30000
INFO 10-31 00:07:55 [launcher.py:29] Available routes are:
INFO 10-31 00:07:55 [launcher.py:37] Route: /openapi.json, Methods: HEAD, GET
INFO 10-31 00:07:55 [launcher.py:37] Route: /docs, Methods: HEAD, GET
INFO 10-31 00:07:55 [launcher.py:37] Route: /redoc, Methods: HEAD, GET
INFO 10-31 00:07:55 [launcher.py:37] Route: /health, Methods: GET
INFO 10-31 00:07:55 [launcher.py:37] Route: /load, Methods: GET
INFO 10-31 00:07:55 [launcher.py:37] Route: /ping, Methods: POST
INFO 10-31 00:07:55 [launcher.py:37] Route: /ping, Methods: GET
INFO 10-31 00:07:55 [launcher.py:37] Route: /tokenize, Methods: POST
INFO 10-31 00:07:55 [launcher.py:37] Route: /detokenize, Methods: POST
INFO 10-31 00:07:55 [launcher.py:37] Route: /v1/models, Methods: GET
```

#### Run the different `start.sh` on differnet machine

This method use `vllm` Data Parallel Function
- On `10.1.18.121`
```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL="/workspace/models/deepseek-7b-chat"
HTTP_PORT=8000
DP_RPC_PORT=13345
GPU_UTIL=0.90
API_SERVER_COUNT=8

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=enp46s0np0
export GLOO_SOCKET_IFNAME=enp46s0np0
export NCCL_IB_DISABLE=1

vllm serve "${MODEL}" \
  --port "${HTTP_PORT}" \
  --api-server-count "${API_SERVER_COUNT}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-address 10.1.18.121 \
  --data-parallel-rpc-port "${DP_RPC_PORT}" \
  --tensor-parallel-size 8
```
- Then on `10.1.18.122`

```bash

#!/usr/bin/env bash
set -euo pipefail

MODEL="/workspace/models/deepseek-7b-chat"
HTTP_PORT=8000
DP_RPC_PORT=13345
GPU_UTIL=0.90
API_SERVER_COUNT=8

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=enp46s0np0
export GLOO_SOCKET_IFNAME=enp46s0np0
export NCCL_IB_DISABLE=1

vllm serve "${MODEL}" \
  --port "${HTTP_PORT}" \
  --api-server-count "${API_SERVER_COUNT}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-address 10.1.18.121 \
  --data-parallel-rpc-port "${DP_RPC_PORT}" \
  --tensor-parallel-size 8
```

### Install Our ComputeEvaltool

#### Download source code

```bash
git clone https://git.cs.ac.cn/hfzh/computeevaltool
```

#### Then install our tool

```bash
source  vllmproj/.venv/bin./actiavte  
uv pip install -e .
uv pip install ".[llmeval]"
uv pip install ".[visioneval]"
```
#### Validate the Install process

```bash
computeEvaltool 
```
If you can see this, the computeEvaltoool installed successfully.
```
usage: computeEvaltool <command> [<args>]

positional arguments:
  {llmeval}      computeEvaltool command line helper.

options:
  -h, --help     show this help message and exit
  -v, --version  show program's version number and exit
```

### Run our Tool
```
MODEL_NAME="/data/models/deepseek-70b"
API_URL="http://0.0.0.0:18000/v1/chat/completions"
OUTPUT_DIR="./sglang-deepseek70b"
TASK_NAME="deepseek-7b-sglang"
PROMPT_LEN=32
MAX_TOKENS=128
SEED=42

echo "🚀 Running computeEvaltool test for $MODEL_NAME on $API_URL ..."

uv run computeEvaltool llmeval \
  --model "$MODEL_NAME" \
  --api openai \
  --url "$API_URL" \
  --number 50 500 500 1000 2000 2000 2000 3000 3000 3000 3000 3000 3000\
  --parallel 1 32 128 256 300 350 400 450 500 650 700 750 800\
  --max-prompt-length "$PROMPT_LEN" \
  --max-tokens "$MAX_TOKENS" \
  --seed "$SEED" \
  --name "$TASK_NAME" \
  --outputs-dir "$OUTPUT_DIR"
```
