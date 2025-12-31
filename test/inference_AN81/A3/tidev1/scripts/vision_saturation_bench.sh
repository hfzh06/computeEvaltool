#!/usr/bin/env bash

set -e

mkdir -p results

workers=1
gpu_per_worker=8
things="10.1.18.121"

ts=$(date +"%Y%m%d%H%M%S")
file="results/result_saturation_$ts.json"

curl -fsSL "http://$things:8801/saturation-bench?workers=$workers&gpu_per_worker=$gpu_per_worker" | jq . | cat > "$file"

tsline=$(date -d "${ts:0:8} ${ts:8:2}:${ts:10:2}:${ts:12:2}" +"%Y-%m-%d %H:%M:%S")
echo "=== Saturation Benchmark Results ($tsline) ==="
echo "File: $file"
python3 scripts/format.py saturation "$file"

echo ""
echo "=== Saturation Points ==="
python3 scripts/format.py saturation_points "$file"
# cat "$file" | jq .
echo "Saved result to $file"