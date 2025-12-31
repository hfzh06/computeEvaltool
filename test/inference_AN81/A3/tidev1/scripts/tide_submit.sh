#!/bin/env bash

mkdir -p /tmp/tide/exec-logs/

things="10.1.18.121"
image="docker.io/tide/visual-bench"

./bin/tide submit -i $image --server "$things:10000" -v '/root/cocodataset/val2017:/data/val2017,/root/test/inference_AN81/A3/tidev1/logs:/opt/app/logs' -c "python /opt/app/app.py"

echo "Submitted job with image $image"
echo "use ./scripts/watch[1|2].sh to watch the logs"
