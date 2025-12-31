#!/usr/bin/env bash

set -e

echo "Waiting for tide server to run container ..."

interval=5
timeout=1200
elapsed=0
while true; do
    cnt=$(
        ls /tmp/tide/exec-logs/ | wc -l
    )
    elapsed=$((elapsed + interval))
    if [ $elapsed -ge $timeout ]; then
        echo "Timeout waiting for container to start."
        exit 1
    fi
    if [ "$cnt" -gt 0 ]; then
        echo "Container started."
        exit 0
    fi
    sleep $interval
done