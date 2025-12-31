#!/usr/bin/env bash

set -e

RESULTS_DIR="./results"

# Find the latest baseline result
LATEST_BASELINE=$(ls $RESULTS_DIR/result_baseline_*.json 2>/dev/null | sort | tail -1)

if [ -z "$LATEST_BASELINE" ]; then
    echo "No baseline result files found."
else
    TS_BASELINE=$(basename "$LATEST_BASELINE" | sed 's/result_baseline_\(.*\)\.json/\1/')
    FORMATTED_TS_BASELINE=$(date -d "${TS_BASELINE:0:8} ${TS_BASELINE:8:2}:${TS_BASELINE:10:2}:${TS_BASELINE:12:2}" +"%Y-%m-%d %H:%M:%S")
    echo "=== Baseline Benchmark Results ($FORMATTED_TS_BASELINE) ==="
    echo "File: $LATEST_BASELINE"
    python3 scripts/format.py baseline "$LATEST_BASELINE"
fi

echo ""

# Find the latest saturation result
LATEST_SATURATION=$(ls $RESULTS_DIR/result_saturation_*.json 2>/dev/null | sort | tail -1)

if [ -z "$LATEST_SATURATION" ]; then
    echo "No saturation result files found."
else
    TS_SATURATION=$(basename "$LATEST_SATURATION" | sed 's/result_saturation_\(.*\)\.json/\1/')
    FORMATTED_TS_SATURATION=$(date -d "${TS_SATURATION:0:8} ${TS_SATURATION:8:2}:${TS_SATURATION:10:2}:${TS_SATURATION:12:2}" +"%Y-%m-%d %H:%M:%S")
    echo "=== Saturation Benchmark Results ($FORMATTED_TS_SATURATION) ==="
    echo "File: $LATEST_SATURATION"
    python3 scripts/format.py saturation "$LATEST_SATURATION"
    
    echo ""
    echo "=== Saturation Points ==="
    python3 scripts/format.py saturation_points "$LATEST_SATURATION"
fi
