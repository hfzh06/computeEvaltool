#!/usr/bin/env bash

set -e

echo "Watching logs:"

cnt=$(
    ls -lt /tmp/tide/exec-logs/ | wc -l
)

logdir=$(
    ls -lt /tmp/tide/exec-logs/ | awk "NR==$cnt {print \$9}"
)

tail -f /tmp/tide/exec-logs/"$logdir"/stdout