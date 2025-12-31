#!/usr/bin/env bash

set -e

echo "Watching logs:"

tail -F /tmp/tide/exec-logs/*/stderr