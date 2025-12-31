#!/usr/bin/env bash

set -e

if [ $(id -u) -ne 0 ]; then
    echo "Please run as root."
    exit 1
fi

sudo ./bin/tide serve --port 10000 --role things