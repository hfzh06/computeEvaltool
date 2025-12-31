#!/usr/bin/env bash

set -e

go build -ldflags="-extldflags=-Wl,-z,lazy" -o bin/tide cmd/tide/main.go

if [ $? -eq 0 ]; then
    echo "Build successful. Binary is at bin/tide"
else
    echo "Build failed."
    exit 1
fi