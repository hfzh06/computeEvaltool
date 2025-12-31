#!/usr/bin/env bash

set -e

if [ $(id -u) -ne 0 ]; then
    echo "Please run as root."
    exit 1
fi

things="9.0.2.60"

# if no other servers, just remove all items in the array
# but remain `servers=()` to avoid syntax error
servers=(
)

address="${things}:10000"

for server in "${servers[@]}"; do
    address="$address,${server}:10001"
done

echo "Starting tide server..."

sudo ./bin/tide serve -c 2 --port 10001 --role cloud --addresses "$address"
