#!/bin/env bash

if [ $(id -u) -ne 0 ]; then
    echo "Please run as root."
    exit 1
fi

echo "Cleaned up containers, images, logs, and message queues."

sudo ctr -n tide t ls | awk 'NR > 1 {print $1}' | xargs -r sudo ctr -n tide t rm -f
sudo ctr -n tide c ls | awk 'NR > 1 {print $1}' | xargs -r sudo ctr -n tide c rm
sudo rm -rf /tmp/tide/exec-logs/*
sudo rm logs/*
sudo rm -rf /dev/mqueue/*

sudo pkill tide

