#!/usr/bin/env bash

set -e

things="9.0.2.60"

curl -fsSL http://$things:8801/health | jq .