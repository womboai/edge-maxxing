#!/bin/bash

set -e

NETWORK_JAIL=$1
echo "LD_PRELOAD=${NETWORK_JAIL}" > .env
echo "HF_DATASETS_OFFLINE=1" >> .env
echo "HF_HUB_OFFLINE=1" >> .env
~/.local/bin/uv run --locked --offline --env-file .env start_inference
rm .env
