#!/bin/bash

set -e

NETWORK_JAIL=$1

echo "LD_PRELOAD=${NETWORK_JAIL}" > jail.env
~/.local/bin/uv sync --frozen
~/.local/bin/uv run --frozen --offline --env-file jail.env start_inference
rm jail.env
