#!/bin/bash

set -e

if ! ~/.local/bin/uv sync; then
    echo "First sync attempt failed. Clearing caches and retrying..."
    ~/.local/bin/uv cache clean
    ~/.local/bin/uv sync
fi