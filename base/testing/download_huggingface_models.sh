#!/bin/bash

set -e

for model in "$@"
do
    HF_HUB_ENABLE_HF_TRANSFER=1 ~/.local/bin/huggingface-cli download "$model"
done