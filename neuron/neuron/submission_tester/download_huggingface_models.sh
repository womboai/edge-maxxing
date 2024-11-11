#!/bin/bash

set -e

MODELS=$1

for model in $MODELS
do
    HF_HUB_ENABLE_HF_TRANSFER=1 ~/.local/bin/huggingface-cli download "$model"
done