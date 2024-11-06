#!/bin/bash

set -e

MODELS=$1

# TODO: remove once all validators update
pipx install --force huggingface-hub[cli,hf_transfer]

for model in $MODELS
do
    HF_HUB_ENABLE_HF_TRANSFER=1 ~/.local/bin/huggingface-cli download "$model"
done