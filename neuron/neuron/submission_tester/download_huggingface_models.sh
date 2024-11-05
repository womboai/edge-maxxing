#!/bin/bash

set -e

MODELS=$1

for model in $MODELS
do
    ~/.local/bin/huggingface-cli download "$model"
done