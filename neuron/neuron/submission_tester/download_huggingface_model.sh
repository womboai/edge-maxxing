#!/bin/bash

set -e

MODEL=$1

~/.local/bin/huggingface-cli download $MODEL
