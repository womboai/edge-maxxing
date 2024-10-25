#!/bin/bash

set -e

cd ~/.cache

find huggingface -type f -delete || true
find pip -type f -delete || true
find pypoetry -type f -delete || true
find uv -type f -delete || true
find lfs-cache -type f -delete || true
