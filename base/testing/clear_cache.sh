#!/bin/bash

set -e

cd ~/.cache

find huggingface -mindepth 1 -delete || true
find pip -mindepth 1 -delete || true
find uv -mindepth 1 -delete || true
