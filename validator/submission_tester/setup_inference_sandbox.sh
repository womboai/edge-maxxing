#!/bin/bash

set -e

REPOSITORY_URL=$1
REVISION=$2
SANDBOX_DIRECTORY="/sandbox"

git clone --depth 1 --recursive --no-checkout "$REPOSITORY_URL" "$SANDBOX_DIRECTORY"

cd "$SANDBOX_DIRECTORY"
git checkout "$REVISION"

python3.10 -m venv /sandbox/.venv

/sandbox/.venv/bin/pip install /sandbox
