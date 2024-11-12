#!/bin/bash

set -e

useradd --shell=/bin/false --create-home --home-dir /home/sandbox sandbox || true

mkdir -p /sandbox

chown sandbox:sandbox /sandbox

useradd --create-home --home-dir /home/api api || true

chown -R api:api /api

./submission_tester/update.sh

su - api -c "cd /api/validator && uv sync"

echo "api ALL = (sandbox) NOPASSWD: ALL" >> /etc/sudoers
echo "Defaults env_keep += \"VALIDATOR_HOTKEY_SS58_ADDRESS VALIDATOR_DEBUG CUDA_VISIBLE_DEVICES\"" >> /etc/sudoers

git config --system advice.detachedHead false
