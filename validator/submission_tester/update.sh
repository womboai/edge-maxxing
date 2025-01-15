#!/bin/bash

set -e

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get -y install sudo pipx git git-lfs build-essential python3-dev python3-opencv

useradd --create-home --home-dir /home/api api || true
useradd --shell=/bin/false --create-home --home-dir /home/sandbox sandbox || true

chown -R api:api /api/pipelines
find /api/base -path "/api/base/.venv" -prune -o -exec chown api:api {} + || true
find /api/validator -path "/api/validator/.venv" -prune -o -exec chown api:api {} + || true

chown -R api:api /api/.git
chown api:api /api/validator/.venv || true
chown api:api /api/validator/uv.lock

pkill -9 -u sandbox || true
pkill -9 -u api || true
pkill -9 -f start_inference || true

rm -rf /sandbox
mkdir /sandbox
chown sandbox:sandbox /sandbox

mkdir -p /home/sandbox/.local
chown sandbox:sandbox /home/sandbox/.local
mkdir -p /home/sandbox/.cache
chown sandbox:sandbox /home/sandbox/.cache

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  unset CUDA_VISIBLE_DEVICES
fi

echo "root ALL=(ALL:ALL) ALL" > /etc/sudoers
echo "api ALL = (sandbox) NOPASSWD: ALL" >> /etc/sudoers
echo "Defaults env_keep += \"VALIDATOR_HOTKEY_SS58_ADDRESS VALIDATOR_DEBUG CUDA_VISIBLE_DEVICES\"" >> /etc/sudoers

git config --system advice.detachedHead false
git config --system --add safe.directory '*'
git config --system rebase.autostash true
git config --system rebase.autosquash true
git config --system pull.autostash true

sudo -u api pipx ensurepath
sudo -u api pipx install uv
sudo -u api /home/api/.local/bin/uv sync --no-dev --frozen

sudo -u sandbox pipx ensurepath
sudo -u sandbox pipx install uv
sudo -u sandbox pipx install huggingface-hub[cli,hf_transfer]
