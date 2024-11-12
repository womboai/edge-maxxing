#!/bin/bash

set -e

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get -y install sudo pipx git git-lfs build-essential python3-dev python3-opencv

if ! id -u api &>/dev/null; then
  useradd --shell=/bin/false --create-home --home-dir /home/sandbox sandbox || true
  useradd --create-home --home-dir /home/api api || true
  chown -R api:api /api
fi

mkdir -p /sandbox
chown sandbox:sandbox /sandbox

echo "root ALL=(ALL:ALL) ALL" > /etc/sudoers
echo "api ALL = (sandbox) NOPASSWD: ALL" >> /etc/sudoers
echo "Defaults env_keep += \"VALIDATOR_HOTKEY_SS58_ADDRESS VALIDATOR_DEBUG CUDA_VISIBLE_DEVICES\"" >> /etc/sudoers

git config --system advice.detachedHead false

sudo -u api pipx ensurepath
sudo -u api pipx install uv
sudo -u api /home/api/.local/bin/uv sync

sudo -u sandbox pipx ensurepath
sudo -u sandbox pipx install uv
sudo -u sandbox pipx install huggingface-hub[cli,hf_transfer]
sudo -u sandbox pipx upgrade-all
