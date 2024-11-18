#!/bin/bash

set -e

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get -y install sudo pipx git git-lfs build-essential python3-dev python3-opencv

if ! id -u api &>/dev/null; then
  useradd --shell=/bin/false --create-home --home-dir /home/sandbox sandbox || true
  useradd --create-home --home-dir /home/api api || true
fi

chown -R api:api /api

pkill -u sandbox || true
pkill -u api || true
pkill -f start_inference || true

rm -rf /sandbox
mkdir /sandbox
chown sandbox:sandbox /sandbox

echo "root ALL=(ALL:ALL) ALL" > /etc/sudoers
echo "api ALL = (sandbox) NOPASSWD: ALL" >> /etc/sudoers
echo "Defaults env_keep += \"VALIDATOR_HOTKEY_SS58_ADDRESS VALIDATOR_DEBUG CUDA_VISIBLE_DEVICES\"" >> /etc/sudoers

git config --system --add safe.directory '*'
git config --system advice.detachedHead false
git config --system rebase.autostash true
git config --system rebase.autosquash true
git config --system pull.autostash true

sudo -u api pipx ensurepath
sudo -u api pipx install uv
sudo -u api /home/api/.local/bin/uv sync

sudo -u sandbox pipx ensurepath
sudo -u sandbox pipx install uv
sudo -u sandbox pipx install huggingface-hub[cli,hf_transfer]
sudo -u sandbox pipx upgrade-all
