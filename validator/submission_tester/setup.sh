#!/bin/bash

set -e

useradd --shell=/bin/false --create-home --home-dir /home/sandbox sandbox || true

mkdir -p /sandbox

chown sandbox:sandbox /sandbox

useradd --create-home --home-dir /home/api api || true

chown -R api:api /api

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get -y install sudo pipx git git-lfs build-essential python3-dev python3-opencv

sudo -u api pipx install poetry
sudo -u api pipx ensurepath

sudo -u sandbox pipx install uv
sudo -u sandbox pipx install huggingface-hub[cli]
sudo -u sandbox pipx ensurepath

su - api -c "cd /api/validator && poetry install"

echo "api ALL = (sandbox) NOPASSWD: ALL" >> /etc/sudoers
echo "Defaults env_keep += \"VALIDATOR_HOTKEY_SS58_ADDRESS VALIDATOR_DEBUG\"" >> /etc/sudoers

git config --system advice.detachedHead false
