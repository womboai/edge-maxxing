#!/bin/bash

set -e

useradd --shell=/bin/false --create-home --home-dir /home/sandbox sandbox || true
useradd --shell=/bin/false --create-home --home-dir /home/baseline-sandbox baseline-sandbox || true

mkdir /sandbox || true
mkdir /baseline-sandbox || true

chown sandbox:sandbox /sandbox
chown baseline-sandbox:baseline-sandbox /baseline-sandbox

useradd --create-home --home-dir /home/api api || true

chown -R api:api /api

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get -y install sudo pipx git git-lfs build-essential python3-dev python3-opencv

su - api -c "
    pipx install poetry
    pipx ensurepath
"

su - api -c "cd /api/validator && poetry install"

echo "api ALL = (sandbox) NOPASSWD: ALL" >> /etc/sudoers
echo "api ALL = (baseline-sandbox) NOPASSWD: ALL" >> /etc/sudoers
echo "Defaults env_keep += \"VALIDATOR_HOTKEY_SS58_ADDRESS VALIDATOR_DEBUG\"" >> /etc/sudoers

git config --system lfs.concurrenttransfers 64
git config --system advice.detachedHead false
git config --global init.defaultBranch main

sudo -u baseline-sandbox git lfs install
sudo -u sandbox git lfs install

CACHE_DIR="/home/sandbox/.cache/lfs-cache"
sudo -u sandbox mkdir -p "$CACHE_DIR"
sudo -u sandbox git config --global lfs.storage "$CACHE_DIR"