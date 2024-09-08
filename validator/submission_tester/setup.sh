#!/bin/bash

set -e

useradd --shell=/bin/false --create-home --home-dir /home/sandbox sandbox
useradd --shell=/bin/false --create-home --home-dir /home/baseline-sandbox baseline-sandbox

mkdir /sandbox
mkdir /baseline-sandbox

chown sandbox:sandbox /sandbox
chown baseline-sandbox:baseline-sandbox /baseline-sandbox

useradd --create-home --home-dir /home/api api

chown -R api:api /api

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get -y install sudo pipx git git-lfs

su - api -c "
    pipx install poetry;
    pipx ensurepath
"

su - api -c "cd /api/validator && poetry install"

echo "api ALL = (sandbox) NOPASSWD: ALL" >> /etc/sudoers
echo "api ALL = (baseline-sandbox) NOPASSWD: ALL" >> /etc/sudoers
