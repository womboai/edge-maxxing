#!/bin/bash

set -e

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get -y install sudo pipx git git-lfs build-essential python3-dev python3-opencv

sudo -u api pipx ensurepath
sudo -u api pipx install uv

sudo -u sandbox pipx ensurepath
sudo -u sandbox pipx install uv
sudo -u sandbox pipx install huggingface-hub[cli,hf_transfer]
sudo -u sandbox pipx upgrade-all
