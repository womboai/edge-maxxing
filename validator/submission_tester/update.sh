#!/bin/bash

set -e

sudo -u sandbox pipx ensurepath
sudo -u sandbox pipx install uv
sudo -u sandbox pipx install huggingface-hub[cli,hf_transfer]
sudo -u sandbox pipx upgrade-all
