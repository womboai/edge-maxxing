#!/bin/bash

set -e

useradd --shell=/bin/false --no-create-home sandbox
mkdir /sandbox
chown sandbox:sandbox /sandbox

useradd api

chown api:api .

apt-get update
apt-get -y install pipx

export PATH="$HOME/.local/bin:$PATH"

pipx install poetry

poetry install
