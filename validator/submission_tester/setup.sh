#!/bin/bash

set -e

useradd --shell=/bin/false --no-create-home sandbox
mkdir /sandbox
chown sandbox:sandbox /sandbox

useradd --create-home --home-dir /home/api api

chown api:api .
chown api:api /api/validator

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get -y install pipx git

su - api -c "
    pipx install poetry;
    pipx ensurepath
"

su - api -c "cd /api/validator && poetry install"