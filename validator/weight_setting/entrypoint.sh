#!/bin/bash

chown -R validator:validator /home/validator/.bittensor
chown -R validator:validator /home/validator/.netrc

exec sudo -u validator /home/validator/.local/bin/uv run start_validator "$@"