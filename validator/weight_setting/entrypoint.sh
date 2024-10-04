#!/bin/bash

chown -R validator:validator /home/validator/.bittensor
chown -R validator:validator /home/validator/.netrc

exec sudo -u validator /home/validator/.local/bin/poetry run start_validator "$@"