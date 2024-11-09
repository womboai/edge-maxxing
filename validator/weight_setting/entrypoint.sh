#!/bin/bash

chown -R validator:validator /home/validator/.bittensor
chown -R validator:validator /home/validator/.netrc

while true; do
  sudo -u validator /home/validator/.local/bin/uv run start_validator "$@"
  exit_code=$?

  if [ $exit_code -eq 75 ]; then
    echo "Auto update initiated, restarting Validator"
    sleep 3
  fi
done