#!/bin/bash

if [ "$EUID" -ne 0 ]; then
    echo "This script must be run as root"
    exit 1
fi

while true; do
  sudo -u api /home/api/.local/bin/uv run uvicorn "$@"
  exit_code=$?

  if [ $exit_code -eq 75 ]; then
    echo "Auto update initiated, updating dependencies and restarting API"
    ./submission_tester/update.sh
  fi
done