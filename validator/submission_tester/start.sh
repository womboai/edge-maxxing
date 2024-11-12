#!/bin/bash

sudo -u api /home/api/.local/bin/uv run uvicorn "$@"
exit_code=$?

if [ $exit_code -eq 75 ]; then
  echo "Auto update initiated, updating dependencies and restarting API"
  ./submission_tester/update.sh
fi
