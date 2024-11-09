#!/bin/bash

chown -R sandbox:sandbox /home/sandbox/.cache/huggingface

while true; do
  sudo -u api /home/api/.local/bin/uv run uvicorn --host 0.0.0.0 --port 8000 submission_tester:app
  exit_code=$?

  if [ $exit_code -eq 75 ]; then
    echo "Auto update initiated, restarting API"
    sleep 3
  fi
done