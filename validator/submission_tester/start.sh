#!/bin/bash

./submission_tester/update.sh
sudo -u api /home/api/.local/bin/uv run uvicorn "$@"
