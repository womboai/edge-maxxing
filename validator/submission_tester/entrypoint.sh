#!/bin/bash

chown -R sandbox:sandbox /home/sandbox/.cache/huggingface

exec sudo -u api /home/api/.local/bin/uv run uvicorn --host 0.0.0.0 --port 8000 submission_tester:app