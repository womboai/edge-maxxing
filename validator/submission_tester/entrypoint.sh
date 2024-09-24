#!/bin/bash

chown -R baseline-sandbox:baseline-sandbox /baseline-sandbox

exec sudo -E -u api /home/api/.local/bin/poetry run uvicorn --host 0.0.0.0 --port 8000 submission_tester:app