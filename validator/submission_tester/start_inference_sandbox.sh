#!/bin/bash

set -e

python3.10 -m venv /sandbox/.venv

/sandbox/.venv/bin/pip install /sandbox

/sandbox/.venv/bin/start_inference
