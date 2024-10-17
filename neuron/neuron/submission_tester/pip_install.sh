#!/bin/bash

set -e

VENV=".venv"
REQUIREMENTS="requirements.txt"

if ! [ -f "$VENV" ]; then
  python3.10 -m venv "$VENV"
fi

if [ -f "$REQUIREMENTS" ]; then
  "$VENV/bin/pip" install -q -r "$REQUIREMENTS" -e .
else
  "$VENV/bin/pip" install -q -e .
fi