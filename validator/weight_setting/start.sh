#!/bin/bash

set -e

git config --local --add safe.directory '*'
git config --local rebase.autostash true
git config --local rebase.autosquash true
git config --local pull.autostash true

~/.local/bin/uv sync --no-dev --frozen --prerelease=allow

~/.local/bin/uv run --no-dev --frozen opentelemetry-instrument \
  --service_name edge-maxxing-validator \
  --exporter_otlp_endpoint http://98.81.78.238:4317 \
  --resource_attributes "neuron.type=validator,$(~/.local/bin/uv run python3 weight_setting/telemetry_attributes.py "$@")" \
  start_validator "$@"
