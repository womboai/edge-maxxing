#!/bin/bash

set -e

~/.local/bin/uv run opentelemetry-instrument \
  --service_name edge-maxxing-validator \
  --exporter_otlp_endpoint http://98.81.78.238:4317 \
  --resource_attributes "neuron.type=validator,$(~/.local/bin/uv run python3 weight_setting/telemetry_attributes.py "$@")" \
  start_validator "$@"
