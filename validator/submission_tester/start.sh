#!/bin/bash

set -e

./update.sh

sudo -u api /home/api/.local/bin/uv run opentelemetry-instrument \
  --service_name edge-maxxing-api \
  --exporter_otlp_endpoint http://98.81.78.238:4317 \
  --resource_attributes "neuron.type=validator,neuron.hotkey=$VALIDATOR_HOTKEY_SS58_ADDRESS" \
  uvicorn "$@"
