#!/bin/bash

set -e

~/.local/bin/uv run start_inference --frozen --no-sources --offline
