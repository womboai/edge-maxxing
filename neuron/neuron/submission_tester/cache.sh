#!/bin/bash

set -e

REPOSITORY_URL=$1
REVISION=$2

CACHE_FILE="cache_info.json"
echo "{\"repository\": \"$REPOSITORY_URL\", \"revision\": \"$REVISION\"}" > "$CACHE_FILE"