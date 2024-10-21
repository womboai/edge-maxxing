#!/bin/bash

set -e

REPOSITORY_URL=$1
REVISION=$2

CACHE_FILE="cache_info.json"

find . -mindepth 1 -delete
GIT_LFS_SKIP_SMUDGE=1 git clone --shallow-submodules "$REPOSITORY_URL" .
git checkout "$REVISION"
echo "{\"repository\": \"$REPOSITORY_URL\", \"revision\": \"$REVISION\"}" > "$CACHE_FILE"
