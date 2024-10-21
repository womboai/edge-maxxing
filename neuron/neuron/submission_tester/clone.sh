#!/bin/bash

set -e

REPOSITORY_URL=$1
REVISION=$2
ENABLE_CACHE=$3
IS_CACHED=$4

CACHE_FILE="cache_info.json"

if $ENABLE_CACHE && $IS_CACHED; then
  echo "Using cached repository"
else
  find . -mindepth 1 -delete

  GIT_LFS_SKIP_SMUDGE=1 git clone --shallow-submodules "$REPOSITORY_URL" .
  git checkout "$REVISION"
  if $ENABLE_CACHE; then
    echo "{\"repository\": \"$REPOSITORY_URL\", \"revision\": \"$REVISION\"}" > "$CACHE_FILE"
  fi
fi
