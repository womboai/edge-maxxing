#!/bin/bash

set -e

REPOSITORY_URL=$1
REVISION=$2
CACHE=$3

CACHE_FILE="cache_info.json"

is_cached() {
  if [ ! -f "$CACHE_FILE" ]; then
      return 1
  fi

    cached_repository=$(jq -r '.repository' "$CACHE_FILE")
    cached_revision=$(jq -r '.revision' "$CACHE_FILE")

  [ "$cached_repository" = "$REPOSITORY_URL" ] && [ "$cached_revision" = "$REVISION" ]
}

if $CACHE && is_cached; then
  echo "Using cached repository"
else
  find . -mindepth 1 -delete

  GIT_LFS_SKIP_SMUDGE=1 git clone --shallow-submodules "$REPOSITORY_URL" .
  git checkout "$REVISION"
  if $CACHE; then
    echo "{\"repository\": \"$REPOSITORY_URL\", \"revision\": \"$REVISION\"}" > "$CACHE_FILE"
  fi
fi
