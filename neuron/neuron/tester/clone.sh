#!/bin/bash

set -e

REPOSITORY_URL=$1
REVISION=$2
BASELINE=$3

READY_MARKER="ready"

if $BASELINE && [ -f "$READY_MARKER" ]; then
  git fetch
else
  find . -mindepth 1 -delete

  GIT_LFS_SKIP_SMUDGE=1 git clone --shallow-submodules "$REPOSITORY_URL" .
  if $BASELINE; then
    touch "$READY_MARKER"
  fi
fi

git checkout "$REVISION"