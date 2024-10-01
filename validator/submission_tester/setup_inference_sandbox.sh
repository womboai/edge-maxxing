#!/bin/bash

set -e

SANDBOX_DIRECTORY=$1
GIT_PROVIDER=$2
REPOSITORY_URL=$3
REVISION=$4
BASELINE=$5

READY_MARKER="$SANDBOX_DIRECTORY/ready"
VENV="$SANDBOX_DIRECTORY/.venv"

cd "$SANDBOX_DIRECTORY"

if $($BASELINE) && [ -f "$READY_MARKER" ]; then
  git fetch
else
  find "$SANDBOX_DIRECTORY" -mindepth 1 -delete

  git clone --shallow-submodules --no-checkout "https://$GIT_PROVIDER/$REPOSITORY_URL" "$SANDBOX_DIRECTORY"
  if $($BASELINE); then
    touch "$READY_MARKER"
  fi
fi

git checkout "$REVISION"
git submodule update --init

if ! [ -f "$VENV" ]; then
  python3.10 -m venv "$VENV"
fi

REQUIREMENTS="$SANDBOX_DIRECTORY/requirements.txt"

if [ -f "$REQUIREMENTS" ]; then
  "$VENV/bin/pip" install -q -r "$REQUIREMENTS" -e "$SANDBOX_DIRECTORY"
else
  "$VENV/bin/pip" install -q -e "$SANDBOX_DIRECTORY"
fi
