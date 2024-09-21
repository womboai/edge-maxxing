#!/bin/bash

set -e

SANDBOX_DIRECTORY=$1
REPOSITORY_URL=$2
REVISION=$3
BASELINE=$4

READY_MARKER="$SANDBOX_DIRECTORY/ready"
VENV="$SANDBOX_DIRECTORY/.venv"

cd "$SANDBOX_DIRECTORY"

if $($BASELINE) && [ -f "$READY_MARKER" ]; then
  git fetch
else
  find "$SANDBOX_DIRECTORY" -mindepth 1 -delete

  git config --global advice.detachedHead false
  git clone --shallow-submodules --no-checkout "$REPOSITORY_URL" "$SANDBOX_DIRECTORY"
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
