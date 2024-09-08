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
  rm -rf "$SANDBOX_DIRECTORY/*"

  git clone --recursive --no-checkout "$REPOSITORY_URL" "$SANDBOX_DIRECTORY"

  if $($BASELINE); then
    touch "$READY_MARKER"
  fi
fi

git checkout "$REVISION"
git submodule update --init

if ! [ -f "$VENV" ]; then
  python3.10 -m venv "$VENV"
fi

"$VENV/bin/pip" install $(cat "$SANDBOX_DIRECTORY/install_args.txt") -e $SANDBOX_DIRECTORY
