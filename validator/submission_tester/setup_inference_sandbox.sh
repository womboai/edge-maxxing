#!/bin/bash

set -e

SANDBOX_DIRECTORY=$1
REPOSITORY_URL=$2
REVISION=$3
BASELINE=$4

READY_MARKER="$SANDBOX_DIRECTORY/ready"
VENV="$SANDBOX_DIRECTORY/.venv"

cd "$SANDBOX_DIRECTORY"

if ! $($BASELINE) || ! [ -f "$READY_MARKER" ]; then
  rm -rf "$SANDBOX_DIRECTORY/*"

  git clone --recursive --no-checkout "$REPOSITORY_URL" "$SANDBOX_DIRECTORY"

  git checkout "$REVISION"
  git submodule update --init

  python3.10 -m venv "$VENV"

  "$VENV/bin/pip" install $(cat "$SANDBOX_DIRECTORY/install_args.txt") -e /sandbox

  if $($BASELINE); then
    touch "$READY_MARKER"
  fi
fi
