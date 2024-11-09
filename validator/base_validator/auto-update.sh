#!/bin/bash

set -e

UPDATE_PACKAGES=$1

echo "Checking for updates..."
VERSION=$(git rev-parse HEAD)

git pull --rebase --autostash origin builtin-auto-updating

NEW_VERSION=$(git rev-parse HEAD)

if [ "$VERSION" != "$NEW_VERSION" ]; then
  if [ "$UPDATE_PACKAGES" ]; then
    echo "Updating packages..."
    ./submission_tester/update.sh || true
  fi
  echo "New version detected: '$NEW_VERSION'. Restarting..."
  exit 75
fi
