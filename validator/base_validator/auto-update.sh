#!/bin/bash

set -e

echo "Checking for updates..."
VERSION=$(git rev-parse HEAD)

git pull --autostash origin main

NEW_VERSION=$(git rev-parse HEAD)

if [ "$VERSION" != "$NEW_VERSION" ]; then
  echo "New version detected: '$NEW_VERSION'. Restarting..."
  exit 75
fi
