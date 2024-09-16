#!/bin/bash

set -e

VERSION=$(git rev-parse HEAD)

git pull --rebase --autostash

NEW_VERSION=$(git rev-parse HEAD)

if [ $VERSION != $NEW_VERSION ]; then
  docker compose down

  docker compose up -d --rebuild
fi
