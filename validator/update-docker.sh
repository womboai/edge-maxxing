#!/bin/bash

set -e

DIRECTORY=$(dirname $(realpath $0))

cd $DIRECTORY

VERSION=$(git rev-parse HEAD)

git pull --rebase --autostash

NEW_VERSION=$(git rev-parse HEAD)

if [ $VERSION != $NEW_VERSION ]; then
  docker compose down

  docker compose up -d --build
fi
