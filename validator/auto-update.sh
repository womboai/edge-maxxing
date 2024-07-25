#!/bin/bash

PM2_PROCESS_NAME=$1

while true; do
  sleep 1800

  VERSION=$(git rev-parse HEAD)

  git pull --rebase --autostash

  NEW_VERSION=$(git rev-parse HEAD)

  if [ $VERSION != $NEW_VERSION ]; then
    poetry install
    pm2 restart $PM2_PROCESS_NAME
  fi
done
