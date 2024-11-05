#!/bin/bash

PM2_PROCESS_NAME=$1

DIRECTORY=$(dirname $(realpath $0))

cd $DIRECTORY

VERSION=$(git rev-parse HEAD)

git pull --rebase --autostash

NEW_VERSION=$(git rev-parse HEAD)

if [ $VERSION != $NEW_VERSION ]; then
  pm2 restart $PM2_PROCESS_NAME
fi
