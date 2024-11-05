#!/bin/bash

set -e

PM2_PROCESS_NAME=$1

DIRECTORY=$(dirname $(realpath $0))

crontab -l > ./cron
echo "*/15 * * * * $DIRECTORY/auto-update.sh $PM2_PROCESS_NAME" >> ./cron
crontab ./cron
rm ./cron
