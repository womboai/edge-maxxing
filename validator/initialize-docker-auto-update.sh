#!/bin/bash

set -e

DIRECTORY=$(dirname $(realpath $0))

crontab -l > ./cron
echo "*/30 * * * * $DIRECTORY/update-docker.sh" >> ./cron
crontab ./cron
rm ./cron
