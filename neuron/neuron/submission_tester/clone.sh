#!/bin/bash

set -e

REPOSITORY_URL=$1
REVISION=$2

find . -maxdepth 1 -not \( -name ".venv" -o -name "." \) -exec rm -rf {} +

git init
git remote add origin "$REPOSITORY_URL"
git fetch origin
git switch --force --detach "$REVISION"
