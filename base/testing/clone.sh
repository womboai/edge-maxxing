#!/bin/bash

set -e

REPOSITORY_URL=$1
REVISION=$2

find . -mindepth 1 -delete || true
git clone "$REPOSITORY_URL" .
git checkout "$REVISION"
