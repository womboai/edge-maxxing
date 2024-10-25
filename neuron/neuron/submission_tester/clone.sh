#!/bin/bash

set -e

REPOSITORY_URL=$1
REVISION=$2

find . -mindepth 1 -delete
GIT_LFS_SKIP_SMUDGE=1 git clone --shallow-submodules "$REPOSITORY_URL" .
git checkout "$REVISION"
