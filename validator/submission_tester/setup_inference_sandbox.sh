#!/bin/bash

set -e

SANDBOX_DIRECTORY=$1
REPOSITORY_URL=$2
REVISION=$3
BASELINE=$4
BLACKLISTED_DEPENDENCIES=$5

READY_MARKER="$SANDBOX_DIRECTORY/ready"
VENV="$SANDBOX_DIRECTORY/.venv"

cd "$SANDBOX_DIRECTORY"

if $($BASELINE) && [ -f "$READY_MARKER" ]; then
  git fetch
else
  find "$SANDBOX_DIRECTORY" -mindepth 1 -delete

  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --shallow-submodules "$REPOSITORY_URL" "$SANDBOX_DIRECTORY"
  if $($BASELINE); then
    touch "$READY_MARKER"
  fi
fi

FULL_REVISION=$(git rev-parse "$REVISION")
git fetch --depth 1 origin "$FULL_REVISION"
git checkout "$REVISION"

check_blacklist() {
  local file="$1"
  if [ -f "$file" ]; then
    while IFS= read -r line; do
      for dependency in $BLACKLISTED_DEPENDENCIES; do
        if [[ "$line" == *"$dependency"* ]]; then
          echo "Found blacklisted dependency '$dependency' in file '$file'"
          exit 2
        fi
      done
    done < "$file"
  fi
}

REQUIREMENTS="$SANDBOX_DIRECTORY/requirements.txt"
PYPROJECT="$SANDBOX_DIRECTORY/pyproject.toml"
PIPELINE="$SANDBOX_DIRECTORY/src/pipeline.py"

check_blacklist "$REQUIREMENTS"
check_blacklist "$PYPROJECT"
check_blacklist "$PIPELINE"

echo "Pulling LFS files..."
git lfs pull
git submodule update --init
echo "Pulled LFS files."

echo "Installing dependencies..."
if ! [ -f "$VENV" ]; then
  python3.10 -m venv "$VENV"
fi

if [ -f "$REQUIREMENTS" ]; then
  "$VENV/bin/pip" install -q -r "$REQUIREMENTS" -e "$SANDBOX_DIRECTORY"
else
  "$VENV/bin/pip" install -q -e "$SANDBOX_DIRECTORY"
fi
echo "Installed dependencies."
