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

echo "Checking for blacklisted dependencies..."
find "$SANDBOX_DIRECTORY" -type f -not -path '*/\.git/*' -print0 | while IFS= read -r -d '' file; do
  for dependency in $BLACKLISTED_DEPENDENCIES; do
     if [[ "$file" == *"$dependency"* ]]; then
       echo "Found blacklisted dependency '$dependency' in filename '$file'"
       exit 2
     fi
  done

  while IFS= read -r line; do
    for dependency in $BLACKLISTED_DEPENDENCIES; do
      if [[ "$line" == *"$dependency"* ]]; then
        echo "Found blacklisted dependency '$dependency' in file '$file'"
        exit 2
      fi
    done
  done < "$file"
done
echo "No blacklisted dependencies found."

echo "Pulling LFS files..."
git lfs pull
git submodule update --init
echo "Pulled LFS files."

echo "Installing dependencies..."
if ! [ -f "$VENV" ]; then
  python3.10 -m venv "$VENV"
fi

REQUIREMENTS="$SANDBOX_DIRECTORY/requirements.txt"

if [ -f "$REQUIREMENTS" ]; then
  "$VENV/bin/pip" install -q -r "$REQUIREMENTS" -e "$SANDBOX_DIRECTORY"
else
  "$VENV/bin/pip" install -q -e "$SANDBOX_DIRECTORY"
fi
echo "Installed dependencies."
