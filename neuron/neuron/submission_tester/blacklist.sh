#!/bin/bash

set -e

BLACKLISTED_DEPENDENCIES=$1

find . -type f -not -path '*/\.git/*' -print0 | while IFS= read -r -d '' file; do
  for dependency in $BLACKLISTED_DEPENDENCIES; do
     if [[ "$file" == *"$dependency"* ]]; then
       echo "Found blacklisted dependency '$dependency' in filename '$file'"
       exit 1
     fi
  done

  while IFS= read -r line; do
    for dependency in $BLACKLISTED_DEPENDENCIES; do
      if [[ "$line" == *"$dependency"* ]]; then
        echo "Found blacklisted dependency '$dependency' in file '$file'"
        exit 1
      fi
    done
  done < "$file"
done