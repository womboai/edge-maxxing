#!/bin/bash

set -e

find . -type f -not -path '*/\.git/*' -not -path '*/.venv/*' -not -path '*/models/*' -print0 | while IFS= read -r -d '' file; do
  for dependency in "$@"; do
     if [[ "$file" == *"$dependency"* ]]; then
       echo "Found blacklisted dependency '$dependency' in filename '$file'"
       exit 1
     fi
  done

  while IFS= read -r line; do
    for dependency in "$@"; do
      if [[ "$line" == *"$dependency"* ]]; then
        echo "Found blacklisted dependency '$dependency' in file '$file'"
        exit 1
      fi
    done
  done < "$file"
done
