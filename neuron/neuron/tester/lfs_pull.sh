#!/bin/bash

set -e

git lfs pull
git submodule update --init