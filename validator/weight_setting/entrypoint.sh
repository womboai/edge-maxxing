#!/bin/bash

chown -R validator:validator /home/validator/.bittensor
chown -R validator:validator /home/validator/.netrc

sudo -u validator ./weight_setting/start.sh "$@"
