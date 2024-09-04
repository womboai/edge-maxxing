#!/bin/bash

useradd --shell=/bin/false --no-create-home sandbox
mkdir /sandbox
chown sandbox:sandbox /sandbox

useradd api

chown api:api .
