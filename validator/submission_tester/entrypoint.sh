#!/bin/bash

chown -R sandbox:sandbox /home/sandbox/.cache/huggingface

./submission_tester/start.sh --host 0.0.0.0 --port 8000 submission_tester:app
