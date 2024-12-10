# Validators

- Requires two components: a scoring validator and at least one API container
    - Scoring validator:
        - Collects all miner submissions at 12 PM PST daily
        - Sends all submissions to be benchmarked on the API container
        - Sets weights based on the results of the contest
    - API container(s):
        - Receives submissions from the scoring validator
        - Benchmarks all submissions and gives a final score for each based on the contest's target metrics
        - Multiple API containers can be used to parallelize the benchmarking process and/or benchmark multiple contests
          simultaneously
        - Requires `Ubuntu 22.04`
        - Each API requires the specified hardware for the contest such as an `NVIDIA GeForce RTX 4090`

## Setup

- If you're running on dedicated hardware, i.e. you have the ability to run Docker, then follow
  the [Docker Compose](#docker-compose) instructions
- If you're running in a containerized environment such as RunPod, which doesn't support Docker, then follow
  the [RunPod/Containers](#runpodcontainers) instructions

### Docker Compose

1. Clone the EdgeMaxxing repository:
    ```bash
    git clone https://github.com/womboai/edge-maxxing
    cd edge-maxxing/validator
    ```
2. Create a `.env` file with the following variables:
    ```bash  
    echo "VALIDATOR_ARGS=--netuid 39 --subtensor.network finney --wallet.name {wallet} --wallet.hotkey {hotkey}" > .env
    echo "VALIDATOR_HOTKEY_SS58_ADDRESS={ss58-address}" >> .env
    ```
3. Modify `compose-gpu-layout.json` to include all CUDA device IDs you'd like to use:
    ```bash
    echo "[0]" > compose-gpu-layout.json
    ```
4. Generate the compose file for the GPUs you've specified:
    ```bash
    python3 ./generate_compose.py
    ```
5. Install and login to [wandb](https://docs.wandb.ai/quickstart/)
6. Start the container:
    ```bash
    docker compose up -d --build
    ```

### RunPod/Containers

##### API Component(s)

1. Clone the EdgeMaxxing repository into the `/api` directory:
    ```bash
    git clone https://github.com/womboai/edge-maxxing /api
    cd /api/validator
    ```
2. set the following environment variables:
    ```bash
    export CUDA_VISIBLE_DEVICES=0
    export VALIDATOR_HOTKEY_SS58_ADDRESS={ss58-address}
    ```
3. Install [PM2](https://pm2.io/docs/runtime/guide/installation/)
4. Start the API:
    ```bash
    pm2 start ./submission_tester/start.sh --name edge-maxxing-submission-tester --interpreter /bin/bash -- \
      --host 0.0.0.0 \
      --port 8000 \
      submission_tester:app
    ```
5. Ensure the port 8000 (or whatever you chose) is exposed.
6. If you'd like to run multiple API containers, repeat steps 1-5 for each pod/container.

#### Scoring Validator

1. Clone the EdgeMaxxing repository:
    ```bash
    git clone https://github.com/womboai/edge-maxxing
    cd edge-maxxing/validator
    ```
2. Install [pipx](https://pipx.pypa.io/stable/installation/)
3. Install `uv`
    ```bash
    pipx ensurepath
    pipx install uv
    ```
4. Install [PM2](https://pm2.io/docs/runtime/guide/installation/)
5. Install and login to [wandb](https://docs.wandb.ai/quickstart/)
5. Start the validator:
    ```bash
    pm2 start ./weight_setting/start.sh --name edge-maxxing-validator --interpreter /bin/bash -- \
        --netuid 39 \
        --subtensor.network finney \
        --wallet.name {wallet} \
        --wallet.hotkey {hotkey} \
        --benchmarker_api {API component routes, space separated if multiple}
    ```
    - Make sure to replace the API component route with the routes to the API containers(which can be something in the
      format of `http://ip:port`), refer to the instructions above at [API Component(s)](#api-components)
