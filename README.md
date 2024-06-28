# edge-optimization-subnet

### Current GPU: NVIDIA GeForce RTX 4090
### Current Model: `stablediffusionapi/newdream-sdxl-20`

## Miners and Validators

To start working with a registered hotkey, clone the repository and install poetry
```bash
# Poetry
if [ "$USER" = "root" ]; then
  apt install pipx
else
  sudo apt install pipx
fi

pipx ensurepath
pipx install poetry

# Repository
git clone https://github.com/womboai/edge-optimization-subnet
cd edge-optimization-subnet
```

There is no need to manage venvs in any way, as poetry will handle that.

### Miner setup
1. Go to the miner directory after cloning the repository,
    ```bash
    cd miner
    ```
2. Clone the base model into a directory `model`
    ```bash
    git clone https://huggingface.co/stablediffusionapi/newdream-sdxl-20 model
    ```
3. Make your own repository on huggingface to optimize in
4. Edit the miner/miner.py file, specifically the [optimize](https://github.com/womboai/edge-optimization-subnet/blob/main/miner/miner/miner.py#L20) function
5. If you have methods of optimization that you do not wish to do through the aforementioned python function, optimize directly in the `model` directory
6. Submit the model, changing the options as necessary
    ```bash
    poetry run python miner/miner.py \
        --repository {huggingface-repository} \
        --netuid {netuid} \
        --subtensor.network finney \
        --wallet.name {wallet} \
        --wallet.hotkey {hotkey} \
        --logging.trace \
        --logging.debug
    ```
    Add `--no_optimizations` if you have not changed the `optimize` function, and `no_commit` if the optimizations are already in the repository, and you just want to make the submission.<br>
    Additionally, you can pass `--commit_message` to add a commit message to the commit made if `no_commit` is not passed.
7. Validators will collect your submission on 12PM New York time and tested in the remainder of the day

### Validator setup
All that is needed for a validator is running on the current contest's GPU with a registered hotkey;<br>
This assumes using PM2, feel free to adjust for anything else
```bash
cd validator

pm2 start poetry --name edge-validator --interpreter none -- \
    run python validator/validator.py \
    --netuid {netuid} \
    --subtensor.network {network} \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --logging.trace \
    --logging.debug
```
