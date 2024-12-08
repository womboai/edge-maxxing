# ruff: noqa: E402

import logging

logging.disable()

from base.config import get_config
from weight_setting.validator_args import add_args

from fiber.chain.chain_utils import load_hotkey_keypair


def log_extra_attributes():
    config = get_config(add_args)

    keypair = load_hotkey_keypair(
        wallet_name=config["wallet.name"],
        hotkey_name=config["wallet.hotkey"],
    )

    print(f"netuid={config['netuid']},neuron.hotkey={keypair.ss58_address}")


if __name__ == "__main__":
    log_extra_attributes()
