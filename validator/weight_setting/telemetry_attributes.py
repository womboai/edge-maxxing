import logging
logging.disable()

import sys

from fiber.chain.chain_utils import load_hotkey_keypair

from neuron import get_config
from weight_setting.validator import Validator


def log_extra_attributes():
    config = get_config(Validator.add_extra_args)

    keypair = load_hotkey_keypair(
        wallet_name=config["wallet.name"],
        hotkey_name=config["wallet.hotkey"],
    )

    sys.stdout.write(f"netuid={config['netuid']},neuron.hotkey={keypair.ss58_address}")
    sys.stdout.flush()

if __name__ == "__main__":
    log_extra_attributes()