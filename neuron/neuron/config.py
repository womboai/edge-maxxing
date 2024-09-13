from argparse import ArgumentParser
from typing import Callable

import neuron.bt as bt


def get_config(add_args: Callable[[ArgumentParser], None] | None = None):
    argument_parser = ArgumentParser()

    argument_parser.add_argument(
        "--netuid",
        dest="netuid",
        type=int,
    )

    bt.subtensor.add_args(argument_parser)
    bt.wallet.add_args(argument_parser)
    bt.logging.add_args(argument_parser)

    if add_args:
        add_args(argument_parser)

    config = bt.config(argument_parser)

    bt.logging(config=config.logging)

    return config
