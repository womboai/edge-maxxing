from argparse import ArgumentParser
from typing import Callable


def get_config(add_args: Callable[[ArgumentParser], None] | None = None):
    import neuron.bt as bt
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
