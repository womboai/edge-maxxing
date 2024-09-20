from argparse import ArgumentParser
from typing import Callable

from fiber.constants import FINNEY_NETWORK


def get_config(add_args: Callable[[ArgumentParser], None] | None = None):
    argument_parser = ArgumentParser()

    argument_parser.add_argument(
        "--subtensor.chain_endpoint",
        type=str,
        required=False,
        help="Chain address",
        default=None,
    )

    argument_parser.add_argument(
        "--subtensor.network",
        type=str,
        required=False,
        help="Chain network",
        default=FINNEY_NETWORK,
    )

    argument_parser.add_argument("--wallet.name", type=str, required=False, help="Wallet name", default="default")
    argument_parser.add_argument("--wallet.hotkey", type=str, required=False, help="Hotkey name", default="default")

    if add_args:
        add_args(argument_parser)

    return vars(argument_parser.parse_args())
