from argparse import ArgumentParser

from .wandb_manager import add_wandb_args


def add_args(argument_parser: ArgumentParser):
    argument_parser.add_argument(
        "--epoch_length",
        type=int,
        help="The default epoch length (how often we pull the metagraph, measured in 12 second blocks).",
        default=100,
    )

    argument_parser.add_argument(
        "--benchmarker_api",
        type=str,
        nargs="*",
        help="The API route to the validator benchmarking API.",
        required=True,
    )

    add_wandb_args(argument_parser)
