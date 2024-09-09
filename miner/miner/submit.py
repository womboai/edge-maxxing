from argparse import ArgumentParser

import bittensor as bt
from git import Repo

from neuron import (
    CheckpointSubmission,
    get_config,
    make_submission,
)

MODEL_DIRECTORY = "model"


def add_extra_args(argument_parser: ArgumentParser):
    argument_parser.add_argument(
        "--repository",
        type=str,
        help="The repository to push to",
    )


def main():
    config = get_config(add_extra_args)

    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)
    wallet = bt.wallet(config=config)

    revision = Repo(MODEL_DIRECTORY).rev_parse("HEAD")
    checkpoint_info = CheckpointSubmission(repository=config.repository, revision=revision)

    make_submission(
        subtensor,
        metagraph,
        wallet,
        checkpoint_info,
    )

    bt.logging.info(f"Submitted {checkpoint_info} as the info for this miner")


if __name__ == '__main__':
    main()
