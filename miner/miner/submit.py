from argparse import ArgumentParser
from os.path import isdir

import bittensor as bt
from diffusers import DiffusionPipeline
from huggingface_hub import upload_folder

from neuron import (
    CheckpointSubmission,
    get_submission,
    get_config,
    compare_checkpoints,
    CURRENT_CONTEST,
    make_submission,
)

MODEL_DIRECTORY = "model"


def optimize(pipeline: DiffusionPipeline) -> DiffusionPipeline:
    # Miners should change this function to optimize the pipeline
    return pipeline


def add_extra_args(argument_parser: ArgumentParser):
    argument_parser.add_argument(
        "--repository",
        type=str,
        help="The repository to push to",
    )

    argument_parser.add_argument(
        "--commit_message",
        type=str,
        help="The commit message",
        required=False,
    )

    argument_parser.add_argument(
        "--no_commit",
        dest="commit",
        action="store_false",
        help="Do not commit to huggingface",
    )

    argument_parser.add_argument(
        "--no_optimizations",
        dest="optimize",
        action="store_false",
        help="Submit/push without running the optimization step, if optimized outside the program",
    )

    argument_parser.set_defaults(commit=True, optimize=True)


def main():
    config = get_config(add_extra_args)

    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)
    wallet = bt.wallet(config=config)

    CURRENT_CONTEST.validate()

    if isdir(MODEL_DIRECTORY):
        repository = MODEL_DIRECTORY
        expected_average_time = None
    else:
        for uid in sorted(range(metagraph.n.item()), key=lambda i: metagraph.incentive[i].item(), reverse=True):
            submission = get_submission(subtensor, metagraph, metagraph.hotkeys[uid])

            if submission:
                info, _ = submission

                repository = info.image
                break
        else:
            repository = CURRENT_CONTEST.baseline_image

    if config.optimize:
        pipeline = optimize(CURRENT_CONTEST.load(repository))

        pipeline.save_pretrained(MODEL_DIRECTORY)

        repository = MODEL_DIRECTORY

    comparison = compare_checkpoints(CURRENT_CONTEST, repository)

    if config.commit:
        if comparison.failed:
            bt.logging.warning("Not pushing to huggingface as the checkpoint failed to beat the baseline.")

            return

        upload_folder(repo_id=config.image, folder_path=MODEL_DIRECTORY, commit_message=config.commit_message)
        bt.logging.info(f"Pushed to huggingface at {config.image}")

    checkpoint_info = CheckpointSubmission(
        repository=config.image,
        average_time=comparison.average_time,
    )

    make_submission(
        subtensor,
        metagraph,
        wallet,
        checkpoint_info,
    )

    bt.logging.info(f"Submitted {checkpoint_info} as the info for this miner")


if __name__ == '__main__':
    main()
