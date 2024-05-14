from argparse import ArgumentParser
from logging import getLogger
from os.path import isdir

import bittensor as bt
from python_coreml_stable_diffusion.pipeline import CoreMLStableDiffusionPipeline

from neuron import (
    AVERAGE_TIME,
    BASELINE_CHECKPOINT,
    CheckpointInfo,
    get_checkpoint_info,
    get_config,
    compare_checkpoints,
    from_pretrained,
)

logger = getLogger(__name__)

MODEL_DIRECTORY = "model"


def optimize(pipeline: CoreMLStableDiffusionPipeline) -> CoreMLStableDiffusionPipeline:
    # Miners should change this function to optimize the pipeline
    return pipeline


def add_extra_args(argument_parser: ArgumentParser):
    argument_parser.add_argument(
        "--repository",
        type=str,
        help="The repository to push to",
        required=True,
    )

    argument_parser.add_argument(
        "--commit_message",
        type=str,
        help="The commit message",
        required=False,
    )


def main():
    config = get_config(add_extra_args)

    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)
    wallet = bt.wallet(config=config)

    baseline_pipeline = from_pretrained(BASELINE_CHECKPOINT).to(config.device)

    if isdir(MODEL_DIRECTORY):
        pipeline = from_pretrained(MODEL_DIRECTORY).to(config.device)
        expected_average_time = AVERAGE_TIME
    else:
        for uid in sorted(range(metagraph.n.item()), key=lambda i: metagraph.incentive[i].item(), reverse=True):
            info = get_checkpoint_info(uid)

            if info:
                repository = info.repository
                expected_average_time = info.average_time
                break
        else:
            repository = BASELINE_CHECKPOINT
            expected_average_time = AVERAGE_TIME

        pipeline = from_pretrained(repository).to(config.device)

        pipeline.save_pretrained(MODEL_DIRECTORY)

    pipeline = optimize(pipeline)

    comparison = compare_checkpoints(baseline_pipeline, pipeline, expected_average_time)

    if comparison.failed:
        logger.warning("Not pushing to huggingface as the checkpoint failed to beat the baseline.")

        return

    if comparison.average_time > expected_average_time:
        logger.warning(
            f"Not pushing to huggingface as the average time {comparison.average_time} "
            f"is worse than the expected {expected_average_time}"
        )

    pipeline.push_to_hub(config.repository, config.commit_message)
    logger.info(f"Pushed to huggingface at {config.repository}")

    checkpoint_info = CheckpointInfo(
        repository=config.repository,
        average_time=comparison.average_time,
    )

    subtensor.commit(wallet, metagraph.netuid, checkpoint_info.json())
    logger.info(f"Submitted {checkpoint_info} as the info for this miner")


if __name__ == '__main__':
    main()
