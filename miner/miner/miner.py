from argparse import ArgumentParser
from os import mkdir, rmdir
from os.path import isdir, join
from shutil import copytree, rmtree

import bittensor as bt
from bittensor.extrinsics.serving import publish_metadata
from huggingface_hub import upload_folder

from neuron import (
    BASELINE_CHECKPOINT,
    CheckpointSubmission,
    get_submission,
    get_config,
    compare_checkpoints,
    from_pretrained,
    CoreMLPipelines,
)

MODEL_DIRECTORY = "model"


def optimize(pipeline: CoreMLPipelines) -> CoreMLPipelines:
    # Miners should change this function to optimize the pipeline
    return pipeline


def add_extra_args(argument_parser: ArgumentParser):
    argument_parser.add_argument(
        "--repository",
        type=str,
        help="The repository to push  to",
    )

    argument_parser.add_argument(
        "--coreml_repository",
        type=str,
        help="The repository to push CoreML models(unet, vae, etc) to",
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

    argument_parser.set_defaults(commit=True)


def main():
    config = get_config(add_extra_args)

    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)
    wallet = bt.wallet(config=config)

    baseline_packages = from_pretrained(BASELINE_CHECKPOINT, config.device)
    baseline_pipeline = baseline_packages.coreml_sdxl_pipeline

    mlpackages_dir = join(MODEL_DIRECTORY, "mlpackages")

    if isdir(MODEL_DIRECTORY):
        pipelines = from_pretrained(MODEL_DIRECTORY, config.device)
        expected_average_time = None
    else:
        for uid in sorted(range(metagraph.n.item()), key=lambda i: metagraph.incentive[i].item(), reverse=True):
            info = get_submission(subtensor, metagraph, metagraph.hotkeys[uid])

            if info:
                repository = info.repository
                expected_average_time = info.average_time
                break
        else:
            repository = BASELINE_CHECKPOINT
            expected_average_time = None

        pipelines = from_pretrained(repository, config.device)

    pipelines = optimize(pipelines)

    pipeline = pipelines.coreml_sdxl_pipeline
    rmtree(MODEL_DIRECTORY, ignore_errors=True)
    pipelines.base_minimal_pipeline.save_pretrained(MODEL_DIRECTORY)

    mkdir(mlpackages_dir)
    copytree(pipelines.coreml_models_path, mlpackages_dir, dirs_exist_ok=True)

    comparison = compare_checkpoints(baseline_pipeline, pipeline, expected_average_time)

    if config.commit:
        if comparison.failed:
            bt.logging.warning("Not pushing to huggingface as the checkpoint failed to beat the baseline.")

            return

        if expected_average_time and comparison.average_time > expected_average_time:
            bt.logging.warning(
                f"Not pushing to huggingface as the average time {comparison.average_time} "
                f"is worse than the expected {expected_average_time}"
            )

            return

        upload_folder(config.repository, MODEL_DIRECTORY, commit_message=config.commit_message)
        bt.logging.info(f"Pushed to huggingface at {config.repository}")

    checkpoint_info = CheckpointSubmission(repository=config.repository, average_time=comparison.average_time)

    encoded = checkpoint_info.to_bytes()
    publish_metadata(subtensor, wallet, metagraph.netuid, f"Raw{len(encoded)}", encoded)

    bt.logging.info(f"Submitted {checkpoint_info} as the info for this miner")


if __name__ == '__main__':
    main()
