from logging import getLogger
from os import urandom
from time import perf_counter

import bittensor as bt
from coremltools import ComputeUnit
from diffusers import StableDiffusionXLPipeline
from pydantic import BaseModel
from python_coreml_stable_diffusion.pipeline import get_coreml_pipe, CoreMLStableDiffusionPipeline
from torch import Generator, cosine_similarity

from neuron import generate_random_prompt

logger = getLogger(__name__)

BASELINE_CHECKPOINT = "stabilityai/stable-diffusion-xl-base-1.0"
AVERAGE_TIME = 10.0
SPEC_VERSION = 20

SAMPLE_COUNT = 10


class CheckpointInfo(BaseModel):
    repository: str = BASELINE_CHECKPOINT
    average_time: float = AVERAGE_TIME
    spec_version: int = SPEC_VERSION


class CheckpointBenchmark:
    def __init__(self, average_time: float, average_similarity: float, failed: bool):
        self.average_time = average_time
        self.average_similarity = average_similarity
        self.failed = failed


def from_pretrained(name: str):
    base_pipeline = StableDiffusionXLPipeline.from_pretrained(name)

    pipeline = get_coreml_pipe(
        pytorch_pipe=base_pipeline,
        mlpackages_dir=mlpackages_dir,
        model_version=name,
        compute_unit=ComputeUnit.CPU_AND_GPU,
    )

    return pipeline


def get_checkpoint_info(subtensor: bt.subtensor, metagraph: bt.metagraph, uid: int) -> CheckpointInfo | None:
    return CheckpointInfo.parse_json(subtensor.get_commitment(metagraph.netuid, uid))


def compare_checkpoints(
    baseline: CoreMLStableDiffusionPipeline,
    miner_checkpoint: CoreMLStableDiffusionPipeline,
    reported_average_time: float,
) -> CheckpointBenchmark:
    if reported_average_time > AVERAGE_TIME:
        logger.info(f"Reported time is {reported_average_time}, which is worse than the baseline of {AVERAGE_TIME}")

        return CheckpointBenchmark(average_time=reported_average_time, average_similarity=1.0, failed=True)

    failed = False

    average_time = AVERAGE_TIME
    average_similarity = 1.0

    i = 0

    # Take {SAMPLE_COUNT} samples, keeping track of how fast/accurate generations have been
    for i in range(SAMPLE_COUNT):
        seed = int.from_bytes(urandom(4), "little")
        prompt = generate_random_prompt()

        base_generator = Generator().manual_seed(seed)
        checkpoint_generator = Generator().manual_seed(seed)
        output_type = "latent"

        logger.info(f"Sample {i}, prompt {prompt} and seed {seed}")

        base_output = baseline(
            prompt=prompt,
            generator=base_generator,
            output_type=output_type,
        ).images

        start = perf_counter()

        output = miner_checkpoint(
            prompt=prompt,
            generator=checkpoint_generator,
            output_type=output_type,
        ).images

        gen_time = perf_counter() - start

        similarity = pow(
            cosine_similarity(
                base_output.flatten(),
                output.flatten(),
                eps=1e-3,
                dim=0
            ).item() * 0.5 + 0.5,
            4,
        )

        logger.info(f"Sample {i} generated with generation time of {gen_time} and similarity {similarity}")

        generated = i
        remaining = SAMPLE_COUNT - generated

        average_time = (average_time * generated + gen_time) / (generated + 1)
        average_similarity = (average_similarity * generated + similarity) / (generated + 1)

        if average_time >= reported_average_time * 1.0625:
            # Too slow compared to reported speed, rank immediately based on current time
            failed = True
            break

        if average_time < AVERAGE_TIME:
            # So far, the average time is better than the baseline, so we can continue
            continue

        needed_time = (AVERAGE_TIME * SAMPLE_COUNT - generated * average_time) / remaining

        if needed_time < average_time / 2:
            # Needs double the current performance to beat the baseline,
            # thus we shouldn't waste compute testing farther
            failed = True
            break

        if average_similarity < 0.85:
            # Deviating too much from original quality
            failed = True
            break

    logger.info(
        f"Tested {i + 1} samples, "
        f"average similarity of {average_similarity}, "
        f"and speed of {average_time}"
    )

    return CheckpointBenchmark(average_time, average_similarity, failed)
