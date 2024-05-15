from logging import getLogger
from os import urandom
from os.path import isdir
from time import perf_counter
from typing import cast

import bittensor as bt
import torch
from bittensor.extrinsics.serving import get_metadata
from coremltools import ComputeUnit
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from python_coreml_stable_diffusion.pipeline import get_coreml_pipe, CoreMLStableDiffusionPipeline
from torch import Generator, cosine_similarity

from .pipeline import StableDiffusionXLMinimalPipeline, CoreMLPipelines
from .random_inputs import generate_random_prompt

logger = getLogger(__name__)

BASELINE_CHECKPOINT = "stabilityai/stable-diffusion-xl-base-1.0"
MLPACKAGES = "apple/coreml-stable-diffusion-xl-base"
CURRENT_CONTEST = "apple-silicon-stable-diffusion-xl-base-optimization"
AVERAGE_TIME = 10.0
SPEC_VERSION = 20

SAMPLE_COUNT = 10


class CheckpointInfo(BaseModel):
    repository: str = BASELINE_CHECKPOINT
    mlpackages: str = MLPACKAGES
    average_time: float = AVERAGE_TIME
    spec_version: int = SPEC_VERSION
    contest: str = CURRENT_CONTEST


class CheckpointBenchmark:
    def __init__(self, average_time: float, average_similarity: float, failed: bool):
        self.average_time = average_time
        self.average_similarity = average_similarity
        self.failed = failed


def from_pretrained(name: str, mlpackages: str, device: str) -> CoreMLPipelines:
    base_pipeline = StableDiffusionXLMinimalPipeline.from_pretrained(name).to(device)

    if isdir(mlpackages):
        coreml_dir = mlpackages
    else:
        coreml_dir = snapshot_download(mlpackages)

    compiled_dir = f"{coreml_dir}/compiled"

    pipeline = get_coreml_pipe(
        pytorch_pipe=base_pipeline,
        mlpackages_dir=compiled_dir,
        model_version="xl",
        compute_unit=ComputeUnit.CPU_AND_GPU.name,
        delete_original_pipe=False,
    )

    return CoreMLPipelines(base_pipeline, pipeline, coreml_dir)


def get_checkpoint_info(subtensor: bt.subtensor, metagraph: bt.metagraph, hotkey: str) -> CheckpointInfo | None:
    metadata = cast(dict[str, dict[str, list[dict[str, str]]]], get_metadata(subtensor, metagraph.netuid, hotkey))

    if not metadata:
        return None

    commitment = metadata["info"]["fields"][0]
    hex_data = commitment[list(commitment.keys())[0]][2:]

    info = CheckpointInfo.parse_raw(bytes.fromhex(hex_data).decode())

    if info.spec_version != SPEC_VERSION or info.contest != CURRENT_CONTEST:
        return None

    return info


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
            num_inference_steps=20,
        ).images

        start = perf_counter()

        output = miner_checkpoint(
            prompt=prompt,
            generator=checkpoint_generator,
            output_type=output_type,
            num_inference_steps=20,
        ).images

        gen_time = perf_counter() - start

        # noinspection PyUnboundLocalVariable
        similarity = pow(
            cosine_similarity(
                torch.from_numpy(base_output).flatten(),
                torch.from_numpy(output).flatten(),
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
