import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from os import urandom
from time import perf_counter
from typing import cast, Any

import bittensor as bt
import torch
from bittensor.extrinsics.serving import get_metadata, publish_metadata
from diffusers import DiffusionPipeline
from numpy import ndarray
from pydantic import BaseModel
from torch import Generator, cosine_similarity, Tensor

from .contest import ContestId, CURRENT_CONTEST, Contest
from .network_commitments import Encoder, Decoder
from .random_inputs import generate_random_prompt

SPEC_VERSION = 2
SAMPLE_COUNT = 5


@dataclass
class GenerationOutput:
    prompt: str
    seed: int
    output: Tensor
    generation_time: float
    vram_used: int
    watts_used: float


class CheckpointSubmission(BaseModel):
    repository: str = CURRENT_CONTEST.baseline_repository
    average_time: float
    contest: ContestId = CURRENT_CONTEST.id

    def encode(self, encoder: Encoder):
        encoder.write_str(self.repository)
        encoder.write_float(self.average_time)
        encoder.write_uint16(self.contest.value)

    @classmethod
    def decode(cls, decoder: Decoder):
        repository = decoder.read_str()
        average_time = decoder.read_float()
        contest_id = ContestId(decoder.read_uint16())

        return cls(
            repository=repository,
            average_time=average_time,
            contest=contest_id,
        )


@dataclass
class CheckpointBenchmark:
    baseline_average: float
    average_time: float
    average_similarity: float
    baseline_size: int
    size: int
    baseline_vram_used: int
    vram_used: int
    baseline_watts_used: float
    watts_used: float
    failed: bool


def make_submission(
    subtensor: bt.subtensor,
    metagraph: bt.metagraph,
    wallet: bt.wallet,
    submission: CheckpointSubmission,
):
    encoder = Encoder()

    encoder.write_uint16(SPEC_VERSION)

    submission.encode(encoder)

    data = encoder.finish()

    publish_metadata(
        subtensor,
        wallet,
        metagraph.netuid,
        f"Raw{len(data)}",
        data,
        wait_for_finalization=False,
    )


def get_submission(
    subtensor: bt.subtensor,
    metagraph: bt.metagraph,
    hotkey: str,
    block: int | None = None
) -> tuple[CheckpointSubmission, int] | None:
    try:
        metadata = cast(dict[str, Any], get_metadata(subtensor, metagraph.netuid, hotkey, block))

        if not metadata:
            return None

        block: int = metadata["block"]
        commitment: dict[str, str] = metadata["info"]["fields"][0]
        hex_data = commitment.values().__iter__().__next__()
        data = bytes.fromhex(hex_data[2:])
        decoder = Decoder(data)

        spec_version = decoder.read_uint16()

        if spec_version != SPEC_VERSION:
            return None

        info = CheckpointSubmission.decode(decoder)

        if info.contest != CURRENT_CONTEST.id or info.repository == CURRENT_CONTEST.baseline_repository:
            return None

        return info, block
    except Exception as e:
        bt.logging.error(f"Failed to get submission from miner {hotkey}, {e}")
        bt.logging.debug(f"Submission parsing error, {traceback.format_exception(e)}")
        return None


def generate(contest: Contest, pipeline: DiffusionPipeline, prompt: str, seed: int) -> GenerationOutput:
    start_vram = contest.get_vram_used(pipeline.device)
    start_watts = contest.get_watts_used(pipeline.device)
    start = perf_counter()

    output = pipeline(
        prompt=prompt,
        generator=Generator(pipeline.device).manual_seed(seed),
        output_type="latent",
        num_inference_steps=20,
    ).images

    generation_time = perf_counter() - start
    vram_used = contest.get_vram_used(pipeline.device) - start_vram
    watts_used = max(0, contest.get_watts_used(pipeline.device) - start_watts)

    if isinstance(output, ndarray):
        output = torch.from_numpy(output)

    return GenerationOutput(
        prompt=prompt,
        seed=seed,
        output=output,
        generation_time=generation_time,
        vram_used=vram_used,
        watts_used=watts_used,
    )


def compare_checkpoints(contest: Contest, repository: str) -> CheckpointBenchmark:
    failed = False

    baseline_pipeline = contest.load_baseline()

    baseline_pipeline(prompt="")

    bt.logging.info("Generating baseline samples to compare")

    baseline_outputs: list[GenerationOutput] = [
        generate(
            contest,
            baseline_pipeline,
            generate_random_prompt(),
            int.from_bytes(urandom(4), "little"),
        )
        for _ in range(SAMPLE_COUNT)
    ]

    del baseline_pipeline

    contest.empty_cache()

    baseline_average = sum([output.generation_time for output in baseline_outputs]) / len(baseline_outputs)
    baseline_size = contest.get_baseline_size()
    baseline_vram_used = int(sum([output.vram_used for output in baseline_outputs]) / len(baseline_outputs))
    baseline_watts_used = sum([output.watts_used for output in baseline_outputs]) / len(baseline_outputs)

    average_time = float("inf")
    average_similarity = 1.0

    with load_pipeline(contest, repository) as pipeline:

        size = contest.get_model_size()

        i = 0

        f"Take {SAMPLE_COUNT} samples, keeping track of how fast/accurate generations have been"
        for i, baseline in enumerate(baseline_outputs):
            bt.logging.info(f"Sample {i}, prompt {baseline.prompt} and seed {baseline.seed}")

            generated = i
            remaining = SAMPLE_COUNT - generated

            generation = generate(
                contest,
                pipeline,
                baseline.prompt,
                baseline.seed,
            )

            similarity = (cosine_similarity(
                baseline.output.flatten(),
                generation.output.flatten(),
                eps=1e-3,
                dim=0,
            ).item() * 0.5 + 0.5) ** 4

            bt.logging.info(
                f"Sample {i} generated "
                f"with generation time of {generation.generation_time} "
                f"and similarity {similarity}"
                f"and VRAM usage of {generation.vram_used}"
                f"and watts usage of {generation.watts_used}"
            )

            if generated:
                average_time = (average_time * generated + generation.generation_time) / (generated + 1)
                vram_used = (baseline.vram_used * generated + generation.vram_used) / (generated + 1)
                watts_used = (baseline.watts_used * generated + generation.watts_used) / (generated + 1)
            else:
                average_time = generation.generation_time
                vram_used = generation.vram_used
                watts_used = generation.watts_used

            average_similarity = (average_similarity * generated + similarity) / (generated + 1)

            if average_time < baseline_average * 1.0625:
                # So far, the average time is better than the baseline, so we can continue
                continue

            needed_time = (baseline_average * SAMPLE_COUNT - generated * average_time) / remaining

            if needed_time < average_time * 0.75:
                # Needs %33 faster than current performance to beat the baseline,
                # thus we shouldn't waste compute testing farther
                failed = True
                bt.logging.info("Current average is 75% of the baseline average")
                break

            if average_similarity < 0.85:
                # Deviating too much from original quality
                bt.logging.info("Too different from baseline, failing")
                failed = True
                break

        bt.logging.info(
            f"Tested {i + 1} samples, "
            f"average similarity of {average_similarity}, "
            f"and speed of {average_time}"
            f"and model size of {size}"
            f"and VRAM usage of {vram_used}"
            f"and watts usage of {watts_used}"
        )

    return CheckpointBenchmark(
        baseline_average,
        average_time,
        average_similarity,
        baseline_size,
        size,
        baseline_vram_used,
        vram_used,
        baseline_watts_used,
        watts_used,
        failed,
    )


@contextmanager
def load_pipeline(contest: Contest, repository: str):
    try:
        pipeline = contest.load(repository)
        yield pipeline
    finally:
        contest.delete_model_cache()
