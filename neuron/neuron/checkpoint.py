import traceback
from os import urandom
from struct import pack, unpack
from time import perf_counter
from typing import cast

import bittensor as bt
import torch
from bittensor.extrinsics.serving import get_metadata
from diffusers import DiffusionPipeline
from numpy import ndarray
from pydantic import BaseModel
from torch import Generator, cosine_similarity

from .contest import ContestId, CURRENT_CONTEST
from .random_inputs import generate_random_prompt

SPEC_VERSION = 20

SAMPLE_COUNT = 5


def float_bits(value: float):
    return unpack(">L", pack(">f", value))[0]


def float_from_bits(bits: int):
    return unpack(">f", pack(">L", bits))[0]


class CheckpointSubmission(BaseModel):
    repository: str = CURRENT_CONTEST.baseline_repository
    average_time: float
    spec_version: int = SPEC_VERSION
    contest: ContestId = CURRENT_CONTEST.id

    def to_bytes(self):
        data = bytearray()

        def write_bytes(byte_data: bytes):
            data.append(len(byte_data))
            data.extend(byte_data)

        def write_int(int_value: int):
            data.extend(int.to_bytes(int_value, 4, "big"))

        write_bytes(self.repository.encode())
        write_int(float_bits(self.average_time))
        write_int(self.spec_version)
        write_int(self.contest.value)

        if len(data) > 128:
            raise RuntimeError(f"CheckpointSubmission {self} is too large({len(data)}, can not exceed 128 bytes.")

        return bytes(data)

    @classmethod
    def from_bytes(cls, data: bytes):
        position = 0

        def read_bytes():
            nonlocal position
            length = data[position]
            position += 1

            value = data[position:position + length]
            position += length

            return value

        def read_int():
            nonlocal position

            value = int.from_bytes(data[position:position + 4], "big")
            position += 4

            return value

        repository = read_bytes().decode()
        average_time = float_from_bits(read_int())
        spec_version = read_int()
        contest_id = ContestId(read_int())

        return cls(
            repository=repository,
            average_time=average_time,
            spec_version=spec_version,
            contest=contest_id,
        )


class CheckpointBenchmark:
    def __init__(self, baseline_average: float, average_time: float, average_similarity: float, failed: bool):
        self.baseline_average = baseline_average
        self.average_time = average_time
        self.average_similarity = average_similarity
        self.failed = failed


def get_submission(subtensor: bt.subtensor, metagraph: bt.metagraph, hotkey: str) -> CheckpointSubmission | None:
    try:
        metadata = cast(dict[str, dict[str, list[dict[str, str]]]], get_metadata(subtensor, metagraph.netuid, hotkey))

        if not metadata:
            return None

        commitment = metadata["info"]["fields"][0]
        hex_data = commitment[list(commitment.keys())[0]][2:]

        info = CheckpointSubmission.from_bytes(bytes.fromhex(hex_data))

        if (
            info.spec_version != SPEC_VERSION or
            info.contest != CURRENT_CONTEST.id or
            info.repository == CURRENT_CONTEST.baseline_repository
        ):
            return None

        return info
    except Exception as e:
        bt.logging.error(f"Failed to get submission from miner {hotkey}", sufix=e)
        bt.logging.debug("Submission parsing error", sufix=traceback.format_exception(e))
        return None


def compare_checkpoints(
    baseline: DiffusionPipeline,
    miner_checkpoint: DiffusionPipeline,
    reported_average_time: float | None = None,
) -> CheckpointBenchmark:
    failed = False

    baseline_average = float("inf")
    average_time = float("inf")
    average_similarity = 1.0

    i = 0

    # Take {SAMPLE_COUNT} samples, keeping track of how fast/accurate generations have been
    for i in range(SAMPLE_COUNT):
        seed = int.from_bytes(urandom(4), "little")
        prompt = generate_random_prompt()

        base_generator = Generator().manual_seed(seed)
        checkpoint_generator = Generator().manual_seed(seed)
        output_type = "latent"

        bt.logging.info(f"Sample {i}, prompt {prompt} and seed {seed}")

        generated = i
        remaining = SAMPLE_COUNT - generated

        start = perf_counter()

        base_output = baseline(
            prompt=prompt,
            generator=base_generator,
            output_type=output_type,
            num_inference_steps=20,
        ).images

        baseline_average = (baseline_average * generated + perf_counter() - start) / (generated + 1)

        start = perf_counter()

        output = miner_checkpoint(
            prompt=prompt,
            generator=checkpoint_generator,
            output_type=output_type,
            num_inference_steps=20,
        ).images

        gen_time = perf_counter() - start

        if isinstance(base_output, ndarray):
            base_output = torch.from_numpy(base_output)

        if isinstance(output, ndarray):
            output = torch.from_numpy(output)

        similarity = (cosine_similarity(
            base_output.flatten(),
            output.flatten(),
            eps=1e-3,
            dim=0,
        ).item() * 0.5 + 0.5) ** 4

        bt.logging.info(f"Sample {i} generated with generation time of {gen_time} and similarity {similarity}")

        average_time = (average_time * generated + gen_time) / (generated + 1)
        average_similarity = (average_similarity * generated + similarity) / (generated + 1)

        if reported_average_time and average_time >= reported_average_time * 1.0625:
            # Too slow compared to reported speed, rank immediately based on current time
            failed = True
            break

        if average_time < baseline_average:
            # So far, the average time is better than the baseline, so we can continue
            continue

        needed_time = (baseline_average * SAMPLE_COUNT - generated * average_time) / remaining

        if needed_time < average_time / 2:
            # Needs double the current performance to beat the baseline,
            # thus we shouldn't waste compute testing farther
            failed = True
            break

        if average_similarity < 0.85:
            # Deviating too much from original quality
            failed = True
            break

    bt.logging.info(
        f"Tested {i + 1} samples, "
        f"average similarity of {average_similarity}, "
        f"and speed of {average_time}"
    )

    return CheckpointBenchmark(
        baseline_average,
        average_time,
        average_similarity,
        failed,
    )
