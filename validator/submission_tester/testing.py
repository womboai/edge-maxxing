import logging
from collections.abc import Iterable
from io import BytesIO
from time import perf_counter

import imagehash
from safetensors import numpy
from PIL import Image

from neuron import (
    GenerationOutput,
    generate_random_prompt,
    VRamMonitor,
    BENCHMARK_SAMPLE_COUNT,
    ModelRepositoryInfo, random_seed, CURRENT_CONTEST, Key,
)
from pipelines import TextToImageRequest
from .inference_sandbox import InferenceSandbox, InvalidSubmissionError
from ..base_validator.metrics import CheckpointBenchmark, MetricData, DuplicateBenchmark

logger = logging.getLogger(__name__)


def generate(
    container: InferenceSandbox,
    prompt: str,
    seed: int,
    width: int | None = None,
    height: int | None = None,
) -> GenerationOutput:
    start_joules = CURRENT_CONTEST.get_joules()
    vram_monitor = VRamMonitor(CURRENT_CONTEST)
    start = perf_counter()

    output = container(
        TextToImageRequest(
            prompt=prompt,
            seed=seed,
            width=width,
            height=height,
        )
    )

    generation_time = perf_counter() - start
    joules_used = CURRENT_CONTEST.get_joules() - start_joules
    watts_used = joules_used / generation_time
    vram_used = vram_monitor.complete()

    return GenerationOutput(
        prompt=prompt,
        seed=seed,
        output=output,
        generation_time=generation_time,
        vram_used=vram_used,
        watts_used=watts_used,
    )


def compare_checkpoints(
    submission: ModelRepositoryInfo,
    existing_benchmarks: Iterable[tuple[Key, CheckpointBenchmark | DuplicateBenchmark | None]],
    hash_prompt: str,
    hash_seed: int,
) -> CheckpointBenchmark | DuplicateBenchmark | None:
    logger.info("Generating model samples")

    outputs: list[GenerationOutput] = []

    try:
        with InferenceSandbox(submission, False) as sandbox:
            size = sandbox.model_size

            hash_output = generate(
                sandbox,
                hash_prompt,
                hash_seed,
                width=512,
                height=512,
            )

            def load_hash(existing_hash: bytes):
                return imagehash.ImageHash(numpy.load(existing_hash)["DEFAULT"])

            with BytesIO(hash_output.output) as data:
                image_hash = imagehash.average_hash(Image.open(data))

                image_hash_bytes = numpy.save(
                    {
                        "DEFAULT": image_hash.hash,
                    }
                )

                match = next(
                    (
                        existing_benchmark
                        for key, existing_benchmark in existing_benchmarks
                        if abs(image_hash - load_hash(existing_benchmark.fingerprint)) < 64
                    ),
                    None
                )

                if match:
                    key, _ = match

                    logger.info(f"Submission {submission} marked as duplicate of hotkey {key}'s submission")

                    return DuplicateBenchmark(copy_of=key, fingerprint=image_hash_bytes)

            f"Take {BENCHMARK_SAMPLE_COUNT} samples, keeping track of how fast/accurate generations have been"
            for i in range(BENCHMARK_SAMPLE_COUNT):
                prompt = generate_random_prompt()
                seed = random_seed()

                logger.info(f"Sample {i + 1}, prompt {prompt} and seed {seed}")

                output = generate(
                    sandbox,
                    prompt,
                    seed,
                )

                logger.info(
                    f"Sample {i} Generated\n"
                    f"Generation Time: {output.generation_time}s\n"
                    f"VRAM Usage: {output.vram_used}b\n"
                    f"Power Usage: {output.watts_used}W"
                )

                outputs.append(output)
    except InvalidSubmissionError as e:
        logger.error(f"Skipping invalid submission '{submission}': '{e}'")
        return None

    average_time = sum(output.generation_time for output in outputs) / len(outputs)
    vram_used = max(output.vram_used for output in outputs)
    watts_used = max(output.watts_used for output in outputs)

    logger.info(
        f"Tested {BENCHMARK_SAMPLE_COUNT} Samples\n"
        f"Average Generation Time: {average_time}s\n"
        f"Model Size: {size}b\n"
        f"Max VRAM Usage: {vram_used}b\n"
        f"Max Power Usage: {watts_used}W"
    )

    logger.info("Generating baseline samples to compare")

    baseline_outputs: list[GenerationOutput] = []

    average_similarity = 1.0

    with InferenceSandbox(CURRENT_CONTEST.baseline_repository, True) as baseline_sandbox:
        baseline_size = baseline_sandbox.model_size

        for i, output in enumerate(outputs):
            baseline = generate(
                baseline_sandbox,
                output.prompt,
                output.seed,
            )

            try:
                similarity = CURRENT_CONTEST.compare_outputs(output.output, baseline.output)
            except:
                logger.info(
                    f"Submission {submission.repository}'s output couldn't be compared in similarity",
                    exc_info=True,
                )

                similarity = 0.0

            logger.info(
                f"Baseline sample {i + 1} Generated\n"
                f"Generation Time: {baseline.generation_time}s\n"
                f"Similarity: {similarity}\n"
                f"VRAM Usage: {baseline.vram_used}b\n"
                f"Power Usage: {baseline.watts_used}W"
            )

            baseline_outputs.append(baseline)

            average_similarity = (average_similarity * i + similarity) / (i + 1)

    baseline_average_time = sum(output.generation_time for output in baseline_outputs) / len(baseline_outputs)
    baseline_vram_used = max(output.vram_used for output in baseline_outputs)
    baseline_watts_used = max(output.watts_used for output in baseline_outputs)

    logger.info(f"Average Similarity: {average_similarity}")

    return CheckpointBenchmark(
        baseline=MetricData(
            generation_time=baseline_average_time,
            size=baseline_size,
            vram_used=baseline_vram_used,
            watts_used=baseline_watts_used,
        ),
        model=MetricData(
            generation_time=average_time,
            size=size,
            vram_used=vram_used,
            watts_used=watts_used,
        ),
        similarity_score=average_similarity,
        fingerprint=image_hash_bytes,
    )
