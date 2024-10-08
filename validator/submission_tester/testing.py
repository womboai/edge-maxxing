import logging
from os import urandom
from statistics import mean
from collections.abc import Iterable
from io import BytesIO
from time import perf_counter

from base_validator.hash import load_image_hash, save_image_hash, HASH_DIFFERENCE_THRESHOLD
from base_validator.metrics import CheckpointBenchmark, MetricData
import imagehash
from PIL import Image

from neuron import (
    GenerationOutput,
    generate_random_prompt,
    VRamMonitor,
    BENCHMARK_SAMPLE_COUNT,
    ModelRepositoryInfo,
    InvalidSubmissionError,
    ModelRepositoryInfo,
    random_seed,
    CURRENT_CONTEST,
    Key,
)
from pipelines import TextToImageRequest
from .inference_sandbox import InferenceSandbox, InvalidSubmissionError

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
    existing_benchmarks: Iterable[tuple[Key, CheckpointBenchmark | None]],
    hash_prompt: str,
    hash_seed: int,
) -> CheckpointBenchmark | None:
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

            with BytesIO(hash_output.output) as data:
                image_hash = imagehash.average_hash(Image.open(data))

                image_hash_bytes = save_image_hash(image_hash)

                match = next(
                    (
                        (key, existing_benchmark)
                        for key, existing_benchmark in existing_benchmarks
                        if image_hash - load_image_hash(existing_benchmark.image_hash) < HASH_DIFFERENCE_THRESHOLD
                    ),
                    None,
                )

                if match:
                    key, benchmark = match

                    logger.info(f"Submission {submission} marked as duplicate of hotkey {key}'s submission")

                    return benchmark

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

    with InferenceSandbox(CURRENT_CONTEST.baseline_repository, True) as baseline_sandbox:
        baseline_size = baseline_sandbox.model_size

        for i, output in enumerate(outputs):
            baseline = generate(
                baseline_sandbox,
                output.prompt,
                output.seed,
            )

            logger.info(
                f"Baseline sample {i + 1} Generated\n"
                f"Generation Time: {baseline.generation_time}s\n"
                f"VRAM Usage: {baseline.vram_used}b\n"
                f"Power Usage: {baseline.watts_used}W"
            )

            baseline_outputs.append(baseline)

    comparator = CURRENT_CONTEST.output_comparator()

    def calculate_similarity(baseline_output: GenerationOutput, optimized_output: GenerationOutput):
        try:
            return comparator(baseline_output.output, optimized_output.output)
        except:
            logger.info(
                f"Submission {submission.url}'s output couldn't be compared in similarity",
                exc_info=True,
            )

            return 0.0

    average_similarity = mean(
        calculate_similarity(baseline_output, output)
        for baseline_output, output in zip(baseline_outputs, outputs)
    )

    baseline_average_time = mean(output.generation_time for output in baseline_outputs)
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
        image_hash=image_hash_bytes,
    )
