import logging
from os import urandom
from statistics import median
from time import perf_counter

from base_validator.metrics import CheckpointBenchmark, MetricData
from pipelines import TextToImageRequest

from neuron import (
    Contest,
    GenerationOutput,
    generate_random_prompt,
    VRamMonitor,
    BENCHMARK_SAMPLE_COUNT,
    ModelRepositoryInfo,
)
from .inference_sandbox import InferenceSandbox, InvalidSubmissionError

logger = logging.getLogger(__name__)


def generate(contest: Contest, container: InferenceSandbox, prompt: str, seed: int) -> GenerationOutput:
    start_joules = contest.get_joules()
    vram_monitor = VRamMonitor(contest)
    start = perf_counter()

    output = container(
        TextToImageRequest(
            prompt=prompt,
            seed=seed,
        )
    )

    generation_time = perf_counter() - start
    joules_used = contest.get_joules() - start_joules
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


def compare_checkpoints(contest: Contest, submission: ModelRepositoryInfo) -> CheckpointBenchmark | None:
    logger.info("Generating model samples")

    outputs: list[GenerationOutput] = []

    try:
        with InferenceSandbox(submission, False) as sandbox:
            size = sandbox.model_size

            f"Take {BENCHMARK_SAMPLE_COUNT} samples, keeping track of how fast/accurate generations have been"
            for i in range(BENCHMARK_SAMPLE_COUNT):
                prompt = generate_random_prompt()
                seed = int.from_bytes(urandom(4), "little")

                logger.info(f"Sample {i + 1}, prompt {prompt} and seed {seed}")

                output = generate(
                    contest,
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

    similarity_scores = []

    with InferenceSandbox(contest.baseline_repository, True) as baseline_sandbox:
        baseline_size = baseline_sandbox.model_size

        for i, output in enumerate(outputs):
            baseline = generate(
                contest,
                baseline_sandbox,
                output.prompt,
                output.seed,
            )

            try:
                similarity = contest.compare_outputs(output.output, baseline.output)
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

            similarity_scores.append(similarity)

    baseline_average_time = sum(output.generation_time for output in baseline_outputs) / len(baseline_outputs)
    baseline_vram_used = max(output.vram_used for output in baseline_outputs)
    baseline_watts_used = max(output.watts_used for output in baseline_outputs)

    similarity_score = median(similarity_scores) * 0.75 + min(similarity_scores) * 0.25

    logger.info(f"Similarity Score: {similarity_score}")

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
        similarity_score=similarity_score,
    )
