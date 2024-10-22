import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, CancelledError
from pathlib import Path
from statistics import mean
from threading import Event
from time import perf_counter

from neuron import (
    GenerationOutput,
    ModelRepositoryInfo,
    CURRENT_CONTEST,
    Key, OutputComparator,
)
from pipelines import TextToImageRequest
from .inference_sandbox import InferenceSandbox
from .metrics import CheckpointBenchmark, MetricData, BaselineBenchmark
from .vram_monitor import VRamMonitor

SANDBOX_DIRECTORY = Path("/sandbox")
BASELINE_SANDBOX_DIRECTORY = Path("/baseline-sandbox")

EXECUTOR = ThreadPoolExecutor(max_workers=2)

logger = logging.getLogger(__name__)


def generate(
    container: InferenceSandbox,
    request: TextToImageRequest,
) -> GenerationOutput:
    start_joules = CURRENT_CONTEST.get_joules()
    vram_monitor = VRamMonitor(CURRENT_CONTEST)
    start = perf_counter()

    output = container(request)

    generation_time = perf_counter() - start
    joules_used = CURRENT_CONTEST.get_joules() - start_joules
    watts_used = joules_used / generation_time
    vram_used = vram_monitor.complete()

    return GenerationOutput(
        output=output,
        generation_time=generation_time,
        vram_used=vram_used,
        watts_used=watts_used,
    )


def generate_baseline(
    inputs: list[TextToImageRequest],
    sandbox_directory: Path = BASELINE_SANDBOX_DIRECTORY,
    switch_user: bool = True,
    cache: bool = True,
    cancelled_event: Event | None = None,
) -> BaselineBenchmark:
    outputs: list[GenerationOutput] = []

    with InferenceSandbox(CURRENT_CONTEST.baseline_repository, True, sandbox_directory, switch_user, cache) as sandbox:
        size = sandbox.model_size

        for index, request in enumerate(inputs):
            if cancelled_event and cancelled_event.is_set():
                raise CancelledError()

            output = generate(sandbox, request)

            logger.info(
                f"Sample {index + 1} Generated\n"
                f"Generation Time: {output.generation_time}s\n"
                f"VRAM Usage: {output.vram_used}b\n"
                f"Power Usage: {output.watts_used}W"
            )

            outputs.append(output)

    generation_time = mean(output.generation_time for output in outputs)
    vram_used = max(output.vram_used for output in outputs)
    watts_used = max(output.watts_used for output in outputs)

    return BaselineBenchmark(
        inputs=inputs,
        outputs=outputs,
        metric_data=MetricData(
            generation_time=generation_time,
            size=size,
            vram_used=vram_used,
            watts_used=watts_used,
        ),
    )


def compare_checkpoints(
    submission: ModelRepositoryInfo,
    inputs: list[TextToImageRequest],
    baseline: BaselineBenchmark,
    sandbox_directory: Path = SANDBOX_DIRECTORY,
    switch_user: bool = True,
    cancelled_event: Event | None = None,
    cache: bool = False,
) -> CheckpointBenchmark | None:
    logger.info("Generating model samples")

    outputs: list[GenerationOutput] = []

    with InferenceSandbox(submission, False, sandbox_directory, switch_user, cache) as sandbox:
        size = sandbox.model_size

        f"Take {len(inputs)} samples, keeping track of how fast/accurate generations have been"
        for index, request in enumerate(inputs):
            logger.info(f"Sample {index + 1}, prompt {request.prompt} and seed {request.seed}")

            if cancelled_event and cancelled_event.is_set():
                raise CancelledError()

            output = generate(sandbox, request)

            logger.info(
                f"Sample {index + 1} Generated\n"
                f"Generation Time: {output.generation_time}s\n"
                f"VRAM Usage: {output.vram_used}b\n"
                f"Power Usage: {output.watts_used}W"
            )

            outputs.append(output)

    average_time = sum(output.generation_time for output in outputs) / len(outputs)
    vram_used = max(output.vram_used for output in outputs)
    watts_used = max(output.watts_used for output in outputs)

    with CURRENT_CONTEST.output_comparator() as output_comparator:
        def calculate_similarity(comparator: OutputComparator, baseline_output: GenerationOutput, optimized_output: GenerationOutput):
            try:
                if cancelled_event and cancelled_event.is_set():
                    raise CancelledError()

                return comparator(
                    baseline_output.output,
                    optimized_output.output,
                )
            except (CancelledError, TimeoutError):
                raise
            except:
                logger.info(
                    f"Submission {submission.url}'s output couldn't be compared in similarity",
                    exc_info=True,
                )

                return 0.0

        similarities = [
            calculate_similarity(output_comparator, baseline_output, output)
            for baseline_output, output in zip(baseline.outputs, outputs)
        ]

        average_similarity = mean(similarities)
        min_similarity = min(similarities)

    benchmark = CheckpointBenchmark(
        model=MetricData(
            generation_time=average_time,
            size=size,
            vram_used=vram_used,
            watts_used=watts_used,
        ),
        average_similarity=average_similarity,
        min_similarity=min_similarity,
    )

    logger.info(
        f"Tested {len(inputs)} Samples\n"
        f"Score: {benchmark.calculate_score(baseline.metric_data)}\n"
        f"Average Similarity: {average_similarity}\n"
        f"Min Similarity: {min_similarity}\n"
        f"Average Generation Time: {average_time}s\n"
        f"Model Size: {size}b\n"
        f"Max VRAM Usage: {vram_used}b\n"
        f"Max Power Usage: {watts_used}W"
    )

    return benchmark
