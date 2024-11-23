import os
from concurrent.futures import CancelledError
from pathlib import Path
from statistics import mean
from threading import Event
from time import perf_counter

from fiber.logging_utils import get_logger
from opentelemetry.instrumentation.aiohttp_server import tracer

from pipelines import TextToImageRequest
from . import InvalidSubmissionError
from .inference_sandbox import InferenceSandbox
from .metrics import CheckpointBenchmark, MetricData, BaselineBenchmark
from .vram_monitor import VRamMonitor
from .. import (
    GenerationOutput,
    ModelRepositoryInfo,
    CURRENT_CONTEST,
    OutputComparator,
)

SANDBOX_DIRECTORY = Path("/sandbox")
DEFAULT_LOAD_TIMEOUT = 1000
MIN_LOAD_TIMEOUT = 240

debug = int(os.getenv("VALIDATOR_DEBUG") or 0) > 0
logger = get_logger(__name__)


@tracer.start_as_current_span("generate")
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


@tracer.start_as_current_span("generate_baseline")
def generate_baseline(
    inputs: list[TextToImageRequest],
    sandbox_directory: Path = SANDBOX_DIRECTORY,
    switch_user: bool = True,
    cancelled_event: Event | None = None,
) -> BaselineBenchmark:
    outputs: list[GenerationOutput] = []

    start_vram = CURRENT_CONTEST.get_vram_used()
    with InferenceSandbox(
        repository_info=CURRENT_CONTEST.baseline_repository,
        baseline=True,
        sandbox_directory=sandbox_directory,
        switch_user=switch_user,
        load_timeout=DEFAULT_LOAD_TIMEOUT,
    ) as sandbox:
        size = sandbox.model_size

        for index, request in enumerate(inputs):
            if cancelled_event and cancelled_event.is_set():
                raise CancelledError()

            with tracer.start_span(f"generate_sample_{index + 1}") as sample_span:
                output = generate(sandbox, request)
                outputs.append(output)

                sample_span.set_attributes({
                    "generation.time_seconds": output.generation_time,
                    "generation.vram_used": output.vram_used,
                    "generation.watts_used": output.watts_used,
                })

    generation_time = mean(output.generation_time for output in outputs)
    vram_used = max(output.vram_used for output in outputs) - start_vram
    watts_used = max(output.watts_used for output in outputs)

    return BaselineBenchmark(
        inputs=inputs,
        outputs=outputs,
        metric_data=MetricData(
            generation_time=generation_time,
            size=size,
            vram_used=vram_used,
            watts_used=watts_used,
            load_time=sandbox.load_time,
        ),
    )


@tracer.start_as_current_span("compare_checkpoints")
def compare_checkpoints(
    submission: ModelRepositoryInfo,
    inputs: list[TextToImageRequest],
    baseline: BaselineBenchmark,
    sandbox_directory: Path = SANDBOX_DIRECTORY,
    switch_user: bool = True,
    load_timeout: int = DEFAULT_LOAD_TIMEOUT,
    cancelled_event: Event | None = None,
) -> CheckpointBenchmark | None:
    outputs: list[GenerationOutput] = []

    start_vram = CURRENT_CONTEST.get_vram_used()
    with InferenceSandbox(
            repository_info=submission,
            baseline=False,
            sandbox_directory=sandbox_directory,
            switch_user=switch_user,
            load_timeout=max(load_timeout, MIN_LOAD_TIMEOUT if not debug else DEFAULT_LOAD_TIMEOUT),
    ) as sandbox:
        size = sandbox.model_size

        try:
            for index, request in enumerate(inputs):
                logger.info(f"Sample {index + 1}, prompt {request.prompt} and seed {request.seed}")

                if cancelled_event and cancelled_event.is_set():
                    raise CancelledError()

                with tracer.start_span(f"generate_sample_{index + 1}") as sample_span:
                    output = generate(sandbox, request)
                    outputs.append(output)

                    sample_span.set_attributes({
                        "generation.time_seconds": output.generation_time,
                        "generation.vram_used": output.vram_used,
                        "generation.watts_used": output.watts_used,
                    })
        except (CancelledError, TimeoutError):
            raise
        except Exception as e:
            raise InvalidSubmissionError(f"Failed to run inference") from e

    average_time = sum(output.generation_time for output in outputs) / len(outputs)
    vram_used = max(output.vram_used for output in outputs) - start_vram
    watts_used = max(output.watts_used for output in outputs)

    with tracer.start_span("compare_outputs") as compare_span:
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
            load_time=sandbox.load_time,
        ),
        average_similarity=average_similarity,
        min_similarity=min_similarity,
    )

    compare_span.set_attributes({
        "benchmark.samples": len(inputs),
        "benchmark.score": benchmark.calculate_score(baseline.metric_data),
        "benchmark.average_similarity": average_similarity,
        "benchmark.min_similarity": min_similarity,
        "benchmark.average_generation_time": average_time,
        "benchmark.model_size": size,
        "benchmark.model_load_time": sandbox.load_time,
        "benchmark.vram_usage": vram_used,
        "benchmark.power_usage": watts_used,
    })

    logger.info(
        f"Tested {len(inputs)} Samples\n"
        f"Score: {benchmark.calculate_score(baseline.metric_data)}\n"
        f"Average Similarity: {average_similarity}\n"
        f"Min Similarity: {min_similarity}\n"
        f"Average Generation Time: {average_time}s\n"
        f"Model Size: {size}b\n"
        f"Model Load Time: {sandbox.load_time}s\n"
        f"Max VRAM Usage: {vram_used}b\n"
        f"Max Power Usage: {watts_used}W"
    )

    return benchmark
