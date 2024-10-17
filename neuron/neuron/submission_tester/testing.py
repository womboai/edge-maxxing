import asyncio
import logging
from pathlib import Path
from statistics import mean
from collections.abc import Iterable
from io import BytesIO
from time import perf_counter

from .hash import load_image_hash, save_image_hash, GENERATION_TIME_DIFFERENCE_THRESHOLD
from .metrics import CheckpointBenchmark, MetricData, BaselineBenchmark
import imagehash
from PIL import Image

from neuron import (
    GenerationOutput,
    ModelRepositoryInfo,
    CURRENT_CONTEST,
    Key, OutputComparator,
)
from .vram_monitor import VRamMonitor
from pipelines import TextToImageRequest
from .inference_sandbox import InferenceSandbox, InvalidSubmissionError

SANDBOX_DIRECTORY = Path("/sandbox")
BASELINE_SANDBOX_DIRECTORY = Path("/baseline-sandbox")

logger = logging.getLogger(__name__)


def __generate_sync(
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


async def generate(
    container: InferenceSandbox,
    request: TextToImageRequest,
):
    loop = asyncio.get_running_loop()

    return await loop.run_in_executor(None, __generate_sync, container, request)


async def generate_baseline(
    inputs: list[TextToImageRequest],
    sandbox_directory: Path = BASELINE_SANDBOX_DIRECTORY,
    switch_user: bool = True,
) -> BaselineBenchmark:
    outputs: list[GenerationOutput] = []

    with InferenceSandbox(CURRENT_CONTEST.baseline_repository, True, sandbox_directory, switch_user) as sandbox:
        size = sandbox.model_size

        for index, request in enumerate(inputs):
            output = await generate(sandbox, request)

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


async def compare_checkpoints(
    submission: ModelRepositoryInfo,
    existing_benchmarks: Iterable[tuple[Key, CheckpointBenchmark | None]],
    inputs: list[TextToImageRequest],
    baseline: BaselineBenchmark,
    sandbox_directory: Path = SANDBOX_DIRECTORY,
    switch_user: bool = True,
) -> CheckpointBenchmark | None:
    logger.info("Generating model samples")

    outputs: list[GenerationOutput] = []

    try:
        with InferenceSandbox(submission, False, sandbox_directory, switch_user) as sandbox:
            size = sandbox.model_size

            image_hash = None

            f"Take {len(inputs)} samples, keeping track of how fast/accurate generations have been"
            for index, request in enumerate(inputs):
                logger.info(f"Sample {index + 1}, prompt {request.prompt} and seed {request.seed}")

                output = await generate(sandbox, request)

                if not image_hash:
                    with BytesIO(output.output) as data:
                        image_hash = imagehash.average_hash(Image.open(data))

                        image_hash_bytes = save_image_hash(image_hash)

                        match = next(
                            (
                                (key, existing_benchmark)
                                for key, existing_benchmark in existing_benchmarks
                                if (
                                    existing_benchmark and
                                    not (image_hash - load_image_hash(existing_benchmark.image_hash)) and
                                    abs(output.generation_time - existing_benchmark.model.generation_time) < GENERATION_TIME_DIFFERENCE_THRESHOLD
                                )
                            ),
                            None,
                        )

                        if match:
                            key, benchmark = match

                            logger.info(f"Submission {submission} marked as duplicate of hotkey {key}'s submission")

                            return benchmark

                logger.info(
                    f"Sample {index + 1} Generated\n"
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

    with CURRENT_CONTEST.output_comparator() as output_comparator:
        async def calculate_similarity(comparator: OutputComparator, baseline_output: GenerationOutput, optimized_output: GenerationOutput):
            loop = asyncio.get_running_loop()

            try:
                return await loop.run_in_executor(
                    None,
                    comparator,
                    baseline_output.output,
                    optimized_output.output,
                )
            except:
                logger.info(
                    f"Submission {submission.url}'s output couldn't be compared in similarity",
                    exc_info=True,
                )

                return 0.0

        average_similarity = mean(
            await calculate_similarity(output_comparator, baseline_output, output)
            for baseline_output, output in zip(baseline.outputs, outputs)
        )

    benchmark = CheckpointBenchmark(
        model=MetricData(
            generation_time=average_time,
            size=size,
            vram_used=vram_used,
            watts_used=watts_used,
        ),
        similarity_score=average_similarity,
        image_hash=image_hash_bytes,
    )

    logger.info(
        f"Tested {len(inputs)} Samples\n"
        f"Score: {benchmark.calculate_score(baseline.metric_data)}\n"
        f"Average Similarity: {average_similarity}\n"
        f"Average Generation Time: {average_time}s\n"
        f"Model Size: {size}b\n"
        f"Max VRAM Usage: {vram_used}b\n"
        f"Max Power Usage: {watts_used}W"
    )
    return benchmark
