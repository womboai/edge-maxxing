import logging
from statistics import mean
from collections.abc import Iterable
from io import BytesIO
from time import perf_counter

from base_validator.hash import load_image_hash, save_image_hash, HASH_DIFFERENCE_THRESHOLD
from base_validator.metrics import CheckpointBenchmark, MetricData, BaselineBenchmark
import imagehash
from PIL import Image

from neuron import (
    GenerationOutput,
    VRamMonitor,
    ModelRepositoryInfo,
    CURRENT_CONTEST,
    Key,
)
from pipelines import TextToImageRequest
from .inference_sandbox import InferenceSandbox, InvalidSubmissionError

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


def generate_baseline(inputs: list[TextToImageRequest]) -> BaselineBenchmark:
    outputs: list[GenerationOutput] = []
    with InferenceSandbox(CURRENT_CONTEST.baseline_repository, True) as sandbox:
        size = sandbox.model_size

        for index, request in enumerate(inputs):
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
    existing_benchmarks: Iterable[tuple[Key, CheckpointBenchmark | None]],
    inputs: list[TextToImageRequest],
    baseline: BaselineBenchmark,
) -> CheckpointBenchmark | None:
    logger.info("Generating model samples")

    outputs: list[GenerationOutput] = []

    try:
        with InferenceSandbox(submission, False) as sandbox:
            size = sandbox.model_size

            image_hash = None

            f"Take {len(inputs)} samples, keeping track of how fast/accurate generations have been"
            for index, request in enumerate(inputs):
                logger.info(f"Sample {index + 1}, prompt {request.prompt} and seed {request.seed}")

                output = generate(sandbox, request)

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
                                    image_hash - load_image_hash(existing_benchmark.image_hash) < HASH_DIFFERENCE_THRESHOLD
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

    logger.info(
        f"Tested {len(inputs)} Samples\n"
        f"Average Generation Time: {average_time}s\n"
        f"Model Size: {size}b\n"
        f"Max VRAM Usage: {vram_used}b\n"
        f"Max Power Usage: {watts_used}W"
    )

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

    similarities = [
        calculate_similarity(baseline_output, output)
        for baseline_output, output in zip(baseline.outputs, outputs)
    ]

    average_similarity = mean(similarities)
    min_similarity = min(similarities)

    del comparator
    CURRENT_CONTEST.clear_cache()

    benchmark = CheckpointBenchmark(
        model=MetricData(
            generation_time=average_time,
            size=size,
            vram_used=vram_used,
            watts_used=watts_used,
        ),
        average_similarity=average_similarity,
        min_similarity=min_similarity,
        image_hash=image_hash_bytes,
    )

    logger.info(f"Average Similarity: {average_similarity}")
    logger.info(f"Score: {benchmark.calculate_score(baseline.metric_data)}")
    return benchmark
