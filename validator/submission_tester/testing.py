from dataclasses import dataclass
from os import urandom
from time import perf_counter

from neuron import Contest, CheckpointSubmission, bt
from pipelines.models import TextToImageRequest
from .inference_sandbox import InferenceSandbox
from .random_inputs import generate_random_prompt
from .vram_monitor import VRamMonitor
from base_validator.metrics import CheckpointBenchmark, MetricData

SAMPLE_COUNT = 5


@dataclass
class GenerationOutput:
    prompt: str
    seed: int
    output: bytes
    generation_time: float
    vram_used: float
    watts_used: float


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


def compare_checkpoints(contest: Contest, submission: CheckpointSubmission) -> CheckpointBenchmark:
    bt.logging.info("Generating baseline samples to compare")

    with InferenceSandbox(contest.baseline_repository, contest.baseline_revision, True) as baseline_sandbox:
        baseline_size = baseline_sandbox.model_size

        baseline_outputs: list[GenerationOutput] = [
            generate(
                contest,
                baseline_sandbox,
                generate_random_prompt(),
                int.from_bytes(urandom(4), "little"),
            )
            for _ in range(SAMPLE_COUNT)
        ]

    baseline_average = sum([output.generation_time for output in baseline_outputs]) / len(baseline_outputs)
    baseline_vram_used = sum([output.vram_used for output in baseline_outputs]) / len(baseline_outputs)
    baseline_watts_used = sum([output.watts_used for output in baseline_outputs]) / len(baseline_outputs)

    average_time = float("inf")
    average_similarity = 1.0

    bt.logging.info("Generating model samples")

    with InferenceSandbox(submission.repository, submission.revision, False) as sandbox:
        average_size = sandbox.model_size

        i = 0

        f"Take {SAMPLE_COUNT} samples, keeping track of how fast/accurate generations have been"
        for i, baseline in enumerate(baseline_outputs):
            bt.logging.info(f"Sample {i}, prompt {baseline.prompt} and seed {baseline.seed}")

            generated = i

            generation = generate(
                contest,
                sandbox,
                baseline.prompt,
                baseline.seed,
            )

            similarity = contest.compare_outputs(baseline.output, generation.output)

            bt.logging.info(
                f"Sample {i} Generated\n"
                f"Generation Time: {generation.generation_time}s\n"
                f"Similarity: {similarity}\n"
                f"VRAM Usage: {generation.vram_used}b\n"
                f"Power Usage: {generation.watts_used}W"
            )

            if generated:
                average_time = (average_time * generated + generation.generation_time) / (generated + 1)
                average_vram_used = (baseline.vram_used * generated + generation.vram_used) / (generated + 1)
                average_watts_used = (baseline.watts_used * generated + generation.watts_used) / (generated + 1)
            else:
                average_time = generation.generation_time
                average_vram_used = generation.vram_used
                average_watts_used = generation.watts_used

            average_similarity = (average_similarity * generated + similarity) / (generated + 1)

            if average_time < baseline_average * 1.0625:
                # So far, the average time is better than the baseline, so we can continue
                continue

        bt.logging.info(
            f"Tested {i + 1} Samples\n"
            f"Average Similarity: {average_similarity}\n"
            f"Average Generation Time: {average_time}s\n"
            f"Average Model Size: {average_size}b\n"
            f"Average VRAM Usage: {average_vram_used}b\n"
            f"Average Power Usage: {average_watts_used}W"
        )

    return CheckpointBenchmark(
        baseline=MetricData(
            generation_time=baseline_average,
            size=baseline_size,
            vram_used=baseline_vram_used,
            watts_used=baseline_watts_used,
        ),
        model=MetricData(
            generation_time=average_time,
            size=average_size,
            vram_used=average_vram_used,
            watts_used=average_watts_used,
        ),
        similarity_score=average_similarity,
    )
