from dataclasses import dataclass
from os import urandom
from time import perf_counter

import bittensor as bt
from pydantic import BaseModel

from neuron import Contest
from pipelines.pipelines.models import TextToImageRequest
from .inference_sandbox import InferenceSandbox
from .random_inputs import generate_random_prompt
from .vram_monitor import VRamMonitor

SAMPLE_COUNT = 5


class MetricData(BaseModel):
    generation_time: float
    size: int
    vram_used: float
    watts_used: float


class CheckpointBenchmark(BaseModel):
    baseline: MetricData
    model: MetricData
    similarity_score: float

    def calculate_score(self) -> float:
        if self.baseline.generation_time < self.model.generation_time * 0.75:
            # Needs %33 faster than current performance to beat the baseline,
            return 0.0

        if self.similarity_score < 0.85:
            # Deviating too much from original quality
            return 0.0

        return max(0.0, self.baseline.generation_time - self.model.generation_time) * self.model.similarity_score


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


def compare_checkpoints(contest: Contest, repository: str, revision: str) -> CheckpointBenchmark:
    with InferenceSandbox(contest.baseline_repository, contest.baseline_revision) as baseline_sandbox:
        bt.logging.info("Generating baseline samples to compare")

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

    with InferenceSandbox(repository, revision) as sandbox:
        size = sandbox.model_size

        i = 0

        f"Take {SAMPLE_COUNT} samples, keeping track of how fast/accurate generations have been"
        for i, baseline in enumerate(baseline_outputs):
            bt.logging.info(f"Sample {i}, prompt {baseline.prompt} and seed {baseline.seed}")

            generated = i
            remaining = SAMPLE_COUNT - generated

            generation = generate(
                contest,
                sandbox,
                baseline.prompt,
                baseline.seed,
            )

            similarity = contest.compare_outputs(baseline.output, generation.output)

            bt.logging.info(
                f"Sample {i} generated "
                f"with generation time of {generation.generation_time}, "
                f"and similarity {similarity}, "
                f"and VRAM usage of {generation.vram_used}, "
                f"and watts usage of {generation.watts_used}."
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

        bt.logging.info(
            f"Tested {i + 1} samples, "
            f"average similarity of {average_similarity}, "
            f"and speed of {average_time}, "
            f"and model size of {size}, "
            f"and VRAM usage of {vram_used}, "
            f"and watts usage of {watts_used}."
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
            size=size,
            vram_used=vram_used,
            watts_used=watts_used,
        ),
        similarity_score=average_similarity,
    )
