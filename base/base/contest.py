from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from math import sqrt, prod, log
from typing import Callable

from pydantic import BaseModel

from .device import Device, CudaDevice, Gpu
from .output_comparator import OutputComparator, ImageOutputComparator

SIMILARITY_SCORE_THRESHOLD = 0.8


class BenchmarkState(IntEnum):
    NOT_STARTED = 0
    IN_PROGRESS = 1
    FINISHED = 2


class MetricType(IntEnum):
    GENERATION_TIME = 1
    SIZE = 2
    VRAM_USED = 3
    WATTS_USED = 4
    LOAD_TIME = 5
    RAM_USED = 6


class ContestId(IntEnum):
    FLUX_NVIDIA_4090 = 0
    SDXL_NEWDREAM_NVIDIA_4090 = 1


class RepositoryInfo(BaseModel):
    url: str
    revision: str


class Submission(BaseModel):
    repository_info: RepositoryInfo
    contest_id: ContestId
    block: int

    def contest(self) -> "Contest":
        return find_contest(self.contest_id)


class Metrics(BaseModel):
    generation_time: float
    size: int
    vram_used: float
    watts_used: float
    load_time: float
    ram_used: float


class Benchmark(BaseModel):
    metrics: Metrics
    average_similarity: float
    min_similarity: float


@dataclass
class Contest:
    id: ContestId
    device: Device
    output_comparator: Callable[[], OutputComparator]
    baseline_repository: RepositoryInfo

    def __init__(
        self,
        contest_id: ContestId,
        device: Device,
        output_comparator: Callable[[], OutputComparator],
        baseline_repository: RepositoryInfo,
    ):
        self.id = contest_id
        self.device = device
        self.output_comparator = output_comparator
        self.baseline_repository = baseline_repository

    def calculate_score(self, baseline: Metrics, benchmark: Benchmark) -> float:
        if benchmark.min_similarity < SIMILARITY_SCORE_THRESHOLD:
            return 0.0

        from .inputs_api import get_inputs_state
        metric_weights = get_inputs_state().get_metric_weights(self.id)

        similarity_scale = 1 / (1 - SIMILARITY_SCORE_THRESHOLD)
        similarity = sqrt((benchmark.average_similarity - SIMILARITY_SCORE_THRESHOLD) * similarity_scale)

        baseline_score = len(metric_weights)
        highest_score = prod(w + 1 for w in metric_weights.values())

        ratio = highest_score / baseline_score

        def calculate_improvement(baseline_value: float, benchmark_value: float, metric_type: MetricType) -> float:
            if baseline_value == 0:
                return 0
            relative_improvement = (baseline_value - benchmark_value) / baseline_value
            return relative_improvement * metric_weights.get(metric_type, 0) + 1

        score = prod([
            calculate_improvement(baseline.generation_time, benchmark.metrics.generation_time, MetricType.GENERATION_TIME),
            calculate_improvement(baseline.size, benchmark.metrics.size, MetricType.SIZE),
            calculate_improvement(baseline.vram_used, benchmark.metrics.vram_used, MetricType.VRAM_USED),
            calculate_improvement(baseline.watts_used, benchmark.metrics.watts_used, MetricType.WATTS_USED),
            calculate_improvement(baseline.load_time, benchmark.metrics.load_time, MetricType.LOAD_TIME),
            calculate_improvement(baseline.ram_used, benchmark.metrics.ram_used, MetricType.RAM_USED),
        ])

        n = (ratio + sqrt(ratio ** 2 - ratio * 4 + 4)) / 2

        score_base = ((n - 1) / baseline_score) * score + 1

        if score_base <= 0:
            return -1

        normalized_score = log(score_base, n) - 1

        if normalized_score < 0:
            normalized_score = normalized_score / similarity
        else:
            normalized_score = normalized_score * similarity

        return max(-1.0, min(1.0, normalized_score))


CUDA_4090_DEVICE = CudaDevice(gpu=Gpu.NVIDIA_RTX_4090)

CONTESTS = [
    Contest(
        contest_id=ContestId.FLUX_NVIDIA_4090,
        device=CUDA_4090_DEVICE,
        output_comparator=partial(ImageOutputComparator, CUDA_4090_DEVICE),
        baseline_repository=RepositoryInfo(url="https://github.com/womboai/flux-schnell-edge-inference", revision="fbfb8f0"),
    ),
    Contest(
        contest_id=ContestId.SDXL_NEWDREAM_NVIDIA_4090,
        device=CUDA_4090_DEVICE,
        output_comparator=partial(ImageOutputComparator, CUDA_4090_DEVICE),
        baseline_repository=RepositoryInfo(url="https://github.com/womboai/sdxl-newdream-20-inference", revision="1b3f9ea"),
    ),
]


def find_contest(contest_id: ContestId):
    for contest in CONTESTS:
        if contest.id != contest_id:
            continue

        return contest

    raise RuntimeError(f"Unknown contest ID requested {contest_id}")


def find_compatible_contests() -> list[ContestId]:
    return [contest.id for contest in CONTESTS if contest.device.is_compatible()]
