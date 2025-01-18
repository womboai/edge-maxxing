from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from math import sqrt, prod, log, gcd
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

    def get_metric_weights(self):
        # Local import due to circular import
        from .inputs_api import get_inputs_state

        return get_inputs_state().get_metric_weights(self.id)

    def calculate_score(self, baseline: Metrics, benchmark: Benchmark) -> float:
        if benchmark.min_similarity < SIMILARITY_SCORE_THRESHOLD:
            return -1

        metric_weights = self.get_metric_weights()
        metric_gcd = gcd(*metric_weights.values())

        similarity_scale = 1 / (1 - SIMILARITY_SCORE_THRESHOLD)
        similarity = sqrt((benchmark.average_similarity - SIMILARITY_SCORE_THRESHOLD) * similarity_scale)

        def calculate_improvement(baseline_value: float, benchmark_value: float, metric_type: MetricType) -> float:
            metric_weight = metric_weights.get(metric_type)

            if not metric_weight:
                return 1

            return pow(baseline_value / benchmark_value, metric_weight / metric_gcd)

        score = prod([
            calculate_improvement(baseline.generation_time, benchmark.metrics.generation_time, MetricType.GENERATION_TIME),
            calculate_improvement(baseline.size, benchmark.metrics.size, MetricType.SIZE),
            calculate_improvement(baseline.vram_used, benchmark.metrics.vram_used, MetricType.VRAM_USED),
            calculate_improvement(baseline.watts_used, benchmark.metrics.watts_used, MetricType.WATTS_USED),
            calculate_improvement(baseline.load_time, benchmark.metrics.load_time, MetricType.LOAD_TIME),
            calculate_improvement(baseline.ram_used, benchmark.metrics.ram_used, MetricType.RAM_USED),
            similarity,
        ])

        score -= 1

        return score


CUDA_4090_DEVICE = CudaDevice(gpu=Gpu.NVIDIA_RTX_4090)

CONTESTS = [
    Contest(
        contest_id=ContestId.FLUX_NVIDIA_4090,
        device=CUDA_4090_DEVICE,
        output_comparator=partial(ImageOutputComparator, CUDA_4090_DEVICE),
        baseline_repository=RepositoryInfo(url="https://github.com/womboai/flux-schnell-edge-inference", revision="2cff82c"),
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
