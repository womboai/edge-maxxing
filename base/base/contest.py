from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from math import sqrt
from typing import Callable

from pydantic import BaseModel

from .device import Device, CudaDevice, Gpu
from .output_comparator import OutputComparator, ImageOutputComparator

SIMILARITY_SCORE_THRESHOLD = 0.7


class MetricType(IntEnum):
    SIMILARITY_SCORE = 0
    GENERATION_TIME = 1
    SIZE = 2
    VRAM_USED = 3
    WATTS_USED = 4
    LOAD_TIME = 5


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
    metric_weights: dict[MetricType, int]

    def __init__(
        self,
        contest_id: ContestId,
        device: Device,
        output_comparator: Callable[[], OutputComparator],
        baseline_repository: RepositoryInfo,
        metric_weights: dict[MetricType, int]
    ):
        self.id = contest_id
        self.device = device
        self.output_comparator = output_comparator
        self.baseline_repository = baseline_repository
        self.metric_weights = metric_weights

    def calculate_score(self, baseline: Metrics, benchmark: Benchmark) -> float:
        if benchmark.min_similarity < SIMILARITY_SCORE_THRESHOLD:
            return 0.0

        total_weight = sum(self.metric_weights.values())

        scale = 1 / (1 - SIMILARITY_SCORE_THRESHOLD)
        similarity = sqrt((benchmark.average_similarity - SIMILARITY_SCORE_THRESHOLD) * scale)

        def normalize(baseline_value: float, benchmark_value: float, metric_type: MetricType) -> float:
            if baseline_value == 0:
                return 0
            relative_improvement = (baseline_value - benchmark_value) / baseline_value
            return (relative_improvement * self.metric_weights.get(metric_type, 0)) / total_weight

        score = sum([
            normalize(baseline.generation_time, benchmark.model.generation_time, MetricType.GENERATION_TIME),
            normalize(baseline.size, benchmark.model.size, MetricType.SIZE),
            normalize(baseline.vram_used, benchmark.model.vram_used, MetricType.VRAM_USED),
            normalize(baseline.watts_used, benchmark.model.watts_used, MetricType.WATTS_USED),
            normalize(baseline.load_time, benchmark.model.load_time, MetricType.LOAD_TIME)
        ])

        return score * similarity * self.metric_weights.get(MetricType.SIMILARITY_SCORE, 0) / total_weight


CUDA_4090_DEVICE = CudaDevice(gpu=Gpu.NVIDIA_RTX_4090)

CONTESTS = [
    Contest(
        contest_id=ContestId.FLUX_NVIDIA_4090,
        device=CUDA_4090_DEVICE,
        output_comparator=partial(ImageOutputComparator, CUDA_4090_DEVICE),
        baseline_repository=RepositoryInfo(url="https://github.com/womboai/flux-schnell-edge-inference", revision="fbfb8f0"),
        metric_weights={
            MetricType.SIMILARITY_SCORE: 3,
            MetricType.VRAM_USED: 3,
            MetricType.GENERATION_TIME: 1,
        }
    ),
    Contest(
        contest_id=ContestId.SDXL_NEWDREAM_NVIDIA_4090,
        device=CUDA_4090_DEVICE,
        output_comparator=partial(ImageOutputComparator, CUDA_4090_DEVICE),
        baseline_repository=RepositoryInfo(url="https://github.com/womboai/sdxl-newdream-20-inference", revision="1b3f9ea"),
        metric_weights={
            MetricType.SIMILARITY_SCORE: 1,
            MetricType.GENERATION_TIME: 1,
        }
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


ACTIVE_CONTESTS = [
    ContestId.FLUX_NVIDIA_4090
]