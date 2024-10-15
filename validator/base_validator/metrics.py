from enum import Enum
from math import sqrt

from pydantic import BaseModel

from neuron import Key

SIMILARITY_SCORE_THRESHOLD = 0.8


class MetricData(BaseModel):
    generation_time: float
    size: int
    vram_used: float
    watts_used: float


class CheckpointBenchmark(BaseModel):
    baseline: MetricData
    model: MetricData
    similarity_score: float
    image_hash: bytes

    def calculate_score(self) -> float:
        scale = 1 / (1 - SIMILARITY_SCORE_THRESHOLD)

        if self.similarity_score < SIMILARITY_SCORE_THRESHOLD:
            return 0.0

        similarity = sqrt((self.similarity_score - SIMILARITY_SCORE_THRESHOLD) * scale)

        return (self.baseline.generation_time / self.model.generation_time) * similarity


class BenchmarkState(Enum):
    NOT_STARTED = 0
    IN_PROGRESS = 1
    FINISHED = 2


class BenchmarkResults(BaseModel):
    state: BenchmarkState
    results: dict[Key, CheckpointBenchmark | None]
    average_benchmark_time: float | None
