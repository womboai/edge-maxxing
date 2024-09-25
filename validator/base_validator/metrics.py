from enum import Enum

from pydantic import BaseModel

from neuron import Key


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
        return (self.baseline.generation_time - self.model.generation_time) * self.similarity_score


class BenchmarkState(Enum):
    NOT_STARTED = 0
    IN_PROGRESS = 1
    FINISHED = 2


class BenchmarkResults(BaseModel):
    state: BenchmarkState
    results: dict[Key, CheckpointBenchmark | None]
    average_benchmark_time: float
