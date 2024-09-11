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
        if self.baseline.generation_time < self.model.generation_time * 0.75:
            # Needs %33 faster than current performance to beat the baseline,
            return 0.0

        if self.similarity_score < 0.85:
            # Deviating too much from original quality
            return 0.0

        return max(0.0, self.baseline.generation_time - self.model.generation_time) * self.model.similarity_score


class BenchmarkState(Enum):
    NOT_STARTED = 0
    IN_PROGRESS = 1
    FINISHED = 2


class BenchmarkResults(BaseModel):
    state: BenchmarkState
    results: dict[Key, CheckpointBenchmark]
