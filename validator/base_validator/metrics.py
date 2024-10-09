from enum import Enum

from pydantic import BaseModel

from neuron import Key, ModelRepositoryInfo


class MetricData(BaseModel):
    generation_time: float
    size: int
    vram_used: float
    watts_used: float


class CheckpointBenchmark(BaseModel):
    baseline: MetricData
    model: MetricData
    similarity_score: float
    fingerprint: bytes

    def calculate_score(self) -> float:
        return (self.baseline.generation_time - self.model.generation_time) * self.similarity_score


class DuplicateBenchmark(BaseModel):
    copy_of: Key
    fingerprint: bytes


class BenchmarkingRequest(BaseModel):
    submissions: dict[Key, ModelRepositoryInfo]
    hash_prompt: str
    hash_seed: int


class BenchmarkState(Enum):
    NOT_STARTED = 0
    IN_PROGRESS = 1
    FINISHED = 2


class BenchmarkResults(BaseModel):
    state: BenchmarkState
    results: dict[Key, CheckpointBenchmark | DuplicateBenchmark | None]
    average_benchmark_time: float | None
