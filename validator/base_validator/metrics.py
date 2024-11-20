from enum import Enum

from pydantic import BaseModel

from neuron import Key
from neuron.contest import CheckpointBenchmark, MetricData, ContestId


class BenchmarkState(Enum):
    NOT_STARTED = 0
    IN_PROGRESS = 1
    FINISHED = 2


class BenchmarkResults(BaseModel):
    state: BenchmarkState
    results: dict[Key, CheckpointBenchmark | None]
    invalid: dict[Key, str]
    baseline_metrics: MetricData | None
    average_benchmark_time: float | None


class ApiMetadata(BaseModel):
    version: str
    compatible_contests: list[ContestId]
