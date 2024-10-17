from enum import Enum

from neuron import Key
from neuron.submission_tester.metrics import *


class BenchmarkState(Enum):
    NOT_STARTED = 0
    IN_PROGRESS = 1
    FINISHED = 2


class BenchmarkResults(BaseModel):
    state: BenchmarkState
    results: dict[Key, CheckpointBenchmark | None]
    baseline_metrics: MetricData | None
    average_benchmark_time: float | None
