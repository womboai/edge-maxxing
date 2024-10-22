from enum import Enum

from neuron import Key
from neuron.submission_tester.metrics import *  # noqa
from neuron.submission_tester.metrics import CheckpointBenchmark, MetricData

from pydantic import BaseModel


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
