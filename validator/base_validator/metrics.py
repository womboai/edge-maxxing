from enum import Enum
from math import sqrt

from pydantic import BaseModel

from neuron import Key, GenerationOutput
from pipelines import TextToImageRequest

SIMILARITY_SCORE_THRESHOLD = 0.8


class MetricData(BaseModel):
    generation_time: float
    size: int
    vram_used: float
    watts_used: float


class BaselineBenchmark(BaseModel):
    inputs: list[TextToImageRequest]
    outputs: list[GenerationOutput]
    metric_data: MetricData


class CheckpointBenchmark(BaseModel):
    model: MetricData
    similarity_score: float
    image_hash: bytes

    def calculate_score(self, baseline_metrics: MetricData) -> float:
        if self.similarity_score < SIMILARITY_SCORE_THRESHOLD:
            return 0.0

        scale = 1 / (1 - SIMILARITY_SCORE_THRESHOLD)
        similarity = sqrt((self.similarity_score - SIMILARITY_SCORE_THRESHOLD) * scale)
        return (baseline_metrics.generation_time - self.model.generation_time) * similarity


class BenchmarkState(Enum):
    NOT_STARTED = 0
    IN_PROGRESS = 1
    FINISHED = 2


class BenchmarkResults(BaseModel):
    state: BenchmarkState
    results: dict[Key, CheckpointBenchmark | None]
    baseline_metrics: MetricData | None
    average_benchmark_time: float | None
