from math import sqrt

from pydantic import BaseModel

from pipelines import TextToImageRequest
from .. import GenerationOutput

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
    average_similarity: float
    min_similarity: float
    image_hash: bytes

    def calculate_score(self, baseline_metrics: MetricData) -> float:
        if self.min_similarity < SIMILARITY_SCORE_THRESHOLD:
            return 0.0

        scale = 1 / (1 - SIMILARITY_SCORE_THRESHOLD)
        similarity = sqrt((self.average_similarity - SIMILARITY_SCORE_THRESHOLD) * scale)
        return (baseline_metrics.generation_time - self.model.generation_time) * similarity
