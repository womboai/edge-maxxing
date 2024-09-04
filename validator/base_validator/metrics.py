import bittensor as bt
from pydantic import BaseModel


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


class BenchmarkState(BaseModel):
    results: dict[str, CheckpointBenchmark] | None


class Metrics:
    metagraph: bt.metagraph

    benchmarks: list[CheckpointBenchmark | None]

    def __init__(self, metagraph: bt.metagraph):
        self.metagraph = metagraph
        self.clear()

    def clear(self):
        self.benchmarks = [None] * self.metagraph.n.item()

    def reset(self, uid: int):
        self.benchmarks[uid] = None

    def update(self, uid: int, benchmark: CheckpointBenchmark):
        self.benchmarks[uid] = benchmark

    def resize(self):
        new_data = [None] * self.metagraph.n.item()
        length = len(self.metagraph.hotkeys)
        new_data[:length] = self.benchmarks[:length]
        self.benchmarks = new_data

    def get_sorted_contestants(self) -> list[tuple[int, float]]:
        contestants = []
        for uid in range(self.metagraph.n.item()):
            metric_data = self.benchmarks[uid]
            if metric_data:
                contestants.append((uid, metric_data.calculate_score()))
        return sorted(contestants, key=lambda score: score[1])

    def set_metagraph(self, metagraph: bt.metagraph):
        self.metagraph = metagraph

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["metagraph"]
        return state

    def __setstate__(self, state):
        if "benchmarks" not in state:
            self.clear()

        self.__dict__.update(state)

    def __repr__(self):
        return f"Metrics(benchmarks={self.benchmarks})"
