import bittensor as bt
from pydantic import BaseModel

from neuron import CheckpointBenchmark


class MetricData(BaseModel):
    generation_time: float
    similarity_score: float
    size: int
    vram_used: float
    watts_used: float


class Metrics:
    metagraph: bt.metagraph

    metrics: list[MetricData | None]

    def __init__(self, metagraph: bt.metagraph):
        self.metagraph = metagraph
        self.clear()

    def clear(self):
        self.metrics = [None] * self.metagraph.n.item()

    def reset(self, uid: int):
        self.metrics[uid] = None

    def update(self, uid: int, benchmark: CheckpointBenchmark):
        self.metrics[uid] = MetricData(
            baseline_average=benchmark.baseline_average,
            model_average=benchmark.average_time,
            similarity_average=benchmark.average_similarity,
            size=benchmark.size,
            vram_used=benchmark.vram_used,
            watts_used=benchmark.watts_used
        )

    def resize(self):
        new_data = [None] * self.metagraph.n.item()
        length = len(self.metagraph.hotkeys)
        new_data[:length] = self.metrics[:length]
        self.metrics = new_data

    def get_sorted_contestants(self) -> list[tuple[int, float]]:
        contestants = []
        for uid in range(self.metagraph.n.item()):
            metric_data = self.metrics[uid]
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

        # For backwards compatibility
        if "metrics" not in state:
            state["metrics"] = [
                MetricData(
                    baseline_average=state["baseline_averages"][i],
                    model_average=state["model_averages"][i],
                    similarity_average=state["similarity_averages"][i],
                    size=state["sizes"][i],
                    vram_used=state["vram_used"][i],
                    watts_used=state["watts_used"][i]
                )
                for i in range(len(state["model_averages"]))]

        self.__dict__.update(state)

    def __repr__(self):
        return f"Metrics(metrics={self.metrics})"
