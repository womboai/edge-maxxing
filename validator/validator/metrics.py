from dataclasses import dataclass

import bittensor as bt

from neuron import CheckpointBenchmark, CURRENT_CONTEST


@dataclass
class MetricData:
    baseline_average: float
    model_average: float
    similarity_average: float
    size: int
    vram_used: float
    watts_used: float

    def __repr__(self):
        return f"Metric(baseline_average={self.baseline_average}, model_average={self.model_average}, similarity_average={self.similarity_average}, size={self.size}, vram_used={self.vram_used}, watts_used={self.watts_used})"


def new_metric_data() -> MetricData:
    return MetricData(0.0, 0.0, 0.0, 0, 0.0, 0.0)


class Metrics:
    metagraph: bt.metagraph

    metrics: list[MetricData]

    def __init__(self, metagraph: bt.metagraph):
        self.metagraph = metagraph
        self.clear()

    def clear(self):
        self.metrics = [new_metric_data()] * self.metagraph.n.item()

    def reset(self, uid: int):
        self.metrics[uid] = new_metric_data()

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
        new_data = [new_metric_data()] * self.metagraph.n.item()
        length = len(self.metagraph.hotkeys)
        new_data[:length] = self.metrics[:length]
        self.metrics = new_data

    def calculate_score(self, uid: int) -> float:
        metric = self.metrics[uid]
        return max(
            0.0,
            (metric.baseline_average or CURRENT_CONTEST.baseline_average) - metric.model_average
        ) * metric.similarity_average

    def set_metagraph(self, metagraph: bt.metagraph):
        self.metagraph = metagraph

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["metagraph"]
        return state

    def __setstate__(self, state):

        # For backwards compatibility
        if not state["metrics"]:
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
