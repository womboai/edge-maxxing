import bittensor as bt

from validator.submission_tester.testing import CheckpointBenchmark, MetricData


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
        self.__dict__.update(state)

    def __repr__(self):
        return f"Metrics(metrics={self.benchmarks})"
