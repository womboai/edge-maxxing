import bittensor as bt

from neuron import CheckpointBenchmark


class Metrics:
    metagraph: bt.metagraph

    baseline_averages: list[float]
    model_averages: list[float]
    similarity_averages: list[float]
    sizes: list[int]
    vram_used: list[float]
    watts_used: list[float]

    def __init__(self, metagraph: bt.metagraph):
        self.metagraph = metagraph
        self.clear()

    def clear(self):
        self.baseline_averages = [0.0] * self.metagraph.n.item()
        self.model_averages = [0.0] * self.metagraph.n.item()
        self.similarity_averages = [0.0] * self.metagraph.n.item()
        self.sizes = [0] * self.metagraph.n.item()
        self.vram_used = [0.0] * self.metagraph.n.item()
        self.watts_used = [0.0] * self.metagraph.n.item()

    def reset(self, uid: int):
        self.baseline_averages[uid] = 0.0
        self.model_averages[uid] = 0.0
        self.similarity_averages[uid] = 0.0
        self.sizes[uid] = 0
        self.vram_used[uid] = 0.0
        self.watts_used[uid] = 0.0

    def update(self, uid: int, benchmark: CheckpointBenchmark):
        self.baseline_averages[uid] = benchmark.baseline_average
        self.model_averages[uid] = benchmark.average_time
        self.similarity_averages[uid] = benchmark.average_similarity
        self.sizes[uid] = benchmark.size
        self.vram_used[uid] = benchmark.vram_used
        self.watts_used[uid] = benchmark.watts_used

    def resize(self):
        def resize_data(data: list) -> list:
            new_data = [0.0] * self.metagraph.n.item()
            length = len(self.metagraph.hotkeys)
            new_data[:length] = data[:length]
            return new_data

        self.baseline_averages = resize_data(self.baseline_averages)
        self.model_averages = resize_data(self.model_averages)
        self.similarity_averages = resize_data(self.similarity_averages)
        self.sizes = resize_data(self.sizes)
        self.vram_used = resize_data(self.vram_used)
        self.watts_used = resize_data(self.watts_used)

    def calculate_score(self, uid: int) -> float:
        return max(
            0.0,
            self.baseline_averages[uid] - self.model_averages[uid]
        ) * self.similarity_averages[uid]

    def set_metagraph(self, metagraph: bt.metagraph):
        self.metagraph = metagraph

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["metagraph"]
        return state

    def __setstate__(self, state):
        self.clear()
        self.__dict__.update(state)

    def __repr__(self):
        return f"Metrics(model_averages={self.model_averages}, similarity_averages={self.similarity_averages})"
