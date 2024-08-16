import bittensor as bt


class Metrics:
    metagraph: bt.metagraph

    baseline_averages: list[float]
    model_averages: list[float]
    similarity_averages: list[float]

    def __init__(self, metagraph: bt.metagraph):
        self.metagraph = metagraph
        self.clear()

    def clear(self):
        self.baseline_averages = [0.0] * self.metagraph.n.item()
        self.model_averages = [0.0] * self.metagraph.n.item()
        self.similarity_averages = [0.0] * self.metagraph.n.item()

    def reset(self, uid: int):
        self.baseline_averages[uid] = 0.0
        self.model_averages[uid] = 0.0
        self.similarity_averages[uid] = 0.0

    def update(self, uid: int, baseline: float, generation_time: float, similarity: float):
        self.baseline_averages[uid] = baseline
        self.model_averages[uid] = generation_time
        self.similarity_averages[uid] = similarity

    def resize(self, length: int):
        def resize_data(data: list[float]) -> list[float]:
            new_data = [0.0] * self.metagraph.n.item()
            new_data[:length] = data[:length]
            return new_data

        self.baseline_averages = resize_data(self.baseline_averages)
        self.model_averages = resize_data(self.model_averages)
        self.similarity_averages = resize_data(self.similarity_averages)

    def calculate_score(self, uid: int) -> float:
        return max(
            0.0,
            self.baseline_averages[uid] - self.model_averages[uid]
        ) * self.similarity_averages[uid]
