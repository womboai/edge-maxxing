import bittensor as bt

from neuron.neuron import CURRENT_BASELINE_AVERAGE


class Metrics:
    metagraph: bt.metagraph

    model_averages: list[float]
    similarity_averages: list[float]

    def __init__(self, metagraph: bt.metagraph):
        self.metagraph = metagraph
        self.clear()

    def clear(self):
        self.model_averages = [0.0] * self.metagraph.n.item()
        self.similarity_averages = [0.0] * self.metagraph.n.item()

    def reset(self, uid: int):
        self.model_averages[uid] = 0.0
        self.similarity_averages[uid] = 0.0

    def update(self, uid: int, generation_time: float, similarity: float):
        self.model_averages[uid] = generation_time
        self.similarity_averages[uid] = similarity

    def resize(self):
        def resize_data(data: list[float]) -> list[float]:
            new_data = [0.0] * self.metagraph.n.item()
            length = len(self.metagraph.hotkeys)
            new_data[:length] = data[:length]
            return new_data

        self.model_averages = resize_data(self.model_averages)
        self.similarity_averages = resize_data(self.similarity_averages)

    def calculate_score(self, uid: int) -> float:
        return max(
            0.0,
            CURRENT_BASELINE_AVERAGE - self.model_averages[uid]
        ) * self.similarity_averages[uid]

    def set_metagraph(self, metagraph: bt.metagraph):
        self.metagraph = metagraph

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["metagraph"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
