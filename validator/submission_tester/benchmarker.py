from base_validator.metrics import CheckpointBenchmark


class Benchmarker:
    submissions: list[tuple[str, str]]
    metrics: dict[str, CheckpointBenchmark]
    done: bool

    def __init__(self):
        self.submissions = []
        self.metrics = {}
        self.done = True

    def start_benchmarking(self, submissions: list[tuple[str, str]]):
        self.submissions = submissions
        self.metrics = {}
        self.done = False
