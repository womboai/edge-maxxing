from random import choice

from base_validator.metrics import CheckpointBenchmark

from neuron import CURRENT_CONTEST
from submission_tester.testing import compare_checkpoints


class Benchmarker:
    submissions: dict[str, tuple[str, str]]
    metrics: dict[str, CheckpointBenchmark]
    done: bool

    def __init__(self):
        self.submissions = {}
        self.metrics = {}
        self.done = True

    def start_benchmarking(self, submissions: dict[str, tuple[str, str]]):
        self.submissions = submissions
        self.metrics = {}
        self.done = False

        while len(self.metrics) != len(self.submissions):
            hotkey = choice(list(self.submissions.keys() - self.metrics.keys()))

            repository, revision = self.submissions[hotkey]

            self.metrics[hotkey] = compare_checkpoints(CURRENT_CONTEST, repository, revision)

        self.done = True
