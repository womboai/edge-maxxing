from random import choice

from base_validator.metrics import CheckpointBenchmark

from neuron import CURRENT_CONTEST, CheckpointSubmission
from submission_tester.testing import compare_checkpoints


class Benchmarker:
    submissions: dict[str, CheckpointSubmission]
    metrics: dict[str, CheckpointBenchmark]
    done: bool

    def __init__(self):
        self.submissions = {}
        self.metrics = {}
        self.done = True

    def start_benchmarking(self, submissions: dict[str, CheckpointSubmission]):
        self.submissions = submissions
        self.metrics = {}
        self.done = False

        while len(self.metrics) != len(self.submissions):
            hotkey = choice(list(self.submissions.keys() - self.metrics.keys()))

            submission = self.submissions[hotkey]

            self.metrics[hotkey] = compare_checkpoints(CURRENT_CONTEST, submission.repository, submission.revision)

        self.done = True
