import traceback
from random import choice

from base_validator.metrics import CheckpointBenchmark

from neuron import CURRENT_CONTEST, CheckpointSubmission
from submission_tester.testing import compare_checkpoints


class Benchmarker:
    submissions: dict[str, CheckpointSubmission]
    benchmarks: dict[str, CheckpointBenchmark | None]
    done: bool

    def __init__(self):
        self.submissions = {}
        self.benchmarks = {}
        self.done = True

    def start_benchmarking(self, submissions: dict[str, CheckpointSubmission]):
        self.submissions = submissions
        self.benchmarks = {}
        self.done = False

        while len(self.benchmarks) != len(self.submissions):
            hotkey = choice(list(self.submissions.keys() - self.benchmarks.keys()))

            submission = self.submissions[hotkey]

            try:
                benchmark = compare_checkpoints(CURRENT_CONTEST, submission.repository, submission.revision)
            except:
                traceback.print_exc()
                self.benchmarks[hotkey] = None
                continue

            self.benchmarks[hotkey] = benchmark

        self.done = True
