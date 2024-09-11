import asyncio
import traceback
from asyncio import Task
from random import choice

from base_validator.metrics import CheckpointBenchmark

from neuron import CURRENT_CONTEST, CheckpointSubmission, Key
from submission_tester.testing import compare_checkpoints


class Benchmarker:
    submissions: dict[Key, CheckpointSubmission]
    benchmarks: dict[Key, CheckpointBenchmark | None]
    started: bool
    done: bool
    benchmark_task: Task | None

    def __init__(self):
        self.submissions = {}
        self.benchmarks = {}
        self.started = False
        self.done = True

    def _benchmark_key(self, hotkey: Key):
        submission = self.submissions[hotkey]

        try:
            benchmark = compare_checkpoints(CURRENT_CONTEST, submission)
            self.benchmarks[hotkey] = benchmark
        except:
            traceback.print_exc()
            self.benchmarks[hotkey] = None

    async def _benchmark_key_async(self, hotkey: Key):
        loop = asyncio.get_running_loop()

        loop.run_in_executor(None, self._benchmark_key, hotkey)

    async def _start_benchmarking(self, submissions: dict[Key, CheckpointSubmission]):
        self.submissions = submissions
        self.benchmarks = {}
        self.started = True
        self.done = False

        while len(self.benchmarks) != len(self.submissions):
            hotkey = choice(list(self.submissions.keys() - self.benchmarks.keys()))

            try:
                await self._benchmark_key_async(hotkey)
            except asyncio.TimeoutError:
                return

        self.done = True

    def start_benchmarking(self, submissions: dict[Key, CheckpointSubmission]):
        if not self.done and self.started:
            self.benchmark_task.cancel()

            self.submissions.update(submissions)

            for hotkey in submissions.keys():
                del self.benchmarks[hotkey]

        self.benchmark_task = asyncio.create_task(self._start_benchmarking(submissions))
