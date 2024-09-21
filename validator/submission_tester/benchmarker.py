import asyncio
import logging
import time
import traceback
from asyncio import Task
from datetime import timedelta, datetime
from random import choice
from zoneinfo import ZoneInfo

from base_validator.metrics import CheckpointBenchmark
from neuron import CURRENT_CONTEST, CheckpointSubmission, Key
from submission_tester.testing import compare_checkpoints

logger = logging.getLogger(__name__)


class Benchmarker:
    submissions: dict[Key, CheckpointSubmission]
    benchmarks: dict[Key, CheckpointBenchmark | None]
    started: bool
    done: bool
    benchmark_task: Task | None
    submission_times: list[float]

    def __init__(self):
        self.submissions = {}
        self.benchmarks = {}
        self.started = False
        self.done = True
        self.submission_times = []

    def _benchmark_key(self, hotkey: Key):
        submission = self.submissions[hotkey]

        try:
            start_time = time.time()
            benchmark = compare_checkpoints(CURRENT_CONTEST, submission)
            self.submission_times.append(time.time() - start_time)
            self.benchmarks[hotkey] = benchmark
        except:
            traceback.print_exc()

    async def _benchmark_key_async(self, hotkey: Key):
        loop = asyncio.get_running_loop()

        await loop.run_in_executor(None, self._benchmark_key, hotkey)

    async def _start_benchmarking(self, submissions: dict[Key, CheckpointSubmission]):
        self.submissions = submissions
        self.benchmarks = {}
        self.submission_times = []
        self.started = True
        self.done = False

        while len(self.benchmarks) != len(self.submissions):
            hotkey = choice(list(self.submissions.keys() - self.benchmarks.keys()))

            try:
                await self._benchmark_key_async(hotkey)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                return

            logger.info(f"{len(self.benchmarks)}/{len(self.submissions)} submissions benchmarked")

            if self.submission_times:
                average_time = sum(self.submission_times) / len(self.submission_times)
                eta = int(average_time * (len(self.submissions) - len(self.benchmarks)))
                if eta > 0:
                    eta_date = datetime.now(tz=ZoneInfo("America/New_York")) + timedelta(seconds=eta)
                    eta_time = eta_date.strftime("%Y-%m-%d %H:%M:%S")

                    logger.info(f"ETA: {eta_time} (EST). Time remaining: {timedelta(seconds=eta)}")

        self.done = True

    async def start_benchmarking(self, submissions: dict[Key, CheckpointSubmission]):
        if not self.done and self.started:
            self.benchmark_task.cancel()

            self.submissions.update(submissions)

            for hotkey in submissions.keys():
                if hotkey in self.benchmarks:
                    del self.benchmarks[hotkey]

        self.benchmark_task = asyncio.create_task(self._start_benchmarking(submissions))
