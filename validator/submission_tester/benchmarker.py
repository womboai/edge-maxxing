import asyncio
import logging
import traceback
from asyncio import Task
from datetime import timedelta, datetime
from random import choice
from threading import Lock
from time import perf_counter
from zoneinfo import ZoneInfo

from submission_tester.testing import compare_checkpoints

from neuron import CURRENT_CONTEST, Key, ModelRepositoryInfo
from validator.base_validator.metrics import CheckpointBenchmark

logger = logging.getLogger(__name__)


class Benchmarker:
    submissions: dict[Key, ModelRepositoryInfo]
    benchmarks: dict[Key, CheckpointBenchmark | None]
    started: bool
    done: bool
    start_timestamp: int
    submission_times: list[float]
    lock: Lock
    benchmark_task: Task | None

    def __init__(self):
        self.submissions = {}
        self.benchmarks = {}
        self.started = False
        self.done = True
        self.start_timestamp = 0
        self.lock = Lock()
        self.submission_times = []

    def _benchmark_key(self, hotkey: Key):
        submission = self.submissions[hotkey]

        try:
            start_time = perf_counter()
            benchmark = compare_checkpoints(CURRENT_CONTEST, submission)
            self.submission_times.append(perf_counter() - start_time)
            self.benchmarks[hotkey] = benchmark
        except:
            traceback.print_exc()

    async def _benchmark_key_async(self, hotkey: Key):
        loop = asyncio.get_running_loop()

        await loop.run_in_executor(None, self._benchmark_key, hotkey)

    async def _start_benchmarking(self, submissions: dict[Key, ModelRepositoryInfo]):
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

            valid_submissions = len([benchmark for benchmark in self.benchmarks.values() if benchmark])
            logger.info(f"{len(self.benchmarks)}/{len(self.submissions)} submissions benchmarked. {valid_submissions} valid.")

            if self.submission_times:
                average_time = sum(self.submission_times) / len(self.submission_times)
                eta = int(average_time * (len(self.submissions) - len(self.benchmarks)))
                if eta > 0:
                    time_left = timedelta(seconds=eta)
                    eta_date = datetime.now(tz=ZoneInfo("America/New_York")) + time_left
                    eta_time = eta_date.strftime("%Y-%m-%d %I:%M:%S %p")

                    logger.info(f"ETA: {eta_time} EST. Time remaining: {time_left}")

        self.done = True

    async def start_benchmarking(self, submissions: dict[Key, ModelRepositoryInfo]):
        if not self.done and self.started:
            self.benchmark_task.cancel()

            self.submissions.update(submissions)

            for hotkey in submissions.keys():
                if hotkey in self.benchmarks:
                    del self.benchmarks[hotkey]

        self.benchmark_task = asyncio.create_task(self._start_benchmarking(submissions))
