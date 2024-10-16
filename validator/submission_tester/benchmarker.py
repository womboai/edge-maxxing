import asyncio
import logging
import traceback
from asyncio import Task
from datetime import timedelta, datetime
from random import choice
from threading import Lock
from time import perf_counter

from neuron.submission_tester import (
    CheckpointBenchmark,
    BaselineBenchmark,
    MetricData,
    compare_checkpoints,
    generate_baseline,
)

from neuron import (
    Key,
    ModelRepositoryInfo,
    TIMEZONE,
    random_inputs,
    InvalidSubmissionError,
)
from pipelines import TextToImageRequest

logger = logging.getLogger(__name__)


class Benchmarker:
    submissions: dict[Key, ModelRepositoryInfo]
    benchmarks: dict[Key, CheckpointBenchmark | None]
    invalid: dict[Key, str]
    baseline: BaselineBenchmark | None
    inputs: list[TextToImageRequest]
    started: bool
    done: bool
    start_timestamp: int
    submission_times: list[float]
    lock: Lock
    benchmark_task: Task | None

    def __init__(self):
        self.submissions = {}
        self.benchmarks = {}
        self.invalid = {}
        self.baseline = None
        self.inputs = []
        self.done = True
        self.start_timestamp = 0
        self.lock = Lock()
        self.submission_times = []
        self.benchmark_task = None

    async def _benchmark_key(self, hotkey: Key):
        submission = self.submissions[hotkey]

        start_time = perf_counter()
        try:
            self.benchmarks[hotkey] = await compare_checkpoints(
                submission,
                self.benchmarks.items(),
                self.inputs,
                self.baseline,
            )
        except InvalidSubmissionError as e:
            logger.error(f"Skipping invalid submission '{submission}': '{e}'")
            self.benchmarks[hotkey] = None
            self.invalid[hotkey] = str(e)
        except:
            traceback.print_exc()
        finally:
            self.submission_times.append(perf_counter() - start_time)

    async def _start_benchmarking(self, submissions: dict[Key, ModelRepositoryInfo]):
        self.submissions = submissions
        self.benchmarks = {}
        self.submission_times = []
        self.inputs = random_inputs()
        self.done = False

        if not self.baseline or self.baseline.inputs != self.inputs:
            logger.info("Generating baseline samples to compare")
            self.baseline = await generate_baseline(self.inputs)

        while len(self.benchmarks) != len(self.submissions):
            hotkey = choice(list(self.submissions.keys() - self.benchmarks.keys()))

            try:
                await self._benchmark_key(hotkey)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                return

            valid_submissions = len([benchmark for benchmark in self.benchmarks.values() if benchmark])
            logger.info(f"{len(self.benchmarks)}/{len(self.submissions)} submissions benchmarked. {valid_submissions} valid.")

            if self.submission_times:
                average_time = sum(self.submission_times) / len(self.submission_times)
                eta = int(average_time * (len(self.submissions) - len(self.benchmarks)))
                if eta > 0:
                    time_left = timedelta(seconds=eta)
                    eta_date = datetime.now(tz=TIMEZONE) + time_left
                    eta_time = eta_date.strftime("%Y-%m-%d %I:%M:%S %p")

                    logger.info(f"ETA: {eta_time} PST. Time remaining: {time_left}")

        self.done = True

    async def start_benchmarking(self, submissions: dict[Key, ModelRepositoryInfo]):
        benchmark_task = self.benchmark_task

        if not self.done and benchmark_task:
            benchmark_task.cancel()

            self.submissions = submissions
            self.benchmarks = {}

        self.benchmark_task = asyncio.create_task(self._start_benchmarking(submissions))

    def get_baseline_metrics(self) -> MetricData | None:
        return self.baseline.metric_data if self.baseline else None
