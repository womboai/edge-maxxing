import logging
import traceback
from concurrent.futures import Future, CancelledError, ThreadPoolExecutor
from datetime import timedelta, datetime
from random import choice
from threading import Lock, Event
from time import perf_counter

from neuron.submission_tester import (
    CheckpointBenchmark,
    BaselineBenchmark,
    MetricData,
    compare_checkpoints,
    generate_baseline,
    InvalidSubmissionError,
)

from neuron import (
    Key,
    ModelRepositoryInfo,
    TIMEZONE,
    random_inputs,
)
from pipelines import TextToImageRequest

logger = logging.getLogger(__name__)
EXECUTOR = ThreadPoolExecutor(1)


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
    benchmark_future: Future | None
    cancelled_event: Event

    def __init__(self):
        self.submissions = {}
        self.benchmarks = {}
        self.invalid = {}
        self.baseline = None
        self.inputs = []
        self.done = False
        self.start_timestamp = 0
        self.lock = Lock()
        self.submission_times = []
        self.benchmark_future = None
        self.cancelled_event = Event()

    def _benchmark_key(self, hotkey: Key):
        submission = self.submissions[hotkey]

        start_time = perf_counter()

        try:
            self.benchmarks[hotkey] = compare_checkpoints(
                submission=submission,
                inputs=self.inputs,
                baseline=self.baseline,
                load_timout=int(self.get_baseline_metrics().load_time * 2),
            )
        except InvalidSubmissionError as e:
            logger.error(f"Skipping invalid submission '{submission}': '{e}'")
            self.benchmarks[hotkey] = None
            self.invalid[hotkey] = str(e)
        except (CancelledError, TimeoutError):
            raise
        except:
            traceback.print_exc()
        finally:
            self.submission_times.append(perf_counter() - start_time)

    def _start_benchmarking(self, submissions: dict[Key, ModelRepositoryInfo]):
        self.submissions = submissions
        self.benchmarks.clear()
        self.invalid.clear()
        self.submission_times.clear()
        self.inputs = random_inputs()
        self.done = False

        try:
            if not self.baseline or self.baseline.inputs != self.inputs:
                logger.info("Generating baseline samples to compare")
                self.baseline = generate_baseline(self.inputs, cancelled_event=self.cancelled_event)

            while len(self.benchmarks) != len(self.submissions):
                hotkey = choice(list(self.submissions.keys() - self.benchmarks.keys()))

                self._benchmark_key(hotkey)

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
        except CancelledError:
            return

        self.done = True

    def start_benchmarking(self, submissions: dict[Key, ModelRepositoryInfo]):
        benchmark_future = self.benchmark_future

        if not self.done and benchmark_future:
            benchmark_future.cancel()
            self.cancelled_event.set()

            if not benchmark_future.cancelled():
                benchmark_future.result()

            self.submissions = submissions
            self.benchmarks = {}
            self.done = False

        self.benchmark_future = EXECUTOR.submit(self._start_benchmarking, submissions)

    def get_baseline_metrics(self) -> MetricData | None:
        return self.baseline.metric_data if self.baseline else None