import logging
import traceback
from concurrent.futures import CancelledError
from datetime import timedelta, datetime
from random import choice
from threading import Lock, Event, Thread
from time import perf_counter, sleep
from typing import cast

from neuron import (
    Key,
    ModelRepositoryInfo,
    TIMEZONE,
    random_inputs,
)
from neuron.submission_tester import (
    CheckpointBenchmark,
    BaselineBenchmark,
    MetricData,
    compare_checkpoints,
    generate_baseline,
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
    thread: Thread | None
    lock: Lock
    cancelled_event: Event

    def __init__(self):
        self.submissions = {}
        self.benchmarks = {}
        self.invalid = {}
        self.baseline = None
        self.inputs = []
        self.done = False
        self.start_timestamp = 0
        self.thread = None
        self.lock = Lock()
        self.submission_times = []
        self.cancelled_event = Event()

    def _benchmark_submission(self, hotkey: Key):
        submission = self.submissions[hotkey]

        start_time = perf_counter()

        try:
            self.benchmarks[hotkey] = compare_checkpoints(
                submission=submission,
                inputs=self.inputs,
                baseline=self.baseline,
                load_timeout=int(cast(MetricData, self.get_baseline_metrics()).load_time * 2),
                cancelled_event=self.cancelled_event,
            )
        except InvalidSubmissionError as e:
            logger.error(f"Skipping invalid submission '{submission}': '{e}'")
            self.benchmarks[hotkey] = None
            self.invalid[hotkey] = str(e)
        except CancelledError:
            logger.warning(f"Benchmarking was canceled while testing '{submission}'")
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
        self.baseline = None

        while not self.baseline and not self.cancelled_event.is_set():
            try:
                logger.info("Generating baseline samples to compare")
                self.baseline = generate_baseline(self.inputs, cancelled_event=self.cancelled_event)
            except CancelledError:
                logger.warning("Benchmarking was canceled while testing the baseline")
                return
            except Exception as e:
                logger.error("Failed to generate baseline samples, retrying in 10 minutes", exc_info=e)
                sleep(600)

        while len(self.benchmarks) != len(self.submissions) and not self.cancelled_event.is_set():
            hotkey = choice(list(self.submissions.keys() - self.benchmarks.keys()))
            self._benchmark_submission(hotkey)

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

    def start_benchmarking(self, submissions: dict[Key, ModelRepositoryInfo]):
        if not submissions:
            logger.warning("No submissions to benchmark")
            return

        logger.info(f"Starting benchmarking for {len(submissions)} submissions")

        if self.thread and self.thread.is_alive():
            logger.info("Attempting to cancel previous benchmarking")
            self.cancelled_event.set()
            self.thread.join(timeout=60)
            if self.thread.is_alive():
                logger.warning("Benchmarking was not stopped gracefully.")
            else:
                logger.info("Benchmarking was stopped gracefully.")

        self.cancelled_event.clear()
        self.thread = Thread(
            target=self._start_benchmarking,
            args=(submissions,),
        )
        self.thread.start()

    def get_baseline_metrics(self) -> MetricData | None:
        return self.baseline.metric_data if self.baseline else None
