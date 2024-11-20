import logging
from concurrent.futures import CancelledError
from dataclasses import dataclass
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
    Contest,
    ContestId,
    find_contest,
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


@dataclass
class ContestBenchmarks:
    submissions: dict[Key, ModelRepositoryInfo]
    benchmarks: dict[Key, CheckpointBenchmark | None]
    invalid: dict[Key, str]
    baseline: BaselineBenchmark | None


class Benchmarker:
    contest_benchmarks: dict[ContestId, ContestBenchmarks]
    inputs: list[TextToImageRequest]
    started: bool
    done: bool
    start_timestamp: int
    submission_times: list[float]
    thread: Thread | None
    lock: Lock
    cancelled_event: Event

    def __init__(self):
        self.contest_benchmarks = {}
        self.inputs = []
        self.done = False
        self.start_timestamp = 0
        self.thread = None
        self.lock = Lock()
        self.submission_times = []
        self.cancelled_event = Event()

    def _benchmark_submission(self, contest: Contest, hotkey: Key, submission: ModelRepositoryInfo):
        start_time = perf_counter()

        try:
            self.contest_benchmarks[contest.id].benchmarks[hotkey] = compare_checkpoints(
                contest=contest,
                submission=submission,
                inputs=self.inputs,
                baseline=self.baseline,
                load_timeout=int(cast(MetricData, self.get_baseline_metrics()).load_time * 2),
                cancelled_event=self.cancelled_event,
            )
        except InvalidSubmissionError as e:
            logger.error(f"Skipping invalid submission '{submission}': '{e}'")
            self.contest_benchmarks[contest.id].benchmarks[hotkey] = None
            self.contest_benchmarks[contest.id].invalid[hotkey] = str(e)
        except CancelledError:
            logger.warning(f"Benchmarking was canceled while testing '{submission}'")
            raise
        except Exception as e:
            logger.error(f"Exception occurred while testing '{submission}'", exc_info=e)
        finally:
            self.submission_times.append(perf_counter() - start_time)

    def _benchmark_submissions(self, contest: Contest, submissions: dict[Key, ModelRepositoryInfo]):
        baseline: BaselineBenchmark | None = None
        while not baseline and not self.cancelled_event.is_set():
            try:
                logger.info("Generating baseline samples to compare")
                baseline = generate_baseline(
                    contest=contest,
                    inputs=self.inputs,
                    cancelled_event=self.cancelled_event
                )
            except CancelledError:
                logger.warning("Benchmarking was canceled while testing the baseline")
                return
            except Exception as e:
                logger.error("Failed to generate baseline samples, retrying in 10 minutes", exc_info=e)
                sleep(600)

        self.contest_benchmarks[contest.id] = ContestBenchmarks(
            submissions=submissions,
            benchmarks={},
            invalid={},
            baseline=baseline,
        )

        while len(self.contest_benchmarks[contest.id].benchmarks) != len(submissions) and self.baseline and not self.cancelled_event.is_set():
            benchmarks = self.contest_benchmarks[contest.id].benchmarks
            hotkey = choice(list(submissions.keys() - benchmarks.keys()))
            self._benchmark_submission(contest, hotkey, submissions[hotkey])

            valid_submissions = len([benchmark for benchmark in benchmarks.values() if benchmark])
            logger.info(f"{len(benchmarks)}/{len(submissions)} submissions benchmarked. {valid_submissions} valid.")

            if self.submission_times:
                average_time = sum(self.submission_times) / len(self.submission_times)
                eta = int(average_time * (len(submissions) - len(benchmarks)))
                if eta > 0:
                    time_left = timedelta(seconds=eta)
                    eta_date = datetime.now(tz=TIMEZONE) + time_left
                    eta_time = eta_date.strftime("%Y-%m-%d %I:%M:%S %p")

                    logger.info(f"ETA: {eta_time} PST. Time remaining: {time_left}")

    def _start_benchmarking(self, contest_submissions: dict[ContestId, dict[Key, ModelRepositoryInfo]]):
        self.contest_benchmarks.clear()
        self.inputs = random_inputs()
        self.done = False
        self.baseline = None

        for contest_id, submissions in contest_submissions.items():
            logger.info(f"Working on contest {contest_id.name}")
            contest = find_contest(contest_id)
            self._benchmark_submissions(contest, submissions)

        self.done = True

    def start_benchmarking(self, contest_submissions: dict[ContestId, dict[Key, ModelRepositoryInfo]]):
        all_submissions: dict[Key, ModelRepositoryInfo] = {}
        for submissions in contest_submissions.values():
            all_submissions.update(submissions)

        if not all_submissions:
            logger.warning("No submissions to benchmark")
            return

        logger.info(f"Starting benchmarking for {len(all_submissions)} submissions")

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
            args=(contest_submissions,),
            daemon=True,
        )
        self.thread.start()

    def get_baseline_metrics(self) -> MetricData | None:
        return self.baseline.metric_data if self.baseline else None
